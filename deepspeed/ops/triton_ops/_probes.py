# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Probe library: one minimal risky usage per Triton kernel.

Every Triton call site validates its kernel through the shared gate
(``deepspeed.ops.triton_ops._triton.is_triton_kernel_supported``) with the probe
defined here. A probe runs the kernel's smallest representative usage - forward
and backward where the kernel has one - on the caller's device and dtype, so the
current backend answers for itself on first use and the answer is cached per
process. Callers that pass compile-relevant shape metadata get a probe that
compiles the same Triton variant production will run; probes that keep a fixed
smoke shape are support checks only. No platform keeps a blacklist, and a
backend that gains or fixes support flips the gate automatically.

Kernel modules are imported lazily inside each probe, so this module stays
import-light and free of import cycles. Probe names follow the module that owns
the kernel. Probe inputs are deterministic constants: a probe never draws
random numbers, so the caller's RNG state is never touched and the probe is
reproducible run over run.
"""

import torch


def _dispatched_class(feature: str, class_name: str, portable_module: str | None = None):
    """The kernel class the production path runs, through the per-hardware dispatch.

    A probe must exercise the class production actually runs - the gate
    protects that implementation - so the class resolves through the dispatch
    exactly like the dispatched public function does (the probe must not call
    the public function itself: that function checks the gate, and a probe
    running inside the gate would deadlock on the probe lock). Platforms
    without a variant get the portable class.
    """
    from deepspeed.ops.triton_ops._triton import resolve_triton_ops_impl

    return getattr(resolve_triton_ops_impl(feature, portable_module), class_name)


# Elementwise kernels cap their tiles here. A probe that reaches this many
# elements exercises the unmasked path and gives shape-derived variants the
# same constexpr tile production uses; a smaller real workload keeps its own
# extent so the compiled variant still matches.
_PROBE_TILE = 2048


def _probe_extent(n_elements, cap: int = _PROBE_TILE) -> int | None:
    """A small element count with the same Triton specialization as ``n_elements``.

    The count also stays in the same tile-size class: for a kernel whose tile is
    derived from the element count, the probe compiles the variant production
    will run. ``None`` means the caller passed no shape hint and the probe uses
    its default toy shape.
    """
    if n_elements is None or n_elements <= 0:
        return None
    if n_elements == 1:
        return 1
    if n_elements < 16:
        # Preserve the non-16-divisible specialization without becoming the
        # constexpr-1 variant.
        return n_elements
    tile = min(cap, 1 << (n_elements - 1).bit_length())
    return tile if n_elements % 16 == 0 else max(1, tile - 1)


def _probe_rows(n_rows) -> int | None:
    """A small row count with the same 16-divisibility specialization as ``n_rows``."""
    if n_rows is None or n_rows <= 0:
        return None
    if n_rows == 1:
        return 1
    return 16 if n_rows % 16 == 0 else 15


def _probe_heads(n_heads) -> int | None:
    """A small head count in the same specialization class, kept cheap to allocate."""
    if n_heads is None or n_heads <= 0:
        return None
    if n_heads == 1:
        return 1
    return 16 if n_heads % 16 == 0 else 2


def probe_extent_key(n_elements) -> str:
    """The cache-key component for a shape-derived elementwise tile."""
    return f"n={_probe_extent(n_elements)}"


def probe_rows_key(n_rows) -> str:
    """The cache-key component for a shape-derived row count."""
    return f"rows={_probe_rows(n_rows)}"


def probe_heads_key(n_heads) -> str:
    """The cache-key component for a shape-derived head count."""
    return f"heads={_probe_heads(n_heads)}"


def probe_packed_key(inter, n_rows) -> str:
    """The cache-key component for the packed SwiGLU constexpr and output class."""
    out_elements = max(1, int(n_rows)) * int(inter) if inter and n_rows else None
    return f"inter={inter}:{probe_extent_key(out_elements)}"


def _portable_rotary_probe(seq_len) -> tuple[int, str]:
    """Portable RoPE fixes BLOCK_S; only ``seq_len``'s scalar specialization matters."""
    if seq_len is None or seq_len <= 0:
        return 8, "seq=default"
    if seq_len == 1:
        return 1, "seq=one"
    return (16, "seq=div16") if seq_len % 16 == 0 else (15, "seq=other")


def _rotary_probe_shape(seq_len, head_dim, n_heads, device) -> tuple[int, str]:
    """The probe sequence length and key component for this platform's RoPE variant.

    A platform whose sequence tile is derived from ``seq_len`` may expose
    ``probe_rotary_shape(seq_len, head_dim, n_heads, device)`` on its variant
    module; the portable kernel keeps its fixed tile and only needs the scalar
    specialization class.
    """
    from deepspeed.ops.triton_ops._triton import resolve_triton_ops_impl

    hook = getattr(resolve_triton_ops_impl("fused_rotary_pos_emb"), "probe_rotary_shape", None)
    if hook is not None:
        return hook(seq_len, head_dim, n_heads, device)
    return _portable_rotary_probe(seq_len)


def probe_rotary_key(head_dim, seq_len, n_heads, device) -> str:
    """The cache-key component for RoPE's head width, sequence variant, and head class."""
    _, seq_key = _rotary_probe_shape(seq_len, head_dim, n_heads, device)
    return f"h={head_dim}:{seq_key}:{probe_heads_key(n_heads)}"


def _probe_tensor(size, dtype, device):
    """A deterministic probe input in [0, 1) with no RNG involvement.

    A probe's job is to make the backend compile and run the kernel's risky
    usage; the input values do not carry the risk (a compile crash or a device
    fault fires for any values), so no randomness is needed. Deterministic
    inputs mean the caller's generator is never touched - some backends cannot
    reliably save/restore their generator state - and the probe runs the same
    values every time.
    """
    numel = torch.Size(size).numel()
    return torch.arange(numel, dtype=torch.float32, device=device).div_(numel).reshape(size).to(dtype=dtype)


def _fwd_bwd(forward, device):
    """Wrap a forward callable in a grad-enabled, device-pinned forward + backward.

    The device pin makes the probe launch on the tensor's device rather than the
    caller's current device. The probe's inputs are deterministic (see
    ``_probe_tensor``), so the caller's RNG state is never touched.
    """

    def run():
        from contextlib import nullcontext

        from deepspeed.ops.triton_ops._triton import current_device

        # current_device takes a tensor or an index; pass the index so a concrete
        # torch.device works directly and index-less devices skip the switch.
        device_ctx = current_device(device.index) if device.index is not None else nullcontext()
        with torch.enable_grad(), device_ctx:
            y = forward()
            y.sum().backward()
            return y.detach()

    return run


def swiglu(dtype, device, n_elements=None):
    _SwiGLUFn = _dispatched_class("swiglu_triton", "_SwiGLUFn")
    extent = _probe_extent(n_elements)
    shape = (1, extent) if extent is not None else (32, 16)

    def forward():
        gate = _probe_tensor(shape, dtype, device).requires_grad_()
        up = _probe_tensor(shape, dtype, device).requires_grad_()
        return _SwiGLUFn.apply(gate, up)

    return _fwd_bwd(forward, device)


def _swiglu_packed_fn():
    """The packed SwiGLU class through the per-hardware dispatch (see ``_dispatched_class``)."""
    return _dispatched_class("swiglu_triton", "_SwiGLUPackedFn")


def _packed_probe_rows(n_rows, inter: int) -> int:
    """Rows whose output keeps the real ``n_rows * inter`` specialization and tile."""
    if inter <= 0:
        return 1
    out_elements = max(1, int(n_rows)) * int(inter) if n_rows else _PROBE_TILE
    target = _probe_extent(out_elements) or _PROBE_TILE
    if inter % 16 == 0:
        # Every row count keeps the output in the divisible-by-16 class.
        return max(1, (target + inter - 1) // inter)
    rows = 16 if n_rows is None or n_rows % 16 == 0 else n_rows % 16
    while rows * inter < target:
        rows += 16
    return max(1, rows)


def swiglu_packed(dtype, device, inter=None, n_rows=None):
    _SwiGLUPackedFn = _swiglu_packed_fn()
    if inter is None or inter <= 0:
        shape = (32, 32)
    else:
        shape = (_packed_probe_rows(n_rows, inter), 2 * inter)

    def forward():
        x = _probe_tensor(shape, dtype, device).requires_grad_()
        return _SwiGLUPackedFn.apply(x)

    return _fwd_bwd(forward, device)


def group_gemm(dtype, device):
    """A support smoke test, not a warm-up: autotune's key is the real K/N/group count."""
    from deepspeed.ops.triton_ops.group_gemm_triton import _GroupGemmFn

    def forward():
        a = _probe_tensor((8, 16), dtype, device).requires_grad_()
        b = _probe_tensor((2, 16, 32), dtype, device).requires_grad_()
        offs = torch.tensor([0, 4, 8], dtype=torch.int32, device=device)
        return _GroupGemmFn.apply(a, b, offs, False)

    return _fwd_bwd(forward, device)


def fused_rms_norm(dtype, device, n_cols=None, n_rows=None):
    _FusedRMSNorm = _dispatched_class("fused_rms_norm", "_FusedRMSNorm")
    width = n_cols if n_cols is not None and n_cols > 0 else 256
    rows = _probe_rows(n_rows) or 2

    def forward():
        hidden = _probe_tensor((rows, width), dtype, device).requires_grad_()
        weight = _probe_tensor((width, ), dtype, device).requires_grad_()
        return _FusedRMSNorm.apply(hidden, weight, 1e-6)

    return _fwd_bwd(forward, device)


def fused_rotary(dtype, device, head_dim=None, seq_len=None, n_heads=None):
    _FusedRotaryPosEmb = _dispatched_class("fused_rotary_pos_emb", "_FusedRotaryPosEmb")
    width = head_dim if head_dim is not None and head_dim > 0 else 64
    # A platform with a shape-derived sequence tile picks a small representative
    # sequence that selects the same tile as production; the portable kernel
    # only needs the scalar specialization class.
    seq, _ = _rotary_probe_shape(seq_len, width, n_heads, device)
    heads = _probe_heads(n_heads) or 4

    def forward():
        q = _probe_tensor((1, heads, seq, width), dtype, device).requires_grad_()
        k = _probe_tensor((1, heads, seq, width), dtype, device).requires_grad_()
        cos = _probe_tensor((1, seq, width), dtype, device)
        sin = _probe_tensor((1, seq, width), dtype, device)
        q_out, k_out = _FusedRotaryPosEmb.apply(q, k, cos, sin)
        return q_out + k_out

    return _fwd_bwd(forward, device)


def fused_weighted_restore(dtype, device, hidden=None, top_k=None):
    from deepspeed.ops.triton_ops.autoep_fused_token_ops import fused_weighted_restore

    width = hidden if hidden is not None and hidden > 0 else 64
    k = top_k if top_k is not None and top_k > 0 else 2

    def forward():
        bsz, seqlen = 1, 1
        combined_rows = _probe_tensor((bsz * seqlen * k, width), dtype, device).requires_grad_()
        top_scores = _probe_tensor((bsz * seqlen, k), torch.float32, device).requires_grad_()
        token_indices_sorted = torch.arange(bsz * seqlen * k, dtype=torch.int32, device=device)
        return fused_weighted_restore(combined_rows, top_scores, token_indices_sorted, k, (bsz, seqlen, width))

    return _fwd_bwd(forward, device)


def fused_row_weighting(dtype, device, hidden=None):
    from deepspeed.ops.triton_ops.autoep_fused_token_ops import _FusedRowWeighting

    width = hidden if hidden is not None and hidden > 0 else 64

    def forward():
        rows = _probe_tensor((2, width), dtype, device).requires_grad_()
        weights = _probe_tensor((2, 1), torch.float32, device).requires_grad_()
        return _FusedRowWeighting.apply(rows, weights)

    return _fwd_bwd(forward, device)


def chunked_cross_entropy(dtype, device, vocab_size=None):
    _TritonCrossEntropy = _dispatched_class("chunked_cross_entropy", "_TritonCrossEntropy",
                                            "deepspeed.runtime.chunked_cross_entropy")
    vocab = max(1, vocab_size) if vocab_size is not None and vocab_size > 0 else 64

    def forward():
        logits = _probe_tensor((2, vocab), dtype, device).requires_grad_()
        target = torch.arange(2, dtype=torch.long, device=device)
        return _TritonCrossEntropy.apply(logits, target, -100)

    return _fwd_bwd(forward, device)


def fp8_gemm(dtype, device, weight_dtype, k, n, quantization_group_size):
    """FP8 GEMM has no autograd; the probe only runs the forward kernel.

    The smoke shape uses M=16, so a production M >= 256 selects a different
    BLOCK_M/BLOCK_N/BLOCK_K and compiles its own variant on first use.
    """
    from deepspeed.ops.fp_quantizer.fp8_gemm_triton import matmul_fp8_triton
    from deepspeed.ops.triton_ops._triton import current_device

    def run():
        from contextlib import nullcontext

        # current_device takes an index; index-less devices skip the switch.
        device_ctx = current_device(device.index) if device.index is not None else nullcontext()
        with device_ctx:
            inp = _probe_tensor((16, k), dtype, device)
            weight = torch.zeros((k, n), dtype=weight_dtype, device=device)
            scale = torch.ones((k // quantization_group_size, n // quantization_group_size), device=device)
            matmul_fp8_triton(inp, weight, scale, quantization_group_size)

    return run


def liger_vocab_parallel_cross_entropy(dtype, device, tp_group, local_vocab_size):
    from deepspeed.sequence.cross_entropy import _liger_vocab_cross_entropy

    def forward():
        logits = torch.zeros((2, local_vocab_size), dtype=dtype, device=device, requires_grad=True)
        target = torch.zeros((2, ), dtype=torch.long, device=device)
        return _liger_vocab_cross_entropy(logits, target, tp_group, -100)

    return _fwd_bwd(forward, device)
