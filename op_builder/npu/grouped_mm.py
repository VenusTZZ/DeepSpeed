# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""NPU grouped GEMM for MoE experts, backed by torchao_npu.

torchao_npu's ``mx_grouped_matmul`` stabilizes the ``torch_npu.
npu_grouped_matmul`` call pattern (BF16 form: list operands, ``group_type``,
``split_item``, ``group_list``) but ships no autograd. This module adds the
autograd wrappers on top, mirroring torchao_npu's own ``W4A8GMMFunction``
three-GEMM structure (forward / dgrad / wgrad).

Two precisions are provided:

  - ``bf16``: one native grouped-matmul kernel call per GEMM.
  - ``hifloat8``: per-expert HiFloat8 GEMMs on the per-tensor-quantized
    ``npu_quant_matmul`` primitives from ``torchao_npu.hifloat8`` (the same
    primitives its ``_HiFloat8MM`` Linear uses); the HiFloat8 kernels are
    only provided for the Ascend 910_95 socVersion.

``get_external_grouped_mm_fn`` always returns a working callable on NPU:
when torchao_npu is unavailable, the precision is unknown, or the hardware
cannot run the configured precision, the returned callable degrades to
plain-torch per-expert matmuls. This keeps the NPU dispatch from ever
reaching ``torch._grouped_mm``, whose backward is broken in torch CPU
builds.

All torchao_npu usage is contained in this file; it is only reached through
``get_accelerator().get_external_grouped_mm_fn()``, so the MoE core code and
the other accelerator backends stay untouched.
"""

import importlib.util
import logging

import torch

__all__ = ["get_external_grouped_mm_fn"]

logger = logging.getLogger(__name__)

_SUPPORTED_PRECISIONS = ("bf16", "hifloat8")


def _to_group_list(offsets, device):
    # npu_grouped_matmul wants int64 cumulative end offsets on device.
    if offsets.dtype != torch.int64 or offsets.device != device:
        offsets = offsets.to(device=device, dtype=torch.int64)
    return offsets


def _torch_grouped_mm_fallback(x, weight, offsets):
    """Most-basic version: sequential per-expert matmuls in plain PyTorch.

    Matches the contract of the kernel paths: x [M, K], weight [E, K, N]
    in GEMM layout, offsets [E] int32 cumulative end offsets.
    """
    parts = []
    prev = 0
    for expert, end in enumerate(offsets.tolist()):
        if end > prev:
            parts.append(torch.matmul(x[prev:end], weight[expert]))
        prev = end
    if not parts:
        return torch.empty(0, weight.shape[-1], dtype=x.dtype, device=x.device)
    return torch.cat(parts)


class _NPUGroupedMM(torch.autograd.Function):
    """Grouped GEMM with autograd on Ascend NPU (FP16/BF16).

    Call contract (shared with the other grouped-mm paths in
    ``deepspeed.moe.ep_experts``):
      - ``x``:       ``[M, K]`` activations, rows grouped by expert
      - ``weight``:  ``[E, K, N]`` expert weights in GEMM layout (the MoE
        core transposes its stored ``[E, N, K]`` weights before calling)
      - ``offsets``: ``[E]`` int32 cumulative end offsets, on the device
      - returns ``[M, N]``

    forward:  y  = grouped(x @ weight)                (group_type=0)
    backward: dx = grouped(grad_out @ weight^T)       (group_type=0)
              dw = grouped(x^T @ grad_out) -> [E,K,N] (group_type=2)

    The three GEMMs mirror the structure of torchao_npu's
    ``W4A8GMMFunction`` (``torchao_npu/w4a8/w4a8_grouped_mm.py``), which is
    the authoritative call pattern for ``torch_npu.npu_grouped_matmul``.
    """

    @staticmethod
    def forward(ctx, x, weight, offsets):
        from torchao_npu.ops import npu

        if x.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(f"_NPUGroupedMM supports fp16/bf16, got {x.dtype}")
        group_list = _to_group_list(offsets, x.device)
        ctx.save_for_backward(x, weight, group_list)
        out = npu.mx_grouped_matmul([x], [weight], group_list=group_list, group_type=0, split_item=3)
        return out[0]

    @staticmethod
    def backward(ctx, grad_out):
        from torchao_npu.ops import npu

        x, weight, group_list = ctx.saved_tensors
        dx = npu.mx_grouped_matmul([grad_out], [weight.transpose(1, 2)],
                                   group_list=group_list,
                                   group_type=0,
                                   split_item=3)[0]
        dw = npu.mx_grouped_matmul([x.transpose(0, 1)], [grad_out], group_list=group_list, group_type=2,
                                   split_item=3)[0]
        return dx, dw, None


class _NPUHiFloat8GroupedMM(torch.autograd.Function):
    """Grouped GEMM in HiFloat8 with autograd on Ascend NPU (BF16 inputs).

    Same call contract as :class:`_NPUGroupedMM`. torchao_npu ships no
    HiFloat8 grouped matmul, so each expert GEMM runs separately on the
    per-tensor-quantized HiFloat8 primitives; the per-expert quantization is
    the natural per-group form of HiFloat8's per-tensor scale.
    """

    @staticmethod
    def forward(ctx, x, weight, offsets):
        from torchao_npu.hifloat8 import _quantize
        from torchao_npu.hifloat8.hifloat8_tensor import hifloat8_matmul

        if x.dtype != torch.bfloat16:
            # Mirrors the upstream HiFloat8 validation: BF16 inputs only.
            raise RuntimeError(f"HiFloat8 grouped GEMM requires bf16 inputs, got {x.dtype}")
        group_list = _to_group_list(offsets, x.device)
        ctx.save_for_backward(x, weight, group_list)

        parts = []
        prev = 0
        for expert, end in enumerate(group_list.tolist()):
            if end > prev:
                x_q = _quantize(x[prev:end])
                w_q = _quantize(weight[expert])
                parts.append(hifloat8_matmul(x_q, w_q, output_dtype=x.dtype))
            prev = end
        if not parts:
            return torch.empty(0, weight.shape[-1], dtype=x.dtype, device=x.device)
        return torch.cat(parts)

    @staticmethod
    def backward(ctx, grad_out):
        from torchao_npu.hifloat8 import _quantize
        from torchao_npu.hifloat8.hifloat8_tensor import hifloat8_matmul

        x, weight, group_list = ctx.saved_tensors
        dx_parts = []
        dw_parts = []
        prev = 0
        for expert, end in enumerate(group_list.tolist()):
            if end > prev:
                grad_q = _quantize(grad_out[prev:end])
                dx_parts.append(hifloat8_matmul(grad_q, _quantize(weight[expert].t()), output_dtype=x.dtype))
                dw_parts.append(hifloat8_matmul(_quantize(x[prev:end].t()), grad_q, output_dtype=x.dtype).unsqueeze(0))
            else:
                dw_parts.append(
                    torch.zeros(1, weight.shape[1], weight.shape[2], dtype=grad_out.dtype, device=grad_out.device))
            prev = end
        if not dx_parts:
            return (torch.empty(0, x.shape[-1], dtype=grad_out.dtype,
                                device=grad_out.device), torch.cat(dw_parts), None)
        return torch.cat(dx_parts), torch.cat(dw_parts), None


_HIFLOAT8_SUPPORT = None


def _hifloat8_supported():
    """Probe the HiFloat8 quantization kernel once per process.

    The HiFloat8 kernels are only provided for the Ascend 910_95
    socVersion; probing the kernel itself is more honest than matching
    device-name strings.
    """
    global _HIFLOAT8_SUPPORT
    if _HIFLOAT8_SUPPORT is None:
        try:
            import torch_npu

            probe = torch.zeros(2, 2, dtype=torch.bfloat16, device="npu")
            torch_npu.npu_dynamic_quant(probe, dst_type=torch_npu.hifloat8, quant_mode="pertensor")
            _HIFLOAT8_SUPPORT = True
        except Exception:
            _HIFLOAT8_SUPPORT = False
    return _HIFLOAT8_SUPPORT


def get_external_grouped_mm_fn(precision="bf16"):
    """Return the NPU grouped-mm callable for ``precision``.

    ``precision`` selects the GEMM precision: "bf16" (native grouped matmul
    kernel) or "hifloat8" (per-expert HiFloat8 GEMMs; the HiFloat8 kernels
    are only provided for the Ascend 910_95 socVersion).

    A callable is always returned on NPU: when torchao_npu is unavailable,
    the precision is unknown, or the hardware cannot run the configured
    precision, the returned callable degrades to plain-torch per-expert
    matmuls (the most basic version).
    """
    if precision not in _SUPPORTED_PRECISIONS:
        logger.warning(
            "unsupported grouped-mm precision %r on NPU; degrading to plain-torch "
            "per-expert GEMMs (expected one of %s)", precision, _SUPPORTED_PRECISIONS)
        return _torch_grouped_mm_fallback
    if importlib.util.find_spec("torchao_npu") is None:
        return _torch_grouped_mm_fallback
    try:
        import torchao_npu  # noqa: F401  activation patches torchao on import
        from torchao_npu.ops import npu  # noqa: F401
    except Exception:
        return _torch_grouped_mm_fallback
    if precision == "bf16":
        return _NPUGroupedMM.apply
    if not _hifloat8_supported():
        logger.warning("HiFloat8 grouped GEMM requested but unsupported on this hardware; "
                       "degrading to plain-torch per-expert GEMMs")
        return _torch_grouped_mm_fallback
    return _NPUHiFloat8GroupedMM.apply
