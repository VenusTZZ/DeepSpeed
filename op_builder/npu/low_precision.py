# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Low-precision training backend for Ascend NPU (``quantization`` config section).

This module is the vendor zone for the NPU low-precision hooks: every
``torchao_npu`` import in DeepSpeed lives here and is lazy, so the shared MoE
code and the other accelerator backends stay untouched. The hooks themselves
are exposed through the accelerator abstraction (``npu_accelerator.py``).

Implemented formats:

- ``hifloat8``: Ascend HiFloat8 through ``torchao_npu.hifloat8``. The quantize
  and GEMM kernels are hardware-gated to Ascend 950 (socVersion 910_95), so on
  other chips (e.g. 910B4) the assert hook fails closed with a clear message.
- ``int8``: per-channel symmetric int8 weight grid with straight-through
  estimator (STE) semantics, computed in pure torch - the weights live on the
  int8 grid while the GEMM runs in BF16. This is the only training-capable
  format on chips without the 950 kernels (verified on 910B4: full
  forward/backward with weight gradients).
- BF16 grouped GEMM: the degradation target for MoE experts when the requested
  format's kernels are unavailable; always available.

Degradation discipline: ``get_grouped_mm_fn`` always returns a usable
callable. If the requested precision is unavailable it degrades to the BF16
grouped path (with a warning) instead of raising at call time; the fail-closed
abort for a format happens earlier, in the assert hook.
"""

import logging
from functools import lru_cache, partial

import torch

__all__ = [
    "assert_low_precision_training_supported",
    "convert_linears_to_low_precision",
    "get_grouped_mm_fn",
]

logger = logging.getLogger(__name__)

_SUPPORTED_FORMATS = ("hifloat8", "int8")
_HIFLOAT8_INSTALL_HINT = ("HiFloat8 on NPU requires the paired torchao_npu extension; install torchao and torchao_npu "
                          "(matching versions, e.g. 0.18) and import torchao_npu before use.")


def _import_hifloat8():
    try:
        from torchao_npu.hifloat8 import HiFloat8Linear, hifloat8_linear  # noqa: F401
        return HiFloat8Linear, hifloat8_linear
    except ImportError as e:
        raise NotImplementedError(f"{_HIFLOAT8_INSTALL_HINT} Original error: {e}") from e


def _resolve_format(format_config):
    fmt_type = format_config.get("type")
    if fmt_type not in _SUPPORTED_FORMATS:
        raise NotImplementedError(
            f"Format {fmt_type!r} is not implemented on the npu accelerator (implemented: "
            f"{list(_SUPPORTED_FORMATS)}). The config layer accepts any format name by design; each backend "
            f"implements a subset and fails closed for the rest.")
    return fmt_type


@lru_cache(maxsize=None)
def _probe_hifloat8():
    """One-shot functional probe of the HiFloat8 kernels, cached per process.

    Quantizes a 2x2 activation and runs a 2x2 HiFloat8 linear forward. The
    kernels are hardware-gated to Ascend 950 (910_95): on other chips the call
    raises and the error message names the actual device.
    """
    HiFloat8Linear, _ = _import_hifloat8()
    try:
        import torch_npu
        device_name = torch_npu.npu.get_device_name(0)
    except Exception:
        device_name = None
    try:
        linear = HiFloat8Linear.from_float(torch.nn.Linear(2, 2, bias=False, device="npu"))
        out = linear(torch.randn(2, 2, device="npu"))
        torch.npu.synchronize()
        return True
    except Exception as e:
        hint = f" (this device reports {device_name}; HiFloat8 kernels require Ascend 950 / socVersion 910_95)" \
            if device_name else ""
        raise NotImplementedError(f"HiFloat8 kernels are unavailable on this device{hint}. Original error: {e}") from e


# ---- int8 weight grid (STE) -------------------------------------------------------


def _int8_grid_quantize(weight, group_size=None):
    """Symmetric int8 grid quantization with straight-through grads.

    Per-channel by default; ``group_size`` splits the trailing dimension into
    groups with one scale each (a finer grid, at the cost of scale storage).
    ``scale`` is detached, so gradients flow to ``weight`` through
    ``torch.round`` (whose gradient is 1) - the same STE semantics QAT uses.
    """
    if group_size is None:
        scale = weight.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 127.0
        q = torch.round(weight / scale)
        return q * scale
    if weight.shape[-1] % group_size != 0:
        raise ValueError(f"int8 group_size={group_size} does not divide the trailing dimension "
                         f"{weight.shape[-1]} of weight {tuple(weight.shape)}")
    flat = weight.reshape(-1, group_size)
    scale = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 127.0
    q = torch.round(flat / scale)
    return (q * scale).reshape(weight.shape)


class _Int8GridLinear(torch.nn.Linear):
    """Linear whose weights live on the symmetric int8 grid; GEMM runs in BF16."""

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, group_size=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.group_size = group_size

    def forward(self, x):
        w = _int8_grid_quantize(self.weight, self.group_size)
        if self.bias is not None:
            return torch.nn.functional.linear(x, w, self.bias)
        return torch.nn.functional.linear(x, w)


def assert_low_precision_training_supported(format_config):
    """Fail-closed check: raise unless this accelerator can train in the requested format."""
    fmt_type = _resolve_format(format_config)
    if fmt_type == "hifloat8":
        _probe_hifloat8()
    elif fmt_type == "int8":
        group_size = format_config.get("group_size")
        if group_size is not None and (not isinstance(group_size, int) or group_size <= 0):
            raise ValueError(f"int8 group_size must be a positive integer or null, got {group_size!r}")
    # "int8" is pure torch (no device kernels): always available.


def _patch_int8_grid_ste(module, group_size=None):
    """Patch a fused expert container's forward with per-call STE quantization.

    Fused containers (e.g. transformers 5.x Qwen2MoeExperts / DeepseekV3Experts)
    read their 3-D expert weights directly in forward. The patched forward
    swaps each 3-D parameter for its grid-quantized tensor for the duration of
    the call (finally-restored), so gradients flow through ``torch.round``
    (STE) back to the master weights. The module's identity and FQN are
    unchanged; checkpoint keys are unaffected.
    """
    original_forward = module.forward

    def ste_forward(*args, **kwargs):
        originals = {}
        for name, param in module.named_parameters(recurse=False):
            originals[name] = param
            module._parameters[name] = _int8_grid_quantize(param, group_size)
        try:
            return original_forward(*args, **kwargs)
        finally:
            for name, param in originals.items():
                module._parameters[name] = param

    module.forward = ste_forward
    module._int8_grid_ste_group_size = group_size


def convert_linears_to_low_precision(module, names, format_config):
    """Adapt the modules at ``names`` (FQNs) to the format, in place.

    ``torch.nn.Linear`` modules are replaced with the format's quantized
    variant; fused expert containers (all direct parameters 3-D) are patched
    with a per-call STE forward. Any other module kind fails closed.
    """
    fmt_type = _resolve_format(format_config)
    if fmt_type == "hifloat8":
        HiFloat8Linear, _ = _import_hifloat8()
    converted = []
    for name in names:
        parent_name, _, child_name = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        target = module.get_submodule(name)
        direct_params = list(target.parameters(recurse=False))
        if isinstance(target, torch.nn.Linear):
            if fmt_type == "int8":
                replacement = _Int8GridLinear(target.in_features,
                                              target.out_features,
                                              bias=target.bias is not None,
                                              device=target.weight.device,
                                              dtype=target.weight.dtype,
                                              group_size=format_config.get("group_size"))
                with torch.no_grad():
                    replacement.weight.copy_(target.weight)
                    if target.bias is not None:
                        replacement.bias.copy_(target.bias)
            else:
                replacement = HiFloat8Linear.from_float(target)
            setattr(parent, child_name, replacement)
            converted.append(name)
        elif direct_params and all(p.ndim == 3 for p in direct_params):
            # Fused expert container: adapt in place with a per-call STE forward.
            if fmt_type != "int8":
                raise NotImplementedError(f"Format {fmt_type!r} is not implemented for fused expert containers on "
                                          f"npu; only 'int8' supports them. Use per-expert Linear patterns or the "
                                          f"GroupedExperts path instead.")
            _patch_int8_grid_ste(target, group_size=format_config.get("group_size"))
            converted.append(name)
        else:
            raise TypeError(f"Cannot convert module {name!r} to {fmt_type}: expected torch.nn.Linear or a fused "
                            f"3-D expert container, got {type(target).__name__}")
    return converted


# ---- grouped GEMM for MoE experts -----------------------------------------------


def _to_group_list(offsets, device):
    # The Ascend grouped matmul wants int64 cumulative end offsets on device.
    if offsets.dtype != torch.int64 or offsets.device != device:
        offsets = offsets.to(device=device, dtype=torch.int64)
    return offsets


def _unwrap(outputs):
    # mx_grouped_matmul takes lists of operands and returns a list of results.
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    return outputs


class _BF16GroupedMM(torch.autograd.Function):
    """BF16 grouped GEMM over the Ascend grouped matmul (torchao_npu's wrapper)."""

    @staticmethod
    def forward(ctx, x, weight, offsets):
        from torchao_npu.ops.npu.mx_ops import mx_grouped_matmul
        group_list = _to_group_list(offsets, x.device)
        out = mx_grouped_matmul([x], [weight], group_list=group_list, group_type=0, split_item=3)
        ctx.save_for_backward(x, weight, group_list)
        return _unwrap(out)

    @staticmethod
    def backward(ctx, grad_out):
        from torchao_npu.ops.npu.mx_ops import mx_grouped_matmul
        x, weight, group_list = ctx.saved_tensors
        dx = _unwrap(
            mx_grouped_matmul([grad_out], [weight.transpose(1, 2)], group_list=group_list, group_type=0, split_item=3))
        dw = _unwrap(
            mx_grouped_matmul([x.transpose(0, 1)], [grad_out], group_list=group_list, group_type=2, split_item=3))
        return dx, dw, None


def _bf16_grouped_mm(x, weight, offsets):
    # The callers pass transposed views (e.g. [E, hidden, dim].transpose(-2, -1));
    # the Ascend grouped matmul needs contiguous operands.
    return _BF16GroupedMM.apply(x.contiguous(), weight.contiguous(), offsets)


def _hifloat8_grouped_mm(x, weight, offsets):
    """Per-expert HiFloat8 GEMM (differentiable per expert via torchao_npu's functional)."""
    _, hifloat8_linear = _import_hifloat8()
    outs = []
    prev = 0
    for end in offsets.tolist():
        outs.append(hifloat8_linear(x[prev:end], weight[len(outs)]))
        prev = end
    return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]


def _int8_grid_grouped_mm(x, weight, offsets, group_size=None):
    """Grouped GEMM with expert weights on the symmetric int8 grid (STE); GEMM runs in BF16."""
    w = _int8_grid_quantize(weight, group_size)
    return _bf16_grouped_mm(x, w, offsets)


def get_grouped_mm_fn(format_config):
    """Return a grouped-GEMM callable ``fn(x, weight, offsets) -> out`` for MoE experts.

    ``x`` is [total_tokens, in_features], ``weight`` is [num_experts, in_features, out_features] and
    ``offsets`` are cumulative expert end boundaries. Always returns a usable callable: if the
    requested format's kernels are unavailable it degrades to BF16 grouped GEMM with a warning
    (the fail-closed abort for a format happens in the assert hook).
    """
    fmt_type = _resolve_format(format_config)
    if fmt_type == "int8":
        return partial(_int8_grid_grouped_mm, group_size=format_config.get("group_size"))
    if fmt_type == "hifloat8":
        try:
            _probe_hifloat8()
        except NotImplementedError as e:
            logger.warning("HiFloat8 grouped GEMM is unavailable (%s); degrading experts to BF16 grouped GEMM.", e)
            return _bf16_grouped_mm
        return _hifloat8_grouped_mm
