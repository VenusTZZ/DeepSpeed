# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""NPU HiFloat8 backend for the engine-level ``low_precision_training`` config.

torchao_npu ships the HiFloat8 GEMM primitives (``_quantize`` /
``hifloat8_matmul``) and a ``HiFloat8Linear`` drop-in for ordinary
``nn.Linear`` modules; this backend assembles the engine-side training
behavior (kernel probe + Linear conversion) from those shipped pieces.

All torchao_npu usage is contained in this file; it is only reached
through ``get_accelerator()`` hooks, so the engine and the other
accelerator backends stay untouched.
"""

import importlib.util
import logging

import torch

__all__ = ["assert_hifloat8_supported", "convert_hifloat8_linears"]

logger = logging.getLogger(__name__)


def _hifloat8_supported():
    # The single process-wide kernel probe, shared with the MoE grouped-mm
    # backend (op_builder/npu/grouped_mm.py).
    from op_builder.npu.grouped_mm import _hifloat8_supported
    return _hifloat8_supported()


def _import_hifloat8():
    """Import torchao_npu.hifloat8, raising a clear RuntimeError on absence."""
    if importlib.util.find_spec("torchao_npu") is None:
        raise RuntimeError("Low-precision training requires the torchao_npu backend, which is not installed")
    try:
        import torchao_npu  # noqa: F401  activation patches torchao on import
    except Exception as error:
        raise RuntimeError(f"torchao_npu import failed before low-precision conversion: {error}") from error
    try:
        from torchao_npu import hifloat8
    except ImportError as error:
        raise RuntimeError(f"installed torchao_npu provides no HiFloat8 module: {error}") from error
    return hifloat8


def assert_hifloat8_supported(config=None):
    """Raise RuntimeError when the installed backend cannot run HiFloat8 here.

    The engine calls this before any module is mutated, so an unsupported
    request fails without touching the model. The installed torchao_npu has
    no per-field policy support, so a non-empty ``config`` policy is reported
    and ignored.
    """
    hifloat8 = _import_hifloat8()
    if config:
        logger.warning("installed torchao_npu has no policy support; the low_precision_training 'config' "
                       "policy is ignored")
    if not _hifloat8_supported():
        raise RuntimeError("Low-precision training requested but the HiFloat8 kernels are not available "
                           "on this hardware")
    return None


def convert_hifloat8_linears(module, names, config=None):
    """Replace the named ``nn.Linear`` modules with HiFloat8 equivalents.

    ``names`` are dotted module paths selected by the engine. The
    ``HiFloat8Linear`` drop-in rebinds the original Parameter objects, so
    parameter identity and state-dict keys are preserved.
    """
    if not names:
        return module
    hifloat8 = _import_hifloat8()
    if config:
        logger.warning("installed torchao_npu has no policy support; the low_precision_training 'config' "
                       "policy is ignored")
    for name in names:
        parent_name, _, child_name = name.rpartition(".")
        parent = module if not parent_name else module.get_submodule(parent_name)
        original = getattr(parent, child_name)
        if type(original) is not torch.nn.Linear:
            raise RuntimeError(f"Low-precision conversion expected nn.Linear at '{name}', got "
                               f"{type(original).__name__}")
        setattr(parent, child_name, hifloat8.HiFloat8Linear.from_float(original))
    return module
