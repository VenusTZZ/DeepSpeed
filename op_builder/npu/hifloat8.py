# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""NPU HiFloat8 backend for the engine-level ``hifloat8`` training config.

torchao_npu ships the HiFloat8 GEMM primitives (``_quantize`` /
``hifloat8_matmul``) and a ``HiFloat8Linear`` drop-in for ordinary
``nn.Linear`` modules. Newer releases additionally provide a training
helper (``HiFloat8Config`` / ``assert_hifloat8_training_available`` /
``convert_to_hifloat8_training``) that owns policy-aware conversion;
when it is present this backend delegates to it, and otherwise it
assembles the same behavior from the shipped primitives.

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
        raise RuntimeError("HiFloat8 training requires the torchao_npu backend, which is not installed")
    try:
        import torchao_npu  # noqa: F401  activation patches torchao on import
    except Exception as error:
        raise RuntimeError(f"torchao_npu import failed before HiFloat8 conversion: {error}") from error
    try:
        from torchao_npu import hifloat8
    except ImportError as error:
        raise RuntimeError(f"installed torchao_npu provides no HiFloat8 module: {error}") from error
    return hifloat8


def assert_hifloat8_supported(device=None, config=None, linear=False, grouped=False):
    """Raise RuntimeError when the installed backend cannot run HiFloat8 here.

    The engine calls this before any module is mutated, so an unsupported
    request fails without touching the model.
    """
    hifloat8 = _import_hifloat8()
    if hasattr(hifloat8, "assert_hifloat8_training_available"):
        # Newer torchao_npu releases own the policy-aware probe (including
        # per-field validation of ``config``).
        policy = hifloat8.HiFloat8Config.from_dict(config) if hasattr(hifloat8, "HiFloat8Config") else None
        return hifloat8.assert_hifloat8_training_available(device=device,
                                                           config=policy,
                                                           linear=linear,
                                                           grouped=grouped)
    if config:
        logger.warning("installed torchao_npu has no HiFloat8 policy support; 'hifloat8.config' is ignored")
    if not _hifloat8_supported():
        raise RuntimeError("HiFloat8 training requested but the HiFloat8 kernels are not available "
                           "on this hardware")
    return None


def convert_hifloat8_linears(module, names, config=None):
    """Replace the named ``nn.Linear`` modules with HiFloat8 equivalents.

    ``names`` are dotted module paths selected by the engine; the returned
    module is authoritative (newer backends may rebuild the tree).
    """
    if not names:
        return module
    hifloat8 = _import_hifloat8()
    if hasattr(hifloat8, "convert_to_hifloat8_training"):
        # Newer torchao_npu releases own the conversion.
        policy = hifloat8.HiFloat8Config.from_dict(config) if hasattr(hifloat8, "HiFloat8Config") else None
        selected = set(names)
        return hifloat8.convert_to_hifloat8_training(module, policy, filter_fn=lambda _m, name: name in selected)
    # torchao_npu 0.18: swap in the shipped HiFloat8Linear drop-in, which
    # rebinds the original Parameter objects (so parameter identity and
    # state-dict keys are preserved).
    if config:
        logger.warning("installed torchao_npu has no HiFloat8 policy support; 'hifloat8.config' is ignored")
    for name in names:
        parent_name, _, child_name = name.rpartition(".")
        parent = module if not parent_name else module.get_submodule(parent_name)
        original = getattr(parent, child_name)
        if type(original) is not torch.nn.Linear:
            raise RuntimeError(f"HiFloat8 conversion expected nn.Linear at '{name}', got "
                               f"{type(original).__name__}")
        setattr(parent, child_name, hifloat8.HiFloat8Linear.from_float(original))
    return module
