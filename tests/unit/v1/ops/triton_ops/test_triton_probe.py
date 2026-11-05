# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Unit tests for the per-kernel Triton probe gate.

These tests pin the probe mechanism's contract, not its timing: the probe runs
at most once per process per key, later calls answer from the cache without
running or constructing the probe again, a failing probe closes the gate with a
cached False, a probe does not advance the caller's RNG sequence, and
DS_TRITON_KERNEL_GATE=0 bypasses probing entirely. The probe's wall-clock cost
is covered by a standalone benchmark outside the repo (numbers in the PR
description).
"""

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.ops.triton_ops import _probes
from deepspeed.ops.triton_ops._triton import (_TRITON_AVAILABLE, _TRITON_OPS_IMPL_CACHE, _TRITON_PROBE_CACHE,
                                              is_triton_kernel_supported, resolve_triton_ops_impl)

_ACCELERATOR = get_accelerator()

pytestmark = pytest.mark.skipif(not _ACCELERATOR.is_available() or not _TRITON_AVAILABLE
                                or not _ACCELERATOR.is_triton_supported(),
                                reason="requires an accelerator with an active Triton backend")


def _dtype_and_device():
    device = torch.device(_ACCELERATOR.current_device_name())
    dtype = torch.bfloat16
    return dtype, device


def _key(name, dtype, device):
    return f"triton_kernel:{name}:{device.type}:{device.index}:{dtype}"


def test_probe_runs_once_and_later_calls_answer_from_the_cache():
    dtype, device = _dtype_and_device()
    calls = {"n": 0}

    def probe_fn():
        calls["n"] += 1
        return _probes.swiglu(dtype, device)

    _TRITON_PROBE_CACHE.pop(_key("unit_probe_once", dtype, device), None)
    first = is_triton_kernel_supported("unit_probe_once", dtype, device, probe_fn)
    second = is_triton_kernel_supported("unit_probe_once", dtype, device, probe_fn)
    assert first == second
    # The second call answered from the cache: the probe factory never ran again.
    assert calls["n"] == 1


def test_failing_probe_closes_the_gate_with_a_cached_false():
    dtype, device = _dtype_and_device()

    def failing_probe_fn():

        def run():
            raise RuntimeError("deliberate unit-test probe failure")

        return run

    _TRITON_PROBE_CACHE.pop(_key("unit_probe_fail", dtype, device), None)
    assert not is_triton_kernel_supported("unit_probe_fail", dtype, device, failing_probe_fn)

    calls = {"n": 0}

    def counting_probe_fn():
        calls["n"] += 1
        return _probes.swiglu(dtype, device)

    # The cached False stands and the replacement probe is never even constructed.
    assert not is_triton_kernel_supported("unit_probe_fail", dtype, device, counting_probe_fn)
    assert calls["n"] == 0


def test_probe_cache_is_scoped_by_dtype():
    dtype, device = _dtype_and_device()
    other_dtype = torch.float16 if dtype != torch.float16 else torch.float32
    name = "unit_probe_dtype"
    calls = []

    def probe_fn(dt):
        calls.append(dt)

        def run():
            if dt != dtype:
                raise RuntimeError("deliberate per-dtype probe failure")
            return None

        return run

    _TRITON_PROBE_CACHE.pop(_key(name, dtype, device), None)
    _TRITON_PROBE_CACHE.pop(_key(name, other_dtype, device), None)

    # A backend can compile a kernel for one dtype and fail for another, so the
    # second dtype must run its own probe instead of reusing the first answer.
    assert is_triton_kernel_supported(name, dtype, device, probe_fn, dtype)
    assert not is_triton_kernel_supported(name, other_dtype, device, probe_fn, other_dtype)
    assert calls == [dtype, other_dtype]

    # Both answers are now cached under their own dtype-scoped keys.
    assert is_triton_kernel_supported(name, dtype, device, probe_fn, dtype)
    assert not is_triton_kernel_supported(name, other_dtype, device, probe_fn, other_dtype)
    assert calls == [dtype, other_dtype]


def test_probe_does_not_advance_the_caller_rng_sequence():
    dtype, device = _dtype_and_device()
    # Compare the first draw after seeding WITH the probe in between against
    # the same draw WITHOUT the probe: reseed, run the probe, draw
    # ``after_probe``, then reseed again and draw ``control``. If the probe
    # consumed any random numbers, ``after_probe`` starts further along the
    # sequence than ``control`` and the two draws differ.
    _ACCELERATOR.manual_seed(4321)
    is_triton_kernel_supported("unit_probe_rng", dtype, device, _probes.fused_rms_norm, dtype, device)
    after_probe = torch.randn(4, device=device)

    _ACCELERATOR.manual_seed(4321)
    control = torch.randn(4, device=device)

    assert torch.equal(after_probe, control)


def test_env_escape_hatch_bypasses_probing(monkeypatch):
    dtype, device = _dtype_and_device()
    monkeypatch.setenv("DS_TRITON_KERNEL_GATE", "0")
    calls = {"n": 0}

    def probe_fn():
        calls["n"] += 1
        return _probes.swiglu(dtype, device)

    # The gate answers at the backend level without ever touching the probe.
    assert is_triton_kernel_supported("unit_probe_env", dtype, device, probe_fn) == _ACCELERATOR.is_triton_supported()
    assert calls["n"] == 0


def test_static_declaration_answers_from_the_backend_gate_without_a_probe():
    dtype, device = _dtype_and_device()
    # probe_fn=None is the static-declaration form for probe-unsafe paths:
    # nothing runs, and a kernel name no backend declares answers with the
    # backend-level gate.
    _TRITON_PROBE_CACHE.pop(_key("unit_probe_static", dtype, device), None)
    assert is_triton_kernel_supported("unit_probe_static", dtype, device, None) == _ACCELERATOR.is_triton_supported()


def _clear_kernel_caches(feature, portable_module=None):
    """Forget in-process compiled variants so the probe has to compile them again."""
    _TRITON_OPS_IMPL_CACHE.clear()
    impl = resolve_triton_ops_impl(feature, portable_module)
    for value in vars(impl).values():
        if hasattr(value, "device_caches"):
            value.device_caches.clear()


def _record_compiles(monkeypatch):
    from triton.runtime.jit import JITFunction

    calls = []
    original = JITFunction._do_compile

    def wrapped(self, key, signature, device, constexprs, options, attrs, warmup):
        calls.append(self._fn_name)
        return original(self, key, signature, device, constexprs, options, attrs, warmup)

    monkeypatch.setattr(JITFunction, "_do_compile", wrapped)
    return calls


def _assert_probe_compiles_the_production_variant(monkeypatch, kernel, probe_fn, probe_args, shape_key, real_call,
                                                  feature, portable_module=None):
    dtype, device = _dtype_and_device()
    _clear_kernel_caches(feature, portable_module)
    calls = _record_compiles(monkeypatch)
    probe_key = f"triton_kernel:{kernel}:{device.type}:{device.index}:{dtype}:{shape_key}"
    _TRITON_PROBE_CACHE.pop(probe_key, None)

    assert is_triton_kernel_supported(kernel, dtype, device, probe_fn, *probe_args, shape_key=shape_key)
    assert calls, "the probe did not compile anything; the test would not prove cache reuse"

    calls.clear()
    real_call()
    _ACCELERATOR.current_stream().synchronize()
    assert calls == [], f"the real call compiled again: {calls}"


def test_shape_aware_probe_prevents_a_second_swiglu_compile(monkeypatch):
    from deepspeed.ops.triton_ops.swiglu_triton import swiglu

    dtype, device = _dtype_and_device()
    gate = torch.randn(4, 64, dtype=dtype, device=device, requires_grad=True)
    up = torch.randn(4, 64, dtype=dtype, device=device, requires_grad=True)
    n_elements = gate.numel()

    def real_call():
        out = swiglu(gate, up)
        out.sum().backward()

    _assert_probe_compiles_the_production_variant(monkeypatch,
                                                  "swiglu",
                                                  _probes.swiglu, (dtype, device, n_elements),
                                                  _probes.probe_extent_key(n_elements),
                                                  real_call,
                                                  "swiglu_triton")


def test_shape_aware_probe_prevents_a_second_rms_norm_compile(monkeypatch):
    from deepspeed.ops.triton_ops.fused_rms_norm import fused_rms_norm

    dtype, device = _dtype_and_device()
    hidden = torch.randn(2, 128, dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn(128, dtype=dtype, device=device, requires_grad=True)
    n_rows, n_cols = hidden.shape

    def real_call():
        out = fused_rms_norm(hidden, weight, 1e-6)
        out.sum().backward()

    shape_key = f"{n_cols}:{_probes.probe_rows_key(n_rows)}"
    _assert_probe_compiles_the_production_variant(monkeypatch,
                                                  "fused_rms_norm",
                                                  _probes.fused_rms_norm, (dtype, device, n_cols, n_rows),
                                                  shape_key,
                                                  real_call,
                                                  "fused_rms_norm")


def test_shape_aware_probe_prevents_a_second_rope_compile(monkeypatch):
    from deepspeed.ops.triton_ops.fused_rotary_pos_emb import fused_apply_rotary_pos_emb

    dtype, device = _dtype_and_device()
    q = torch.randn(1, 8, 32, 64, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(1, 8, 32, 64, dtype=dtype, device=device, requires_grad=True)
    cos = torch.randn(1, 32, 64, dtype=dtype, device=device)
    sin = torch.randn(1, 32, 64, dtype=dtype, device=device)
    head_dim, seq_len, n_heads = q.shape[-1], q.shape[2], q.shape[1]

    def real_call():
        q_out, k_out = fused_apply_rotary_pos_emb(q, k, cos, sin)
        (q_out.sum() + k_out.sum()).backward()

    shape_key = _probes.probe_rotary_key(head_dim, seq_len, n_heads, device)
    _assert_probe_compiles_the_production_variant(monkeypatch,
                                                  "fused_rotary_pos_emb",
                                                  _probes.fused_rotary, (dtype, device, head_dim, seq_len, n_heads),
                                                  shape_key,
                                                  real_call,
                                                  "fused_rotary_pos_emb")
