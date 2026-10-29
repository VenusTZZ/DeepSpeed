# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

from deepspeed.accelerator.abstract_accelerator import DeepSpeedAccelerator


class _StubAccelerator:
    # Duck-typed stand-in so the default implementations can be exercised
    # without implementing every abstract method of a real backend.
    def device_name(self):
        return "stub-accel"


def _format_config():
    return {"type": "hifloat8", "backend": None}


class TestLowPrecisionHookDefaults:

    def test_assert_hook_default_fails_closed(self):
        with pytest.raises(NotImplementedError, match="stub-accel"):
            DeepSpeedAccelerator.assert_low_precision_training_supported(_StubAccelerator(), _format_config())

    def test_convert_hook_default_fails_closed(self):
        with pytest.raises(NotImplementedError, match="stub-accel"):
            DeepSpeedAccelerator.convert_linears_to_low_precision(_StubAccelerator(), None, [], _format_config())

    def test_grouped_mm_hook_default_fails_closed(self):
        with pytest.raises(NotImplementedError, match="stub-accel"):
            DeepSpeedAccelerator.get_grouped_mm_fn(_StubAccelerator(), _format_config())


class TestNpuBackendValidation:
    """``op_builder/npu/low_precision.py``: named backends fail closed on unknown names.

    The int8 format is pure torch, so the assert hook runs on any machine
    without torch_npu.
    """

    @staticmethod
    def _npu_assert(format_config):
        from op_builder.npu.low_precision import assert_low_precision_training_supported
        return assert_low_precision_training_supported(format_config)

    def test_null_backend_selects_platform_default(self):
        self._npu_assert({"type": "int8", "backend": None})

    def test_named_backend_torchao_accepted(self):
        self._npu_assert({"type": "int8", "backend": "torchao"})

    def test_unknown_backend_fails_closed(self):
        with pytest.raises(NotImplementedError, match="Backend 'cuda-native'"):
            self._npu_assert({"type": "int8", "backend": "cuda-native"})
