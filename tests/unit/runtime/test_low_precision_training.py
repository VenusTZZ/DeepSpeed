# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import torch.nn as nn

from deepspeed.runtime.config import DeepSpeedConfig, DeepSpeedConfigError
from deepspeed.runtime.engine import _collect_low_precision_modules


def _qcfg(quantization):
    return DeepSpeedConfig({"train_batch_size": 64, "quantization": quantization}).quantization_config


def _int8():
    return {"type": "int8", "backend": None}


class _FusedExperts(nn.Module):
    """仿 transformers 5.x 融合专家容器（3-D 权重）。"""

    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(4, 16, 8))
        self.down_proj = nn.Parameter(torch.randn(4, 8, 8))

    def forward(self, x):
        return x


class _Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.embed = nn.Embedding(8, 4)
        self.act = nn.ReLU()


class TestCollectLowPrecisionModules:

    def test_exact_fqn_buckets_only_that_linear(self):
        linears, grouped, containers = _collect_low_precision_modules(
            _Net(), _qcfg({"module_configs": {
                "layers.0": _int8(),
                "_default": None
            }}))
        assert not grouped and not containers
        assert len(linears) == 1
        (fmt, names), = list(linears.values())
        assert fmt == _int8()
        assert names == ["layers.0"]

    def test_regex_and_null_default_leave_others_alone(self):
        linears, grouped, containers = _collect_low_precision_modules(
            _Net(), _qcfg({"module_configs": {
                "re:layers\\..+": _int8(),
                "_default": None
            }}))
        assert not grouped and not containers
        (fmt, names), = list(linears.values())
        assert names == ["layers.0", "layers.1"]
        # embed/act are untouched: _default is null and parameter-free modules
        # never participate in matching.

    def test_matched_unsupported_module_fails_closed(self):
        with pytest.raises(RuntimeError, match="embed"):
            _collect_low_precision_modules(_Net(), _qcfg({"module_configs": {"re:.*": _int8(), "_default": None}}))

    def test_unmatched_module_without_default_fails_closed(self):
        with pytest.raises(DeepSpeedConfigError):
            _collect_low_precision_modules(_Net(), _qcfg({"module_configs": {"layers.0": _int8()}}))

    def test_grouped_experts_bucketed(self):
        from deepspeed.moe.ep_experts import GroupedExperts
        net = nn.Module()
        net.experts = GroupedExperts(dim=8, hidden_dim=16, num_experts=2, use_grouped_mm=False)
        linears, grouped, containers = _collect_low_precision_modules(net,
                                                                      _qcfg({"module_configs": {
                                                                          "experts": _int8()
                                                                      }}))
        assert not linears and not containers
        (fmt, modules), = list(grouped.values())
        assert fmt == _int8()
        assert [name for name, _ in modules] == ["experts"]

    def test_fused_expert_container_bucketed(self):
        net = nn.Module()
        net.mlp = nn.Module()
        net.mlp.experts = _FusedExperts()
        linears, grouped, containers = _collect_low_precision_modules(
            net, _qcfg({"module_configs": {
                "mlp.experts": _int8(),
                "_default": None
            }}))
        assert not linears and not grouped
        (fmt, names), = list(containers.values())
        assert fmt == _int8()
        assert names == ["mlp.experts"]
