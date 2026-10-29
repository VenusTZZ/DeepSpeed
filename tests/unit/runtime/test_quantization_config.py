# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

from pydantic import ValidationError

from deepspeed.runtime import config as ds_config


def _fmt(fmt_type):
    return {"type": fmt_type}


def _parsed(quantization):
    return ds_config.DeepSpeedConfig({"train_batch_size": 64, "quantization": quantization}).quantization_config


class TestQuantizationConfigSection:

    def test_absent_section_disables_quantization(self):
        config = ds_config.DeepSpeedConfig({"train_batch_size": 64})
        assert not config.quantization_config.enabled

    def test_empty_section_disables_quantization(self):
        assert not _parsed({}).enabled

    def test_default_alone_applies_to_every_module(self):
        qcfg = _parsed({"default": _fmt("hifloat8")})
        assert qcfg.enabled
        assert qcfg.match_module("model.decoder.layers.3.self_attn.q_proj") == _fmt("hifloat8")
        assert qcfg.match_module("any.other.module") == _fmt("hifloat8")

    def test_module_configs_priority_exact_over_regex_over_default_key(self):
        qcfg = _parsed({
            "module_configs": {
                "model.layers.3.q_proj": _fmt("format_exact"),
                "re:model\\.layers\\..+\\.q_proj": _fmt("format_regex"),
                "_default": _fmt("format_fallback"),
            },
        })
        assert qcfg.match_module("model.layers.3.q_proj") == _fmt("format_exact")
        assert qcfg.match_module("model.layers.5.q_proj") == _fmt("format_regex")
        assert qcfg.match_module("model.layers.5.k_proj") == _fmt("format_fallback")

    def test_default_shorthand_equivalent_to_default_key(self):
        a = _parsed({"default": _fmt("hifloat8")})
        b = _parsed({"module_configs": {"_default": _fmt("hifloat8")}})
        assert a.match_module("m") == b.match_module("m")

    def test_default_and_default_key_conflict_rejected(self):
        with pytest.raises(ds_config.DeepSpeedConfigError):
            _parsed({"default": _fmt("hifloat8"), "module_configs": {"_default": _fmt("fp8")}})

    def test_regex_keys_use_fullmatch(self):
        qcfg = _parsed({
            "module_configs": {
                "re:model\\.layers\\..+\\.out_pro": _fmt("format_regex"),
                "_default": _fmt("format_fallback"),
            },
        })
        # "out_pro" must not match "out_proj": no partial matches.
        assert qcfg.match_module("model.layers.3.self_attn.out_proj") == _fmt("format_fallback")
        assert qcfg.match_module("model.layers.3.self_attn.out_pro") == _fmt("format_regex")

    def test_null_default_key_means_top_level_precision(self):
        qcfg = _parsed({
            "module_configs": {
                "re:model\\.experts\\.\\d+": _fmt("hifloat8"),
                "_default": None
            },
        })
        assert qcfg.match_module("model.experts.0") == _fmt("hifloat8")
        assert qcfg.match_module("model.lm_head") is None

    def test_no_default_and_no_match_raises(self):
        qcfg = _parsed({"module_configs": {"re:model\\.experts\\.\\d+": _fmt("hifloat8")}})
        with pytest.raises(ds_config.DeepSpeedConfigError):
            qcfg.match_module("model.lm_head")

    def test_top_level_precision_sections_are_untouched(self):
        config = ds_config.DeepSpeedConfig({
            "train_batch_size": 64,
            "bf16": {
                "enabled": True
            },
            "quantization": {
                "default": _fmt("hifloat8")
            },
        })
        assert config.bfloat16_config.enabled
        assert config.quantization_config.match_module("m") == _fmt("hifloat8")


class TestQuantizationConfigValidation:

    def test_missing_type_rejected(self):
        with pytest.raises(ds_config.DeepSpeedConfigError):
            _parsed({"default": {"group_size": 128}})

    def test_empty_type_rejected(self):
        with pytest.raises(ds_config.DeepSpeedConfigError):
            _parsed({"default": _fmt("")})

    def test_type_may_be_any_non_empty_string(self):
        # No format whitelist in the config layer.
        qcfg = _parsed({"default": _fmt("some_future_format")})
        assert qcfg.match_module("m") == _fmt("some_future_format")

    def test_module_entry_missing_type_rejected(self):
        with pytest.raises(ds_config.DeepSpeedConfigError):
            _parsed({"module_configs": {"model.layers.0": {"group_size": 128}}})

    def test_backend_null_or_non_empty_string(self):
        _parsed({"default": _fmt("hifloat8")})  # absent backend
        _parsed({"default": {"type": "hifloat8", "backend": None}})
        _parsed({"default": {"type": "hifloat8", "backend": "torchao"}})
        with pytest.raises(ds_config.DeepSpeedConfigError):
            _parsed({"default": {"type": "hifloat8", "backend": ""}})

    def test_invalid_regex_key_rejected(self):
        with pytest.raises(ds_config.DeepSpeedConfigError):
            _parsed({"module_configs": {"re:(": _fmt("hifloat8")}})

    def test_unknown_section_key_rejected(self):
        with pytest.raises(ValidationError):
            _parsed({"formats": [_fmt("hifloat8")]})

    def test_non_dict_module_entry_rejected(self):
        with pytest.raises(ValidationError):
            _parsed({"module_configs": {"model.layers.0": "hifloat8"}})
