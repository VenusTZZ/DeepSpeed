# -*- coding: utf-8 -*-
"""DeepSpeed `quantization` 段演示脚本（纯 CPU 可跑，不依赖 NPU/torchao）。

跑法（本机 cann 环境，PYTHONPATH 保证 import 到工作区的 DeepSpeed 当前分支
npu-torchao-quantization，而不是已安装的版本）：

    PYTHONPATH=/home/tangzizhao/workspace/DeepSpeed \
        /home/tangzizhao/miniconda3/envs/cann/bin/python \
        /home/tangzizhao/workspace/test_npu/deepspeed/roadmap2026q4/torchao_npu/demo_quantization_config.py

四幕内容：
  第 1 幕  全量简写：default 一个格式应用到所有模块
  第 2 幕  细粒度：module_configs 精确 FQN > 正则 > _default，_default: null 表示其余模块不动
  第 3 幕  fail-closed：无 _default 且未命中时定向报错
  第 4 幕  结构校验：冲突 / 缺 type / 非法正则，解析阶段就报错
"""

import logging

import torch.nn as nn

from deepspeed.runtime.config import DeepSpeedConfig, DeepSpeedConfigError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("quantization-demo")


def build_demo_model():
    """搭一棵 Qwen 风格的模块树（只关心 FQN 名字，权重只是占位）。

    结构：model.decoder.layers.<i>.{self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj}
    第 1 层额外带 MoE 专家：model.decoder.layers.1.moe.experts.<0..3>
    """
    root = nn.Module()
    model = nn.Module()
    root.model = model
    decoder = nn.Module()
    model.decoder = decoder
    layers = nn.Module()
    decoder.layers = layers
    for i in range(3):
        layer = nn.Module()
        setattr(layers, str(i), layer)
        attn = nn.Module()
        layer.self_attn = attn
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(attn, proj, nn.Linear(8, 8))
        mlp = nn.Module()
        layer.mlp = mlp
        for proj in ("gate_proj", "up_proj", "down_proj"):
            setattr(mlp, proj, nn.Linear(8, 8))
    moe = nn.Module()
    layers._modules["1"].moe = moe
    setattr(moe, "experts", nn.ModuleList([nn.Linear(8, 8) for _ in range(4)]))
    model.norm = nn.Module()
    model.lm_head = nn.Linear(8, 8)
    return root


def module_fqns(model):
    return [name for name, _ in model.named_modules() if name]


def walk_and_match(qcfg, fqns, label, note=None):
    """模拟引擎的模块选择循环：对每个 FQN 调 match_module 并打印命中结果。"""
    log.info("===== %s =====", label)
    if note:
        log.info("（%s）", note)
    for fqn in fqns:
        fmt = qcfg.match_module(fqn)
        if fmt is None:
            log.info("  %-56s -> (null) 保持顶层 bf16/fp16 精度", fqn)
        else:
            log.info("  %-56s -> %s", fqn, fmt)


def main():
    model = build_demo_model()
    fqns = module_fqns(model)
    log.info("演示模型共 %d 个模块，FQN 样例：%s ... %s", len(fqns), fqns[1], fqns[-1])

    # ---------- 第 1 幕：全量简写 ----------
    log.info("【第 1 幕】全量简写：default 一个格式应用到所有模块")
    ds_config = {
        "train_batch_size": 64,
        "bf16": {"enabled": True},
        "quantization": {
            "default": {"type": "hifloat8", "backend": None, "group_size": 128},
        },
    }
    qcfg = DeepSpeedConfig(ds_config).quantization_config
    log.info("quantization 段解析完成：enabled=%s，default=%s", qcfg.enabled, qcfg.default)
    sample = [f for f in fqns if f in ("model.decoder.layers.0.self_attn.q_proj", "model.decoder.layers.1.moe.experts.2",
                                       "model.lm_head", "model.norm")]
    walk_and_match(qcfg, sample, "第 1 幕：任意模块都命中同一个格式")

    # ---------- 第 2 幕：细粒度 ----------
    log.info("【第 2 幕】细粒度：module_configs 精确 FQN > 正则 > _default")
    ds_config = {
        "train_batch_size": 64,
        "bf16": {"enabled": True},
        "quantization": {
            "module_configs": {
                # 精确 FQN（最高优先级）：只对第 0 层的 q_proj 生效
                "model.decoder.layers.0.self_attn.q_proj": {"type": "int4wo", "group_size": 64},
                # 正则（re: 前缀，fullmatch 锚定整串）：所有层的 q/k/v_proj
                "re:model\\.decoder\\.layers\\..+\\.self_attn\\.[qkv]_proj": {"type": "hifloat8", "backend": None},
                # 正则：所有 MoE 专家模块
                "re:model\\.decoder\\.layers\\..+\\.moe\\.experts\\.\\d+": {"type": "hifloat8", "backend": None},
                # 显式出口：其余模块保持顶层 bf16，不做量化
                "_default": None,
            },
        },
    }
    qcfg = DeepSpeedConfig(ds_config).quantization_config
    walk_and_match(qcfg, fqns, "第 2 幕：不同模块各自命中；容器模块（无 leaf 的中间节点）也参与匹配，引擎只转换真正的 Linear")

    # ---------- 第 3 幕：fail-closed ----------
    log.info("【第 3 幕】fail-closed：无 _default 且未命中 → 定向报错")
    ds_config = {
        "train_batch_size": 64,
        "quantization": {
            "module_configs": {
                "re:model\\.decoder\\.layers\\..+\\.moe\\.experts\\.\\d+": {"type": "hifloat8"},
            },
        },
    }
    qcfg = DeepSpeedConfig(ds_config).quantization_config
    for fqn in ("model.decoder.layers.1.moe.experts.0", "model.lm_head"):
        try:
            log.info("  %-56s -> %s", fqn, qcfg.match_module(fqn))
        except DeepSpeedConfigError as e:
            log.error("  %-56s -> 报错（fail-closed）：%s", fqn, e)

    # ---------- 第 4 幕：结构校验 ----------
    log.info("【第 4 幕】结构校验：坏配置在解析阶段就报错")
    bad_configs = [
        ("default 与 _default 冲突", {
            "default": {"type": "hifloat8"},
            "module_configs": {"_default": {"type": "fp8"}},
        }),
        ("格式配置缺 type", {
            "default": {"group_size": 128}
        }),
        ("非法正则 key", {
            "module_configs": {"re:(": {"type": "hifloat8"}}
        }),
    ]
    for label, quant_section in bad_configs:
        try:
            DeepSpeedConfig({"train_batch_size": 64, "quantization": quant_section})
            log.info("  %-24s -> （未报错？）", label)
        except Exception as e:
            log.error("  %-24s -> %s: %s", label, type(e).__name__, e)

    # ---------- 解耦说明 ----------
    log.info("===== 解耦说明 =====")
    log.info("1. config 层只做匹配与校验：type 是任意非空字符串（int4wo/hifloat8 对它没有区别），")
    log.info("   格式参数（group_size 等）原样透传——config 层不认识任何量化库，零 vendor 依赖。")
    log.info("2. match_module() 是纯查询：FQN 进、格式配置 dict 出，不碰张量、不做转换。")
    log.info("3. 具体量化动作（内核、转换）由执行层负责：engine 选模块 -> accelerator 钩子 -> torchao 后端；")
    log.info("   将来加 FP8/int4 等新格式，只需在配置里写新的 type，config 层一行不用改。")


if __name__ == "__main__":
    main()
