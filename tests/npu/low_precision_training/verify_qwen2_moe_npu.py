# -*- coding: utf-8 -*-
"""真实主流 MoE 结构（Qwen2-MoE）训练验证：bf16 基线 vs int8 量化专家（910B4）。

跑法：
    PYTHONPATH=/home/tangzizhao/workspace/DeepSpeed \
      /home/tangzizhao/miniconda3/envs/cann/bin/python \
      /home/tangzizhao/workspace/test_npu/deepspeed/roadmap2026q4/torchao_npu/verify_qwen2_moe_npu.py

结构 = transformers 官方 Qwen2MoeForCausalLM（Qwen2 旋转位置编码 + SwiGLU 专家 +
TopK 路由），尺寸经 config 缩小（2 层 / hidden 512 / 8 专家 / top-2 / vocab 2048，
约 55.6M 参数），权重随机初始化不下载。量化命中融合专家容器（STE 前向补丁）与
共享专家投影（_Int8GridLinear 替换），同种子同数据各训 5 步 + 同权重前向对拍。
"""

import logging
import os
import time

os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("LOCAL_RANK", "0")

import torch  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("verify-qwen2-moe")

import deepspeed  # noqa: E402
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM  # noqa: E402

BATCH, SEQ, STEPS = 2, 128, 5


def phase(i, total, title):
    log.info("========== [阶段 %d/%d] %s ==========", i, total, title)


def build_model():
    config = Qwen2MoeConfig(num_hidden_layers=2,
                            hidden_size=512,
                            intermediate_size=1024,
                            num_attention_heads=8,
                            num_key_value_heads=4,
                            vocab_size=2048,
                            max_position_embeddings=SEQ * 4,
                            num_experts=8,
                            num_experts_per_tok=2,
                            attention_dropout=0.0)
    return Qwen2MoeForCausalLM(config)


def base_config(**extra):
    cfg = {
        "train_batch_size": BATCH,
        "train_micro_batch_size_per_gpu": BATCH,
        "bf16": {"enabled": True},
        "optimizer": {"type": "AdamW", "params": {"lr": 1e-3}},
    }
    cfg.update(extra)
    return cfg


def describe_quantization(engine):
    """打印量化计划：哪些 Linear 被替换、哪些融合容器被打补丁。"""
    replaced = [n for n, m in engine.module.named_modules() if type(m).__name__ == "_Int8GridLinear"]
    patched = [n for n, m in engine.module.named_modules() if hasattr(m, "_int8_grid_ste_group_size")]
    log.info("  [量化计划] Linear -> _Int8GridLinear 替换 %d 个", len(replaced))
    for n in replaced:
        log.info("      - %s", n)
    log.info("  [量化计划] 融合专家容器 -> STE 前向补丁 %d 个", len(patched))
    for n in patched:
        log.info("      - %s", n)
    return replaced, patched


def train_arm(label, model, config, batches):
    log.info("========== %s ==========", label)
    t0 = time.time()
    engine, *_ = deepspeed.initialize(model=model, config=config)
    log.info("  [init] deepspeed.initialize 完成，耗时 %.1fs（引擎: %s）", time.time() - t0, type(engine).__name__)
    replaced, patched = describe_quantization(engine)
    if not replaced and not patched:
        log.info("  [量化计划] 本 arm 无量化（基线）")
    losses = []
    for step, ids in enumerate(batches):
        t0 = time.time()
        loss = engine(ids, labels=ids).loss
        engine.backward(loss)
        engine.step()
        torch.npu.synchronize()
        losses.append(loss.item())
        log.info("  [step %d/%d] loss=%.4f 耗时 %.1fs", step + 1, STEPS, losses[-1], time.time() - t0)
    return losses, engine


def main():
    phase(1, 6, "构建 Qwen2-MoE（官方结构，随机初始化）")
    torch.manual_seed(0)
    batches = [torch.randint(0, 2048, (BATCH, SEQ), device="npu") for _ in range(STEPS)]
    log.info("  模型: Qwen2MoeForCausalLM，2 层 / hidden 512 / 8 专家 / top-2 / vocab 2048")
    torch.manual_seed(0)
    model_a = build_model()
    torch.manual_seed(0)
    model_b = build_model()
    log.info("  参数量: %.1f M（两臂同种子同数据，数据共 %d 个 batch）",
             sum(p.numel() for p in model_a.parameters()) / 1e6, STEPS)

    phase(2, 6, "Arm A：bf16 基线训练（无量化）")
    losses_a, engine_a = train_arm("Arm A", model_a, base_config(), batches)

    phase(3, 6, "Arm B：int8 量化训练（专家容器 + 共享专家投影）")
    losses_b, _ = train_arm(
        "Arm B",
        model_b,
        base_config(quantization={"module_configs": {
            "re:model\\.layers\\..+\\.mlp\\.experts": {
                "type": "int8",
                "backend": None,
                "group_size": 32
            },
            "re:model\\.layers\\..+\\.mlp\\.shared_expert\\.(gate|up|down)_proj": {
                "type": "int8",
                "backend": None,
                "group_size": 32
            },
            "_default": None,
        }}),
        batches,
    )

    phase(4, 6, "step 对照（同种子同数据）")
    log.info("  step |  bf16  |  int8  |    Δ")
    for i, (a, b) in enumerate(zip(losses_a, losses_b)):
        log.info("  %4d | %.4f | %.4f | %+.4f", i + 1, a, b, b - a)

    phase(5, 6, "同权重前向对拍（决定性：量化前向误差的全部代价）")
    log.info("  把 Arm A 训好的权重原样装进量化模型，同一个 batch 各算一次 LM loss……")
    model_d = build_model()
    model_d.load_state_dict({k: v.detach() for k, v in model_a.state_dict().items()})
    engine_d, *_ = deepspeed.initialize(model=model_d,
                                        config=base_config(quantization={"module_configs": {
                                            "re:model\\.layers\\..+\\.mlp\\.experts": {
                                                "type": "int8",
                                                "backend": None,
                                                "group_size": 32
                                            },
                                            "re:model\\.layers\\..+\\.mlp\\.shared_expert\\.(gate|up|down)_proj": {
                                                "type": "int8",
                                                "backend": None,
                                                "group_size": 32
                                            },
                                            "_default": None,
                                        }}))
    ids = batches[-1]
    with torch.no_grad():
        loss_a = engine_a(ids, labels=ids).loss.item()
        loss_d = engine_d(ids, labels=ids).loss.item()
    rel = abs(loss_d - loss_a) / loss_a
    log.info("  同权重 LM loss：bf16=%.4f，int8=%.4f，相对差 %.3f%%", loss_a, loss_d, rel * 100)

    phase(6, 6, "结论")
    if rel < 0.02:
        log.info("  [PASS] 主流结构（Qwen2-MoE）上精度方法 OK：同权重下量化前向的 loss 代价 < 2%%")
    else:
        log.info("  [WARN] 同权重 loss 差距偏大，需要检查量化实现")


if __name__ == "__main__":
    main()
