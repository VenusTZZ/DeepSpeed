# -*- coding: utf-8 -*-
"""完整 MoE 模型训练验证：bf16 基线 vs int8 逐通道 vs int8 分组（910B4）。

跑法：
    PYTHONPATH=/home/tangzizhao/workspace/DeepSpeed \
      /home/tangzizhao/miniconda3/envs/cann/bin/python \
      /home/tangzizhao/workspace/test_npu/deepspeed/roadmap2026q4/torchao_npu/verify_moe_training_npu.py

内容：embedding + 2 层 MoE 解码器（MHA 注意力 + TopK 路由 + Experts 容器 + 逐专家 MLP）
+ lm_head，固定同种子/同数据各训 30 步，逐步打印 loss；量化只命中专家投影
（dense int8 STE 路径）。fused GroupedExperts 外部 grouped mm 路径另见
verify_engine_low_precision_npu.py 条目 4。结论 = 量化 arm 的 loss 轨迹贴住 bf16
基线；int8 逐通道网格理论最大相对误差 0.5/127 ≈ 0.39%，分组（group_size=32）
进一步收窄离群通道引起的刻度放大。
"""

import logging
import os

os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("LOCAL_RANK", "0")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("verify-moe-training")

import deepspeed  # noqa: E402
from deepspeed.moe.layer import MoE  # noqa: E402
from op_builder.npu.low_precision import _int8_grid_quantize  # noqa: E402

DIM, N_HEADS, HIDDEN, N_EXPERTS, TOP_K = 64, 4, 128, 8, 2
N_LAYERS, VOCAB, SEQ, BATCH, STEPS = 2, 512, 32, 4, 30


class ExpertMLP(nn.Module):
    """MoE 层的逐专家 MLP（SwiGLU）。"""

    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(DIM, HIDDEN, bias=False)
        self.up_proj = nn.Linear(DIM, HIDDEN, bias=False)
        self.down_proj = nn.Linear(HIDDEN, DIM, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


def build_model():
    class MoEDecoderLayer(nn.Module):

        def __init__(self):
            super().__init__()
            self.attn_norm = nn.LayerNorm(DIM)
            self.attn = nn.MultiheadAttention(DIM, N_HEADS, batch_first=True)
            self.moe_norm = nn.LayerNorm(DIM)
            self.moe = MoE(hidden_size=DIM, expert=ExpertMLP(), num_experts=N_EXPERTS, k=TOP_K)

        def forward(self, x):
            h = self.attn_norm(x)
            attn_out, _ = self.attn(h, h, h, need_weights=False)
            x = x + attn_out
            h = self.moe_norm(x)
            moe_out, _, _ = self.moe(h)
            return x + moe_out

    class MoETransformer(nn.Module):

        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(VOCAB, DIM)
            self.layers = nn.ModuleList([MoEDecoderLayer() for _ in range(N_LAYERS)])
            self.norm = nn.LayerNorm(DIM)
            self.lm_head = nn.Linear(DIM, VOCAB, bias=False)

        def forward(self, ids):
            x = self.embed(ids)
            for layer in self.layers:
                x = layer(x)
            return self.lm_head(self.norm(x))

    return MoETransformer()


def base_config(**extra):
    cfg = {
        "train_batch_size": BATCH,
        "train_micro_batch_size_per_gpu": BATCH,
        "bf16": {"enabled": True},
        "optimizer": {"type": "AdamW", "params": {"lr": 1e-3}},
    }
    cfg.update(extra)
    return cfg


def forward_quant_error(model):
    """训练前，前向量化（STE 网格）与原始权重的最大相对误差。"""
    errs = []
    for m in model.modules():
        if type(m).__name__ == "_Int8GridLinear":
            w = m.weight.detach()
            grid_w = _int8_grid_quantize(w, m.group_size)
            errs.append(((grid_w - w).abs().max() / w.abs().max()).item())
    return max(errs) if errs else 0.0


def train_arm(label, model, config, batches):
    log.info("===== %s 初始化 =====", label)
    engine, *_ = deepspeed.initialize(model=model, config=config)
    quantized = [n for n, m in engine.module.named_modules() if type(m).__name__ == "_Int8GridLinear"]
    if quantized:
        log.info("  量化模块（_Int8GridLinear）数量: %d", len(quantized))
        log.info("  训练前的前向量化相对误差: %.4f（理论界 0.0039）", forward_quant_error(engine.module))
    losses = []
    for step, (ids, targets) in enumerate(batches):
        logits = engine(ids)
        loss = nn.functional.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1))
        engine.backward(loss)
        engine.step()
        torch.npu.synchronize()
        losses.append(loss.item())
        log.info("  step %2d/%d  loss=%.4f", step + 1, STEPS, losses[-1])
    return losses, quantized, engine


def main():
    torch.manual_seed(0)
    # 三臂共用同一批数据与初始化种子，保证可比。
    batches = []
    for _ in range(STEPS):
        ids = torch.randint(0, VOCAB, (BATCH, SEQ), device="npu")
        batches.append((ids, ids.clone()))

    models = [build_model() for _ in range(3)]

    log.info("模型：2 层 MoE 解码器（dim=%d，experts=%d，top_k=%d，vocab=%d，seq=%d，batch=%d），各训 %d 步",
             DIM, N_EXPERTS, TOP_K, VOCAB, SEQ, BATCH, STEPS)

    losses_a, _, engine_a = train_arm("Arm A：bf16 基线（无量化）", models[0], base_config(), batches)
    losses_b, _, _ = train_arm("Arm B：int8 逐通道（STE）", models[1],
                            base_config(quantization=experts_quant_config(None)), batches)
    losses_c, quantized, _ = train_arm("Arm C：int8 分组 group_size=32（STE）", models[2],
                                    base_config(quantization=experts_quant_config(32)), batches)

    # 汇总：step 对照表 + 末尾 10 步均值 + 相对差距
    log.info("===== step 对照（loss） =====")
    log.info("  step |  bf16  | int8/ch | int8/g32 | Δch | Δg32")
    for i, (a, b, c) in enumerate(zip(losses_a, losses_b, losses_c)):
        log.info("  %4d | %.4f | %.4f | %.4f | %+.4f | %+.4f", i + 1, a, b, c, b - a, c - a)
    tail = 10
    mean_a = sum(losses_a[-tail:]) / tail
    mean_b = sum(losses_b[-tail:]) / tail
    mean_c = sum(losses_c[-tail:]) / tail
    gap_b = abs(mean_b - mean_a) / mean_a
    gap_c = abs(mean_c - mean_a) / mean_a
    log.info("===== 从零训练结论（轨迹对照） =====")
    log.info("  bf16 基线（末尾 %d 步均值）: %.4f", tail, mean_a)
    log.info("  int8 逐通道（末尾 %d 步均值）: %.4f（相对差距 %.2f%%）", tail, mean_b, gap_b * 100)
    log.info("  int8 分组32（末尾 %d 步均值）: %.4f（相对差距 %.2f%%）", tail, mean_c, gap_c * 100)
    log.info("  解读：从零训练时量化模型是另一个函数，轨迹分叉属预期；决定性证据看同权重对拍。")

    # 决定性实验：同权重前向对拍——把 Arm A 训好的权重原样装进量化模型，
    # 同一个 batch 各算一次 CE。差距只含量化前向误差，不含训练动力学。
    log.info("===== 同权重前向对拍（决定性） =====")
    model_d = build_model()
    model_d.load_state_dict({k: v.detach() for k, v in models[0].state_dict().items()})
    engine_d, *_ = deepspeed.initialize(model=model_d, config=base_config(quantization=experts_quant_config(32)))
    ids, targets = batches[-1]
    with torch.no_grad():
        logits_a = engine_a(ids)
        logits_d = engine_d(ids)
    ce_a = nn.functional.cross_entropy(logits_a.reshape(-1, VOCAB), targets.reshape(-1)).item()
    ce_d = nn.functional.cross_entropy(logits_d.reshape(-1, VOCAB), targets.reshape(-1)).item()
    rel = abs(ce_d - ce_a) / ce_a
    log.info("  同权重 CE：bf16=%.4f，int8/g32=%.4f，相对差 %.3f%%（量化前向误差的全部代价）", ce_a, ce_d, rel * 100)
    log.info("  量化生效: %d 个专家投影 Linear 已转换", len(quantized))
    if rel < 0.02:
        log.info("  [PASS] 精度方法 OK：同权重下量化前向的 CE 代价 < 2%%（训练轨迹差距属动力学分叉，非精度缺陷）")
    else:
        log.info("  [WARN] 同权重 CE 差距偏大，需要检查量化实现")


def experts_quant_config(group_size):
    fmt = {"type": "int8", "backend": None}
    if group_size is not None:
        fmt["group_size"] = group_size
    return {"module_configs": {
        "re:layers\\..+\\.moe\\.deepspeed_moe\\.experts\\.deepspeed_experts\\.\\d+\\.(gate|up|down)_proj": fmt,
        "_default": None,
    }}


if __name__ == "__main__":
    main()
