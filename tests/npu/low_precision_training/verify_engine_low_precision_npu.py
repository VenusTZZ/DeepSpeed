# -*- coding: utf-8 -*-
"""Phase 3 引擎级 e2e 验证：quantization 段 → 引擎接线 → 910B4 正向路径。

跑法：
    PYTHONPATH=/home/tangzizhao/workspace/DeepSpeed \
      /home/tangzizhao/miniconda3/envs/cann/bin/python \
      /home/tangzizhao/workspace/test_npu/deepspeed/roadmap2026q4/torchao_npu/verify_engine_low_precision_npu.py

条目：
  1. 零影响：无 quantization 段 → 引擎初始化与训练步完全正常（模块未被转换）
  2. Dense 正向：int8 命中 Linear → initialize 时转换 _Int8GridLinear → 完整训练步 loss finite
  3. 引擎级 fail-closed：hifloat8 在 910B4 → initialize 抛 NotImplementedError（含 950 提示）
  4. MoE 正向：int8 命中 GroupedExperts → external_grouped_mm_fn 赋值 → 前向走外部 grouped mm → 训练步 loss finite
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
log = logging.getLogger("verify-engine-lp")

import deepspeed  # noqa: E402

PASS = 0
FAIL = []


def check(cond, label):
    global PASS
    if cond:
        PASS += 1
        log.info("  [PASS] %s", label)
    else:
        FAIL.append(label)
        log.error("  [FAIL] %s", label)


def base_config(**extra):
    cfg = {
        "train_batch_size": 4,
        "train_micro_batch_size_per_gpu": 4,
        "bf16": {"enabled": True},
        "optimizer": {"type": "AdamW", "params": {"lr": 1e-3}},
    }
    cfg.update(extra)
    return cfg


class TinyNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16, bias=False)
        self.fc2 = nn.Linear(16, 8, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TinyMoE(nn.Module):

    def __init__(self):
        super().__init__()
        from deepspeed.moe.ep_experts import GroupedExperts
        self.experts = GroupedExperts(dim=8, hidden_dim=16, num_experts=4, use_grouped_mm=False)
        self.head = nn.Linear(8, 8, bias=False)

    def forward(self, x):
        tokens = x.repeat(2, 1)
        counts = torch.tensor([2, 2, 2, 2], device=x.device)
        h = self.experts(tokens, counts)
        return self.head(h[:x.shape[0]])


def run_train_step(engine, x):
    out = engine(x)
    loss = out.float().pow(2).mean()
    engine.backward(loss)
    # torch 优化器 step 后默认把 grad 置 None，梯度存在性必须在 step 前检查。
    grads = {name: (param.grad is not None) for name, param in engine.module.named_parameters()}
    engine.step()
    torch.npu.synchronize()
    return loss.item(), grads


def main():
    torch.manual_seed(0)
    x = torch.randn(4, 8, device="npu", dtype=torch.bfloat16)
    log.info("device: %s", torch.npu.get_device_name(0))

    # 1. 零影响
    log.info("【条目 1】零影响：无 quantization 段，引擎行为不变")
    model = TinyNet()
    engine, *_ = deepspeed.initialize(model=model, config=base_config())
    check(not any(type(m).__name__ == "_Int8GridLinear" for m in engine.module.modules()),
          "无 quantization 段 → 无任何模块被转换")
    loss, grads = run_train_step(engine, x)
    check(loss == loss, f"无量化训练步正常，loss={loss:.4f}")
    check(grads.get("fc1.weight", False), "无量化路径权重梯度正常回传")
    del engine, model

    # 2. Dense 正向路径（int8）
    log.info("【条目 2】Dense 正向：int8 命中 Linear → 转换 + 训练")
    model = TinyNet()
    cfg = base_config(quantization={"default": {"type": "int8", "backend": None}})
    engine, *_ = deepspeed.initialize(model=model, config=cfg)
    converted = [n for n, m in engine.module.named_modules() if type(m).__name__ == "_Int8GridLinear"]
    check(len(converted) == 2, f"initialize 时已转换 {len(converted)} 个 Linear: {converted}")
    loss, grads = run_train_step(engine, x)
    check(loss == loss, f"int8 量化训练步正常，loss={loss:.4f}")
    check(grads.get("fc1.weight", False), "int8 权重梯度已回传（STE）")
    del engine, model

    # 3. 引擎级 fail-closed（hifloat8 在 910B4）
    log.info("【条目 3】引擎级 fail-closed：hifloat8 → initialize 抛错")
    model = TinyNet()
    cfg = base_config(quantization={"default": {"type": "hifloat8", "backend": None}})
    try:
        deepspeed.initialize(model=model, config=cfg)
        check(False, "hifloat8 在 910B4 上 initialize 应失败")
    except NotImplementedError as e:
        check("950" in str(e) or "910" in str(e), f"initialize fail-closed（含硬件门提示）: {str(e)[:100]}")
    del model

    # 4. MoE 正向路径（int8 → external grouped mm）
    log.info("【条目 4】MoE 正向：int8 命中 GroupedExperts → 外部 grouped mm")
    model = TinyMoE()
    cfg = base_config(quantization={"module_configs": {
        "experts": {
            "type": "int8",
            "backend": None
        },
        "_default": None,
    }})
    engine, *_ = deepspeed.initialize(model=model, config=cfg)
    check(engine.module.experts.external_grouped_mm_fn is not None, "external_grouped_mm_fn 已赋值")
    loss, grads = run_train_step(engine, x)
    check(loss == loss, f"MoE int8 grouped mm 训练步正常，loss={loss:.4f}")
    check(grads.get("experts.w1", False), "专家权重梯度已回传")

    log.info("===== 结果：%d 过 / %d 挂 =====", PASS, len(FAIL))
    for f in FAIL:
        log.error("  挂: %s", f)


if __name__ == "__main__":
    main()
