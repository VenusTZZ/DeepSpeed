# -*- coding: utf-8 -*-
"""HF Trainer 风格入口验证：transformers Trainer + DeepSpeed JSON 配置跑 MoE 低精度训练（910B4）。

跑法（默认以子进程隔离各跑 bf16 / int8 两臂并对照）：
    PYTHONPATH=/home/tangzizhao/workspace/DeepSpeed \
      /home/tangzizhao/miniconda3/envs/cann/bin/python \
      /home/tangzizhao/workspace/test_npu/deepspeed/roadmap2026q4/torchao_npu/verify_hf_trainer_moe_npu.py

只跑单臂：
    ... verify_hf_trainer_moe_npu.py --arm int8 --steps 10

验证内容：按 transformers 官方 Trainer 用法——Qwen2MoeForCausalLM（随机初始化，不下载）
+ 定长 token 数据集 + TrainingArguments(deepspeed="ds_config.json") + Trainer.train()。
量化段直接写在 ds_config.json 里（配置进 JSON 文件本身就是 transformers 风格的一部分），
经 HfTrainerDeepSpeedConfig → accelerate DeepSpeedPlugin → deepspeed.initialize 原样透传。

判定用「记忆化冒烟」：任务是 copy 任务（input_ids == labels），训练集只有 5 个 batch、
每轮重复（30 步 = 6 轮）；评测批与训练集第一批评测同 seed 同数据，模型每轮都训它，
所以评测 loss 下降快且确定（等价于验证 前向/反向/优化器更新 整条链路真的在工作）。
大模型 + MoE 路由噪声下 copy 任务 30 步来不及泛化，新鲜随机批上的 loss 会停在
ln(vocab)≈7.62——所以判定只看评测批 loss，不看新鲜批。
判定：评测 loss 明显下降 + 量化计划生效（Linear 替换 / 融合专家容器 STE 补丁）。
"""

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("LOCAL_RANK", "0")

import torch  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM, Trainer, TrainingArguments  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("verify-hf-trainer-moe")

BATCH, SEQ, STEPS = 2, 128, 30
VOCAB = 2048


def build_model():
    config = Qwen2MoeConfig(num_hidden_layers=2,
                            hidden_size=512,
                            intermediate_size=1024,
                            num_attention_heads=8,
                            num_key_value_heads=4,
                            vocab_size=VOCAB,
                            max_position_embeddings=SEQ * 4,
                            num_experts=8,
                            num_experts_per_tok=2,
                            attention_dropout=0.0)
    return Qwen2MoeForCausalLM(config)


class TinyTokenDataset(Dataset):
    """定长随机 token 序列（input_ids == labels），无需分词器、不下载任何数据。"""

    def __init__(self, n_samples, seq_len, vocab_size, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.ids = [torch.randint(1, vocab_size, (seq_len, ), generator=g) for _ in range(n_samples)]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "labels": self.ids[i]}


def ds_config_dict(quantized):
    """transformers 风格：DeepSpeed 配置就是一份 JSON。量化段按需写进同一份文件。"""
    cfg = {
        "train_batch_size": BATCH,
        "train_micro_batch_size_per_gpu": BATCH,
        "gradient_accumulation_steps": 1,
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 2
        },
    }
    if quantized:
        cfg["quantization"] = {
            "module_configs": {
                r"re:model\.layers\..+\.mlp\.experts": {
                    "type": "int8",
                    "backend": "torchao",
                    "group_size": 32
                },
                r"re:model\.layers\..+\.mlp\.shared_expert\.(gate|up|down)_proj": {
                    "type": "int8",
                    "backend": "torchao",
                    "group_size": 32
                },
                "_default": None,
            }
        }
    return cfg


def write_ds_config(quantized):
    path = os.path.join(tempfile.mkdtemp(prefix="hf-ds-config-"), "ds_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ds_config_dict(quantized), f)
    return path


def describe_quantization(module):
    """量化计划：Linear 替换数、融合专家容器 STE 补丁数。"""
    replaced = [n for n, m in module.named_modules() if type(m).__name__ == "_Int8GridLinear"]
    patched = [n for n, m in module.named_modules() if hasattr(m, "_int8_grid_ste_group_size")]
    return len(replaced), len(patched)


def run_arm(arm, steps, out_path=None):
    """跑一条臂：bf16 基线（配置无量化段）或 int8 量化（配置带量化段）。"""
    torch.manual_seed(0)
    model = build_model()
    # 5 个 batch 反复训（steps 步 = steps/5 轮）；评测批与训练第一批评测同 seed 同数据。
    train_dataset = TinyTokenDataset(5 * BATCH, SEQ, VOCAB, seed=0)
    eval_dataset = TinyTokenDataset(BATCH, SEQ, VOCAB, seed=0)
    cfg_path = write_ds_config(quantized=(arm == "int8"))

    log.info("========== arm=%s：TrainingArguments(deepspeed=%s) ==========", arm, cfg_path)
    args = TrainingArguments(output_dir=tempfile.mkdtemp(prefix=f"hf-trainer-{arm}-"),
                             deepspeed=cfg_path,
                             per_device_train_batch_size=BATCH,
                             per_device_eval_batch_size=BATCH,
                             gradient_accumulation_steps=1,
                             learning_rate=1e-3,
                             bf16=True,
                             max_steps=steps,
                             # 数据集每 epoch 只有 5 个 batch，epochs 要给足，否则 Trainer 会在
                             # 15 步（3 个 epoch）处提前收工，静默少于 max_steps。
                             num_train_epochs=math.ceil(steps / 5),
                             logging_steps=1,
                             eval_strategy="steps",
                             eval_steps=5,
                             report_to=[],
                             save_strategy="no",
                             disable_tqdm=True,
                             dataloader_num_workers=0,
                             seed=0)
    trainer = Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset)
    trainer.train()

    engine = trainer.model_wrapped
    module = getattr(engine, "module", engine)
    replaced, patched = describe_quantization(module)
    log.info("  [量化计划] Linear -> _Int8GridLinear 替换 %d 个 / 融合专家容器 STE 补丁 %d 个", replaced, patched)
    for n, m in module.named_modules():
        if type(m).__name__ == "_Int8GridLinear":
            log.info("      - %s", n)

    train_losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    eval_losses = [h["eval_loss"] for h in trainer.state.log_history if "eval_loss" in h]
    log.info("  [%s] 训练 loss %s（5 个 batch 反复训，记忆化后应下降）", arm,
             " ".join(f"{l:.4f}" for l in train_losses))
    log.info("  [%s] 评测 loss %s", arm, " ".join(f"{l:.4f}" for l in eval_losses))

    ok = (bool(eval_losses) and all(torch.isfinite(torch.tensor(l)).item() for l in eval_losses)
          and eval_losses[-1] < eval_losses[0] * 0.95)
    if arm == "int8" and not (replaced or patched):
        ok = False
    summary = {
        "arm": arm,
        "losses": train_losses,
        "eval_losses": eval_losses,
        "replaced": replaced,
        "patched": patched,
        "ok": ok,
    }
    log.info("  [%s] 判定 %s（评测 loss 首=%.4f 末=%.4f）", arm, "PASS" if ok else "FAIL",
             eval_losses[0] if eval_losses else float("nan"), eval_losses[-1] if eval_losses else float("nan"))

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f)
    return 0 if ok else 1


def run_subprocess(arm, steps):
    """子进程隔离跑一条臂（每次训练一个干净进程，互不干扰）。"""
    out_dir = tempfile.mkdtemp(prefix=f"hf-arm-{arm}-")
    out_path = os.path.join(out_dir, "result.json")
    cmd = [sys.executable, os.path.abspath(__file__), "--arm", arm, "--steps", str(steps), "--out", out_path]
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "/home/tangzizhao/workspace/DeepSpeed")
    proc = subprocess.run(cmd, env=env)
    with open(out_path, encoding="utf-8") as f:
        result = json.load(f)
    result["returncode"] = proc.returncode
    return result


def main():
    parser = argparse.ArgumentParser(description="HF Trainer 风格 MoE 低精度训练验证")
    parser.add_argument("--arm", choices=["bf16", "int8", "both"], default="both")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--out", default=None, help="单臂模式：结果 JSON 输出路径")
    args = parser.parse_args()

    if args.arm != "both":
        sys.exit(run_arm(args.arm, args.steps, args.out))

    log.info("========== 两臂对照（子进程隔离，各 %d 步）==========", args.steps)
    results = {arm: run_subprocess(arm, args.steps) for arm in ("bf16", "int8")}
    log.info("  arm   | 评测 loss 首 | 评测 loss 末 |   下降   | Linear替换 | 容器补丁 | 退出码")
    for arm in ("bf16", "int8"):
        r = results[arm]
        first, last = r["eval_losses"][0], r["eval_losses"][-1]
        log.info("  %-5s | %12.4f | %12.4f | %+.4f | %10d | %8d | %d", arm, first, last, last - first, r["replaced"],
                 r["patched"], r["returncode"])

    both_ok = all(r["ok"] and r["returncode"] == 0 for r in results.values())
    int8_applied = results["int8"]["replaced"] + results["int8"]["patched"] > 0
    if both_ok and int8_applied:
        log.info("[PASS] HF Trainer 入口 OK：bf16 与 int8 两臂评测 loss 均明显下降，且 int8 量化计划生效")
    else:
        log.info("[FAIL] 见上表：两臂退出码=%s，int8 量化计划生效=%s",
                 [r["returncode"] for r in results.values()], int8_applied)
        sys.exit(1)


if __name__ == "__main__":
    main()
