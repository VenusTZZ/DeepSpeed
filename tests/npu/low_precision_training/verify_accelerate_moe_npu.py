# -*- coding: utf-8 -*-
"""accelerate 风格入口验证：DeepSpeedPlugin + Accelerator 手动循环跑 MoE 低精度训练（910B4）。

跑法（默认以子进程隔离各跑 bf16 / int8 两臂并对照）：
    PYTHONPATH=/home/tangzizhao/workspace/DeepSpeed \
      /home/tangzizhao/miniconda3/envs/cann/bin/python \
      /home/tangzizhao/workspace/test_npu/deepspeed/roadmap2026q4/torchao_npu/verify_accelerate_moe_npu.py

只跑单臂：
    ... verify_accelerate_moe_npu.py --arm int8 --steps 10

验证内容：按 accelerate 官方文档/example 的写法——DeepSpeedPlugin(hf_ds_config=JSON) 指到
DeepSpeed 配置、Accelerator(deepspeed_plugin=...) 统一 prepare(model, optimizer, dataloader)、
`with accelerator.accumulate(model)` + accelerator.backward 的手动训练循环（deepspeed 插件下
optimizer.step()/zero_grad() 是官方设计的有意 no-op，真正的 step 在 accelerator.backward 里）。
量化段写在 ds_config.json 里，DeepSpeedPlugin 原样透传给 deepspeed.initialize。

判定用「记忆化冒烟」：任务是 copy 任务（input_ids == labels），训练集只有 5 个 batch、
每轮重复（30 步 = 6 轮）；评测批与训练集第一批评测同 seed 同数据，模型每轮都训它，
所以评测 loss 下降快且确定（等价于验证 前向/反向/优化器更新 整条链路真的在工作）。
大模型 + MoE 路由噪声下 copy 任务 30 步来不及泛化，新鲜随机批上的 loss 会停在
ln(vocab)≈7.62——所以判定只看评测批 loss，不看新鲜批。
判定：评测 loss 明显下降 + 量化计划生效（Linear 替换 / 融合专家容器 STE 补丁）。
"""

import argparse
import itertools
import json
import logging
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
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM  # noqa: E402
from accelerate import Accelerator, DeepSpeedPlugin  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("verify-accelerate-moe")

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
    """accelerate 风格：DeepSpeed 配置就是一份 JSON，交给 DeepSpeedPlugin。量化段按需写入。"""
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
    path = os.path.join(tempfile.mkdtemp(prefix="acc-ds-config-"), "ds_config.json")
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # 5 个 batch 反复训（islice 跨 epoch 取满 steps 步）；评测批与训练第一批评测同 seed 同数据。
    train_dataloader = DataLoader(TinyTokenDataset(5 * BATCH, SEQ, VOCAB, seed=0), batch_size=BATCH)
    eval_batch = next(iter(DataLoader(TinyTokenDataset(BATCH, SEQ, VOCAB, seed=0), batch_size=BATCH)))
    cfg_path = write_ds_config(quantized=(arm == "int8"))

    log.info("========== arm=%s：DeepSpeedPlugin(hf_ds_config=%s) ==========", arm, cfg_path)
    plugin = DeepSpeedPlugin(hf_ds_config=cfg_path, zero_stage=2, gradient_accumulation_steps=1)
    accelerator = Accelerator(deepspeed_plugin=plugin, gradient_accumulation_steps=1)
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    eval_batch = {k: v.to(accelerator.device) for k, v in eval_batch.items()}

    def evaluate():
        model.eval()
        with torch.no_grad():
            out = model(**eval_batch)
            loss = out.loss if hasattr(out, "loss") else out
        model.train()
        return loss.item()

    losses, eval_losses = [], [evaluate()]
    log.info("  [eval step 0] loss=%.4f", eval_losses[-1])
    # cycle：dataloader 一个 epoch 只有 5 个 batch，跨 epoch 循环取满 steps 步
    #（islice 不会自动重启 epoch，不 cycle 会在 5 步后静默停止）。
    for step, batch in enumerate(itertools.islice(itertools.cycle(train_dataloader), steps)):
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs
            accelerator.backward(loss)
            optimizer.step()  # deepspeed 插件下为有意 no-op，step 在 accelerator.backward 里发生
            optimizer.zero_grad()
            losses.append(loss.item())
        torch.npu.synchronize()
        if (step + 1) % 5 == 0:
            eval_losses.append(evaluate())
            log.info("  [step %d/%d] loss=%.4f eval_loss=%.4f", step + 1, steps, losses[-1], eval_losses[-1])
        else:
            log.info("  [step %d/%d] loss=%.4f", step + 1, steps, losses[-1])
    assert len(losses) == steps, f"expected {steps} training steps, got {len(losses)}"

    module = getattr(model, "module", model)
    replaced, patched = describe_quantization(module)
    log.info("  [量化计划] Linear -> _Int8GridLinear 替换 %d 个 / 融合专家容器 STE 补丁 %d 个", replaced, patched)
    for n, m in module.named_modules():
        if type(m).__name__ == "_Int8GridLinear":
            log.info("      - %s", n)

    log.info("  [%s] 训练 loss %s（5 个 batch 反复训，记忆化后应下降）", arm,
             " ".join(f"{l:.4f}" for l in losses))
    log.info("  [%s] 评测 loss %s", arm, " ".join(f"{l:.4f}" for l in eval_losses))

    ok = (bool(eval_losses) and all(torch.isfinite(torch.tensor(l)).item() for l in eval_losses)
          and eval_losses[-1] < eval_losses[0] * 0.95)
    if arm == "int8" and not (replaced or patched):
        ok = False
    summary = {
        "arm": arm,
        "losses": losses,
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
    out_dir = tempfile.mkdtemp(prefix=f"acc-arm-{arm}-")
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
    parser = argparse.ArgumentParser(description="accelerate 风格 MoE 低精度训练验证")
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
        log.info("[PASS] accelerate 入口 OK：bf16 与 int8 两臂评测 loss 均明显下降，且 int8 量化计划生效")
    else:
        log.info("[FAIL] 见上表：两臂退出码=%s，int8 量化计划生效=%s",
                 [r["returncode"] for r in results.values()], int8_applied)
        sys.exit(1)


if __name__ == "__main__":
    main()
