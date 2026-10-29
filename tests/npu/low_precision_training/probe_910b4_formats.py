# -*- coding: utf-8 -*-
"""910B4 低精度格式可用性探针 v2（找正向路径测试能用的格式）。

跑法：
    PYTHONPATH=/home/tangzizhao/workspace/DeepSpeed \
      /home/tangzizhao/miniconda3/envs/cann/bin/python \
      /home/tangzizhao/workspace/test_npu/deepspeed/roadmap2026q4/torchao_npu/probe_910b4_formats.py

每个候选：量化一个 8x8 Linear → 前向 → autograd 反传（dx/dw）→ 报告可用性。
"""

import logging

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("probe-formats")

DEV = "npu"
DT = torch.bfloat16


def run_case(name, setup_fn):
    torch.manual_seed(0)
    try:
        m = nn.Linear(8, 8, bias=False).to(DEV).to(DT)
        ref_w = m.weight.data.clone()
        setup_fn(m)
        x = torch.randn(4, 8, device=DEV, dtype=DT, requires_grad=True)
        out = m(x)
        torch.npu.synchronize()
        with torch.no_grad():
            ref = x.detach() @ ref_w.t()
        diff = (out.detach().float() - ref.float()).abs().max().item()
        grads = torch.autograd.grad(out.sum(), (x, m.weight), allow_unused=True, retain_graph=True)
        dx_ok = grads[0] is not None
        dw_ok = grads[1] is not None
        log.info("[%s] fwd=OK diff=%.4f dx=%s dw=%s weight_type=%s", name, diff, dx_ok, dw_ok, type(m.weight).__name__)
        return True, dx_ok, dw_ok
    except Exception as e:
        log.info("[%s] FAIL: %s: %s", name, type(e).__name__, str(e)[:140])
        return False, False, False


def main():
    log.info("device: %s，torch %s", torch.npu.get_device_name(0), torch.__version__)

    from torchao.quantization import quantize_

    # 1. torchao 官方 int8 weight-only（fwd 可用、autograd 断链嫌疑）
    from torchao.quantization import Int8WeightOnlyConfig
    run_case("torchao.Int8WeightOnly", lambda m: quantize_(m, Int8WeightOnlyConfig()))

    # 2. torchao 官方 int4 weight-only
    from torchao.quantization import Int4WeightOnlyConfig
    run_case("torchao.Int4WeightOnly", lambda m: quantize_(m, Int4WeightOnlyConfig()))

    # 3. torchao QAT（fake-quant 训练语义，无自研内核嫌疑）
    try:
        from torchao.quantization.qat import IntXQuantizationAwareTrainingConfig, IntxFakeQuantizeConfig

        def qat_int8(m):
            act_cfg = IntxFakeQuantizeConfig(dtype=torch.int8, granularity="per_token")
            w_cfg = IntxFakeQuantizeConfig(dtype=torch.int8, granularity="per_channel")
            quantize_(m, IntXQuantizationAwareTrainingConfig(activation_config=act_cfg, weight_config=w_cfg))

        run_case("torchao.IntXQAT(int8)", qat_int8)
    except Exception as e:
        log.info("[torchao.IntXQAT] 构造 FAIL: %s: %s", type(e).__name__, str(e)[:140])

    # 4. torchao_npu W4A8 QAT（docstring 自称 950 内核，确认 fail-closed）
    try:
        from torchao_npu.w4a8 import W4A8QATConfig
        run_case("torchao_npu.W4A8QAT", lambda m: quantize_(m, W4A8QATConfig()))
    except Exception as e:
        log.info("[torchao_npu.W4A8QAT] 导入 FAIL: %s: %s", type(e).__name__, str(e)[:140])

    # 5. 手工 int8 存储 + bf16 计算（纯 torch，autograd 必然完整）
    def manual_int8(m):
        w = m.weight.data
        scale = w.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-6) / 127.0
        q = torch.round(w / scale).clamp(-128, 127).to(torch.int8)
        m._q = q
        m._scale = scale
        m.weight.data.copy_((q.float() * scale))    # 存储量化语义：master 权重 = 反量化后的网格值
        # 前向仍是 bf16 matmul；网格化权重可训（梯度会更新 master，再投影回网格由 QAT 语义决定，此处仅探数值链路）

    run_case("manual_int8_storage(bf16-math)", manual_int8)

    log.info("===== 探针结束 =====")


if __name__ == "__main__":
    main()
