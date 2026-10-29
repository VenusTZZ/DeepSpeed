# -*- coding: utf-8 -*-
"""Phase 2 演示：低精度训练三钩子的解耦与实机行为（910B4）。

跑法：
    PYTHONPATH=/home/tangzizhao/workspace/DeepSpeed \
      /home/tangzizhao/miniconda3/envs/cann/bin/python \
      /home/tangzizhao/workspace/test_npu/deepspeed/roadmap2026q4/torchao_npu/demo_low_precision_hooks.py

脚本沿"配置 → 抽象钩子 → 厂商实现"三级链路走一遍，每级打日志：
  第 0 幕  三级解耦长什么样（config 层 / accelerator 抽象 / vendor 区）
  第 1 幕  config 层收了、后端不认的格式（int4wo）→ assert 钩子 fail-closed
  第 2 幕  hifloat8 硬件门（910B4 vs 950）→ 探针 fail-closed，报错带原始证据
  第 3 幕  grouped mm 退化纪律 → 返回可用 callable，数值与 for-loop 位级一致
  第 4 幕  convert 钩子 → 拒绝非 Linear + Linear 替换为 HiFloat8Linear
"""

import logging

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("demo-low-precision")

from deepspeed.accelerator import get_accelerator  # noqa: E402
from deepspeed.runtime.config import DeepSpeedConfig  # noqa: E402


def hifloat8_cfg():
    return {"type": "hifloat8", "backend": "torchao"}


def ref_grouped_mm(x, weight, offsets):
    """逐专家 for-loop 参考实现（torch matmul，自带 autograd）。"""
    outs, prev = [], 0
    for i, end in enumerate(offsets.tolist()):
        outs.append(x[prev:end] @ weight[i])
        prev = end
    return torch.cat(outs, dim=0)


def main():
    torch.manual_seed(7)
    acc = get_accelerator()
    log.info("device: %s，accelerator 实现: %s", torch.npu.get_device_name(0), type(acc).__name__)

    # ---------- 第 0 幕：三级解耦 ----------
    log.info("【第 0 幕】三级解耦：config 层 → accelerator 抽象 → vendor 区")
    ds_config = {
        "train_batch_size": 64,
        "bf16": {"enabled": True},
        "quantization": {
            "module_configs": {
                "re:model\\.decoder\\.layers\\..+\\.moe\\.experts\\.\\d+": hifloat8_cfg(),
                "_default": None,
            },
        },
    }
    qcfg = DeepSpeedConfig(ds_config).quantization_config
    fqn = "model.decoder.layers.1.moe.experts.0"
    fmt = qcfg.match_module(fqn)
    log.info("① config 层：match_module(%r) -> %s", fqn, fmt)
    log.info("   （config 层只认识字符串与 type 字段，不认识任何量化库）")
    log.info("② accelerator 抽象：三钩子挂在 %s 上，未实现的后端继承默认 raise", type(acc).__name__)
    log.info("③ vendor 区：钩子经 op_builder_dir() 惰性加载 op_builder/npu/low_precision.py，")
    log.info("   全仓唯一的 torchao_npu import 点，且全部惰性")

    # ---------- 第 1 幕：后端不认的格式 ----------
    log.info("【第 1 幕】config 层收了、后端不认的格式（int4wo）→ fail-closed")
    try:
        acc.assert_low_precision_training_supported({"type": "int4wo"})
    except NotImplementedError as e:
        log.error("  assert 钩子拒绝：%s", str(e)[:110])

    # ---------- 第 2 幕：硬件门 ----------
    log.info("【第 2 幕】hifloat8 硬件门：910B4 上探针 fail-closed")
    try:
        acc.assert_low_precision_training_supported(hifloat8_cfg())
        log.info("  探针通过（在 Ascend 950 上才会走到这里）")
    except NotImplementedError as e:
        log.error("  探针拒绝：%s", str(e)[:120])

    # ---------- 第 3 幕：退化纪律 ----------
    log.info("【第 3 幕】grouped mm 退化纪律：hifloat8 不可用 → BF16 grouped mm（带 warning）")
    fn = acc.get_grouped_mm_fn(hifloat8_cfg())
    log.info("  返回 callable=%s（退化纪律：永远返回可用实现，上方应有 warning）", getattr(fn, "__name__", fn))
    H, O = 16, 32
    tokens = [5, 3, 0, 7]    # 专家 2 零 token
    offsets = torch.cumsum(torch.tensor([0] + tokens, dtype=torch.int64), dim=0)[1:]
    x = torch.randn(sum(tokens), H, device="npu", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(4, H, O, device="npu", dtype=torch.bfloat16, requires_grad=True)
    grad_out = torch.randn(sum(tokens), O, device="npu", dtype=torch.bfloat16)
    out_k, out_r = fn(x, w, offsets), ref_grouped_mm(x, w, offsets)
    log.info("  fwd 位级一致：%s（含零 token 专家）", torch.equal(out_k, out_r))
    dx_k, dw_k = torch.autograd.grad(out_k, (x, w), grad_out)
    dx_r, dw_r = torch.autograd.grad(out_r, (x, w), grad_out)
    log.info("  dx 位级一致：%s", torch.equal(dx_k, dx_r))
    log.info("  dw 位级一致：%s（零 token 专家行为全零）", torch.equal(dw_k, dw_r))

    # ---------- 第 4 幕：convert ----------
    log.info("【第 4 幕】convert 钩子：拒绝非 Linear + Linear 替换")
    holder = torch.nn.Module()
    holder.proj = torch.nn.Linear(8, 8)
    holder.act = torch.nn.ReLU()
    try:
        acc.convert_linears_to_low_precision(holder, ["act"], hifloat8_cfg())
    except TypeError as e:
        log.error("  非 Linear 被拒：%s", str(e)[:80])
    acc.convert_linears_to_low_precision(holder, ["proj"], hifloat8_cfg())
    log.info("  Linear 已替换：type(holder.proj) = %s（910B4 只做构造冒烟，前向需 950）", type(holder.proj).__name__)

    # ---------- 解耦说明 ----------
    log.info("===== 解耦说明 =====")
    log.info("1. config 层（Phase 1）只做匹配：FQN 进、格式配置 dict 出，type 无白名单——int4wo 照样收，")
    log.info("   后端认不认是钩子的事（第 1 幕）。")
    log.info("2. accelerator 抽象（本层）只定接口：默认实现 raise NotImplementedError——")
    log.info("   CUDA/XPU 没实现低精度就得到干净报错，公共层不需要任何 vendor 代码。")
    log.info("3. vendor 区（op_builder/npu/low_precision.py）只装实现：torchao_npu 的 import 全仓只此一处、")
    log.info("   且全部惰性；探针、退化、转换都在这里——换硬件/换格式只动这一层。")
    log.info("4. 引擎（Phase 3）将来只调 get_accelerator().<钩子>，对 1-3 层的任何细节无感知。")
    log.info("5. 硬件门证据（第 2 幕）：探针真实执行内核，报错带 CANN 原始信息——不是按设备名猜的。")


if __name__ == "__main__":
    main()
