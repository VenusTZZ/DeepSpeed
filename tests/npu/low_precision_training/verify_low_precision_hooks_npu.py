# -*- coding: utf-8 -*-
"""Phase 2 契约验证：accelerator 低精度钩子的 NPU 实机行为（910B4）。

跑法：
    PYTHONPATH=/home/tangzizhao/workspace/DeepSpeed \
      /home/tangzizhao/miniconda3/envs/cann/bin/python \
      /home/tangzizhao/workspace/test_npu/deepspeed/roadmap2026q4/torchao_npu/verify_low_precision_hooks_npu.py

契约条目（对应仓库 accelerator/npu_accelerator.py + op_builder/npu/low_precision.py）：
  1. 抽象钩子经 accelerator 分发到 NPU 实现（三个方法都存在且可调用）
  2. assert 钩子 fail-closed：hifloat8 在 910B4 上定向报错（含 950/910_95 硬件门提示）
  3. assert 钩子 fail-closed：未实现格式（int4wo）定向报错
  4. grouped mm 钩子退化纪律：hifloat8 不可用 → 返回可用的 BF16 grouped mm（带 warning），
     fwd/dx/dw 与逐专家 for-loop 位级一致（含零 token 专家）
  5. convert 钩子：非 Linear 拒绝（TypeError）；Linear 构造替换为 HiFloat8Linear（仅构造冒烟，910B4 不做前向）
"""

import logging

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("verify-low-precision-hooks")

from deepspeed.accelerator import get_accelerator  # noqa: E402

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
    torch.manual_seed(42)
    acc = get_accelerator()
    log.info("accelerator: %s，device: %s", type(acc).__name__, torch.npu.get_device_name(0))

    # 1. 钩子经 accelerator 分发
    log.info("【条目 1】钩子分发")
    for hook in ("assert_low_precision_training_supported", "convert_linears_to_low_precision", "get_grouped_mm_fn"):
        check(callable(getattr(acc, hook, None)), f"accelerator 上存在 {hook}")

    # 2. hifloat8 在 910B4 上 fail-closed
    log.info("【条目 2】assert(hifloat8) 在 910B4 上 fail-closed")
    try:
        acc.assert_low_precision_training_supported(hifloat8_cfg())
        check(False, "assert(hifloat8) 在 910B4 上应报错")
    except NotImplementedError as e:
        msg = str(e)
        check("950" in msg or "910" in msg, f"fail-closed 且提示硬件门（950/910_95）: {msg[:110]}...")

    # 3. 未实现格式 fail-closed
    log.info("【条目 3】assert(int4wo) 未实现格式 fail-closed")
    try:
        acc.assert_low_precision_training_supported({"type": "int4wo"})
        check(False, "assert(int4wo) 应报错（未实现格式）")
    except NotImplementedError as e:
        check("int4wo" in str(e) and "hifloat8" in str(e), f"fail-closed 且列出已实现格式: {str(e)[:100]}...")

    # 4. grouped mm 退化纪律 + 数值契约
    log.info("【条目 4】grouped mm：hifloat8 不可用 → 退化 BF16，数值与 for-loop 位级一致")
    fn = None
    try:
        fn = acc.get_grouped_mm_fn(hifloat8_cfg())
    except Exception as e:
        check(False, f"get_grouped_mm_fn(hifloat8) 应返回可用 callable，却抛 {type(e).__name__}")
    if fn is not None:
        check(callable(fn), "get_grouped_mm_fn(hifloat8) 返回 callable（退化纪律，上方应有 warning）")
        H, O = 16, 32
        tokens = [5, 3, 0, 7]    # 专家 2 零 token
        offsets = torch.cumsum(torch.tensor([0] + tokens, dtype=torch.int64), dim=0)[1:]
        x = torch.randn(sum(tokens), H, device="npu", dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(4, H, O, device="npu", dtype=torch.bfloat16, requires_grad=True)
        grad_out = torch.randn(sum(tokens), O, device="npu", dtype=torch.bfloat16)

        out_k = fn(x, w, offsets)
        out_r = ref_grouped_mm(x, w, offsets)
        check(torch.equal(out_k, out_r), "fwd 与逐专家 for-loop 位级一致（含零 token 专家）")

        dx_k, dw_k = torch.autograd.grad(out_k, (x, w), grad_out)
        dx_r, dw_r = torch.autograd.grad(out_r, (x, w), grad_out)
        check(torch.equal(dx_k, dx_r), "dx 位级一致")
        check(torch.equal(dw_k, dw_r), "dw 位级一致")
        check(torch.equal(dw_k[2], torch.zeros_like(dw_k[2])), "零 token 专家的 dw 为全零")

    # 5. convert 钩子
    log.info("【条目 5】convert 钩子：拒绝非 Linear + Linear 构造替换")
    holder = torch.nn.Module()
    holder.proj = torch.nn.Linear(8, 8)
    holder.act = torch.nn.ReLU()
    try:
        acc.convert_linears_to_low_precision(holder, ["act"], hifloat8_cfg())
        check(False, "convert(非 Linear) 应报 TypeError")
    except TypeError as e:
        check("act" in str(e), f"convert(非 Linear) fail-closed: {str(e)[:80]}")
    try:
        converted = acc.convert_linears_to_low_precision(holder, ["proj"], hifloat8_cfg())
        check(converted == ["proj"], "convert 返回已转换模块名单")
        check(type(holder.proj).__name__ == "HiFloat8Linear", f"Linear 已替换为 {type(holder.proj).__name__}")
    except Exception as e:
        check(False, f"convert(Linear) 构造替换失败: {type(e).__name__}: {str(e)[:120]}")

    # 6. int8 格式正向路径（910B4 可跑的训练格式，纯 torch + STE）
    log.info("【条目 6】int8 格式（910B4 正向路径）：assert 通过 + grouped mm STE + convert 训练语义")
    int8_cfg = {"type": "int8", "backend": "torchao"}
    try:
        acc.assert_low_precision_training_supported(int8_cfg)
        check(True, "assert(int8) 通过（纯 torch 格式，无硬件门）")
    except Exception as e:
        check(False, f"assert(int8) 应通过，却抛 {type(e).__name__}: {str(e)[:100]}")

    def ste_quant(w):
        scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 127.0
        return torch.round(w / scale) * scale

    def ste_ref_grouped_mm(x, weight, offsets):
        outs, prev = [], 0
        for i, end in enumerate(offsets.tolist()):
            outs.append(x[prev:end] @ ste_quant(weight[i]))
            prev = end
        return torch.cat(outs, dim=0)

    try:
        fn_i8 = acc.get_grouped_mm_fn(int8_cfg)
        check(callable(fn_i8), "get_grouped_mm_fn(int8) 返回 callable")
        H, O = 16, 32
        offsets = torch.cumsum(torch.tensor([0, 5, 3, 0, 7], dtype=torch.int64), dim=0)[1:]
        x = torch.randn(15, H, device="npu", dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(4, H, O, device="npu", dtype=torch.bfloat16, requires_grad=True)
        grad_out = torch.randn(15, O, device="npu", dtype=torch.bfloat16)
        out_k = fn_i8(x, w, offsets)
        out_r = ste_ref_grouped_mm(x, w, offsets)
        check(torch.equal(out_k, out_r), "int8 grouped mm fwd 与 STE 参考位级一致")
        dx_k, dw_k = torch.autograd.grad(out_k, (x, w), grad_out)
        dx_r, dw_r = torch.autograd.grad(out_r, (x, w), grad_out)
        check(torch.equal(dx_k, dx_r), "int8 grouped mm dx 位级一致")
        check(torch.equal(dw_k, dw_r), "int8 grouped mm dw 位级一致（梯度过 STE 流回权重）")
        check(dw_k.abs().sum() > 0, "int8 grouped mm 权重梯度非零（可训练）")
    except Exception as e:
        check(False, f"int8 grouped mm 失败: {type(e).__name__}: {str(e)[:120]}")

    try:
        holder8 = torch.nn.Module()
        holder8.proj = torch.nn.Linear(8, 8, bias=False).to("npu").to(torch.bfloat16)
        ref_w = holder8.proj.weight.data.clone()
        converted = acc.convert_linears_to_low_precision(holder8, ["proj"], int8_cfg)
        check(converted == ["proj"] and type(holder8.proj).__name__ == "_Int8GridLinear",
              f"convert(int8) Linear 替换为 {type(holder8.proj).__name__}")
        xin = torch.randn(4, 8, device="npu", dtype=torch.bfloat16, requires_grad=True)
        out_q = holder8.proj(xin)
        ref_out = xin @ ref_w.t()
        rel_err = ((out_q - ref_out).abs().max() / ref_out.abs().max()).item()
        check(rel_err < 0.01, f"convert(int8) 前向相对误差 {rel_err:.4f} < 1%%（int8 网格量级）")
        dx, dw = torch.autograd.grad(out_q.sum(), (xin, holder8.proj.weight))
        check(dx is not None and dw is not None and dw.abs().sum() > 0, "convert(int8) dx/dw 可反传（可训练）")
    except Exception as e:
        check(False, f"convert(int8) 失败: {type(e).__name__}: {str(e)[:120]}")

    # 7. int8 group_size 变体
    log.info("【条目 7】int8 group_size 变体：分组网格 + 参数校验")
    try:
        fn_g = acc.get_grouped_mm_fn({"type": "int8", "backend": "torchao", "group_size": 8})
        check(callable(fn_g), "get_grouped_mm_fn(int8, group_size=8) 返回 callable")
        H, O = 16, 32
        offsets = torch.cumsum(torch.tensor([0, 5, 3, 0, 7], dtype=torch.int64), dim=0)[1:]
        x = torch.randn(15, H, device="npu", dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(4, H, O, device="npu", dtype=torch.bfloat16, requires_grad=True)
        grad_out = torch.randn(15, O, device="npu", dtype=torch.bfloat16)
        out_k = fn_g(x, w, offsets)

        def ste_quant_g(wt, gs):
            flat = wt.reshape(-1, gs)
            scale = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 127.0
            return (torch.round(flat / scale) * scale).reshape(wt.shape)

        def ste_ref_grouped_mm_g(xx, weight, offs):
            outs, prev = [], 0
            for i, end in enumerate(offs.tolist()):
                outs.append(xx[prev:end] @ ste_quant_g(weight[i], 8))
                prev = end
            return torch.cat(outs, dim=0)

        out_r = ste_ref_grouped_mm_g(x, w, offsets)
        check(torch.equal(out_k, out_r), "group_size=8 grouped mm fwd 与 STE 参考位级一致")
        dx_k, dw_k = torch.autograd.grad(out_k, (x, w), grad_out)
        dx_r, dw_r = torch.autograd.grad(out_r, (x, w), grad_out)
        check(torch.equal(dx_k, dx_r) and torch.equal(dw_k, dw_r), "group_size=8 dx/dw 位级一致")
    except Exception as e:
        check(False, f"group_size 变体失败: {type(e).__name__}: {str(e)[:120]}")

    try:
        acc.assert_low_precision_training_supported({"type": "int8", "group_size": 0})
        check(False, "group_size=0 应报错")
    except ValueError as e:
        check("group_size" in str(e), f"group_size 非法值 fail-closed: {str(e)[:80]}")

    try:
        fn_bad = acc.get_grouped_mm_fn({"type": "int8", "group_size": 7})
        fn_bad(torch.randn(2, 4, device="npu"), torch.randn(1, 4, 32, device="npu"),
               torch.tensor([2], device="npu"))
        check(False, "group_size=7（不整除 32）调用应报错")
    except ValueError as e:
        check("group_size" in str(e), f"group_size 不整除 fail-closed: {str(e)[:80]}")

    try:
        holder_g = torch.nn.Module()
        holder_g.proj = torch.nn.Linear(8, 32, bias=False).to("npu").to(torch.bfloat16)
        acc.convert_linears_to_low_precision(holder_g, ["proj"], {"type": "int8", "group_size": 8})
        check(type(holder_g.proj).__name__ == "_Int8GridLinear" and holder_g.proj.group_size == 8,
              "convert(int8, group_size=8) 保留分组参数")
    except Exception as e:
        check(False, f"convert group_size 失败: {type(e).__name__}: {str(e)[:120]}")

    log.info("===== 结果：%d 过 / %d 挂 =====", PASS, len(FAIL))
    for f in FAIL:
        log.error("  挂: %s", f)


if __name__ == "__main__":
    main()
