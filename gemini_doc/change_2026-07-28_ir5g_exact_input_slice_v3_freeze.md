# 变更记录：IR-5G exact-input-slice residual final v3 freeze

> 日期：2026-07-28
> 分支：`feat/compiler-ir-stack-v1`
> 父基线：`db8f17d`
> 状态：v2 方法学缺陷已修复；v3 final identities 未执行

## 1. 唯一修复范围

IR-5F 证明 v2 的 fixed-single 不是 batch 第一 query，因为同 seed、不同 tensor shape 的
`torch.randn` 不保证前缀关系。v3 不改 backend、planner、预算、p90/Pareto 阈值或 final
shape，只修复输入身份：

1. convolutional workload builder 接受可选 `input_center`；
2. override 必须与声明 shape 完全一致、dtype 必须为 float32，并复制后绑定；
3. runner 先构建 batched workload；
4. fixed-single 使用
   `batched_prepared.input_spec.center[:1].detach().clone()`；
5. 在 final-bound semantic gate 之前先以 `torch.equal` 验证 input center exact identity；
6. `split.json` 固化
   `single_query_binding_contract=exact_clone_of_batched_input_center_first_query`。

strict final-bound allclose、from-forward-trace baseline 和所有性能门禁均未放宽。

## 2. residual-final-v3 冻结内容

- schema：`boundflow.ir5-residual-final-artifact/v3`；
- calibration：沿用 v2 前已允许的 chain-CNN `7201/7202`；
- final shapes：沿用 v2 的 gray `4×1×14×14, block=5, output=12` 与 color
  `4×3×18×18, block=7, output=12`；
- fresh identities：
  - `final-residual-gray-v3`, seed `7501`；
  - `final-residual-color-v3`, seed `7502`；
- suite：CUDA-only、from-forward-trace、8 contexts × 6 policies；
- hard fields：Global p90 `≤1.20×`、两个 workload 均存在 compiler latency-memory
  Pareto；multi-budget switch 独立报告。

v2 已写出的 timing 没有被读取，v3 shape/backend/budget 未按 v2 performance 调整。
`7501/7502` 在本 protocol commit 前不得执行。

## 3. 非 final 验证

- 用独立 dummy identity/seed `76` 和奇数 image size 5 验证：
  - explicit single center 与 batched query zero `torch.equal`；
  - residual reference final lower/upper 完全按既有 allclose 门禁对齐；
- residual/fair/protocol 定向测试：`10 passed`；
- Mypy：修改的 builder/runner 零问题；
- 旧 family-fair-v1 replay 保持兼容（提交前复核）。

这些测试不构建 `7501/7502`，不消费 v3 final。

## 4. 下一步

提交 v3 protocol 后，只允许一次：

```bash
python scripts/run_ir5_family_fair_artifact.py generate \
  --suite residual-final-v3 \
  --out-dir artifacts/ir5/residual-final-v3-20260728 \
  --device cuda --warm-samples 9
```

随后执行 integrity + semantic replay，并按冻结 summary 原样判定。任何失败都必须记录，
不得再次旋转 final 或根据其数据调参。
