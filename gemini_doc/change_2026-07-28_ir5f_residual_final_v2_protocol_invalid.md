# 变更记录：IR-5F residual-final-v2 protocol invalid

> 日期：2026-07-28
> protocol commit：`b3762bf`
> suite：`residual-final-v2`
> 判定：benchmark protocol invalid；未形成性能 artifact，`7401/7402` 永久退役

## 1. 一次性运行结果

在 clean `b3762bf` 上第一次执行：

```bash
python scripts/run_ir5_family_fair_artifact.py generate \
  --suite residual-final-v2 \
  --out-dir artifacts/ir5/residual-final-v2-20260728 \
  --device cuda --warm-samples 9
```

Runner 完成 8 条 calibration 与 8 条 held-out compiler measurement 后，在第一个
fixed-single ↔ batched-first semantic gate 处 fail closed：

```text
ValueError: fixed-single semantics differ from batched first query
```

没有生成 baseline、outcomes、summary 或 manifest，因此这不是一个完整 artifact，不能用于
任何性能 claim。已写出的 held-out timing 未被读取或用于后续决策。

## 2. 独立根因审计

对 batch 与 single builder 的静态输入做比较：

| workload | 参数 max diff | input center first-query max diff | lower max diff | upper max diff |
|---|---:|---:|---:|---:|
| `final-residual-gray-v2` | 0 | 3.73509 | 216.447 | 203.340 |
| `final-residual-color-v2` | 0 | 2.16740 | 222.121 | 222.225 |

根因是 runner 用“相同 seed + batch=1 重新调用 `torch.randn`”近似 batch 第一 query。
PyTorch normal random generation 对不同 tensor shape 不保证前缀相同；旧 chain-CNN shape
偶然通过，不能构成协议保证。

因此该失败不能归因于：

- residual `add_backward` 数值实现；
- prepared execution；
- backend candidate；
- CUDA 浮点 tolerance。

两边实际输入不同，fixed-single semantic gate 的前提没有成立。

## 3. 处置

- `residual-final-v2` 标记为 **PROTOCOL-INVALID**；
- exact final identities/seeds `7401/7402` 已消费并永久退役；
- 不放宽 allclose tolerance，不删除 semantic gate；
- 不读取 v2 held-out latency 来选择新 shape/backend/budget；
- 唯一允许修复是让 single builder 显式使用
  `batched_prepared.input_spec.center[:1].clone()`，并验证 tensor identity；
- 修复后必须升级 suite/schema、旋转到新 IDs/seeds，再形成新的 protocol commit 后一次性运行。

IR-5C3 正式 No-Go 仍是当前最后一个完整性能 artifact；IR-6 继续 blocked。
