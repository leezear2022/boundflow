---
status: validated-mr7r-host-boundary-opportunity
updated: 2026-08-26T01:35:00+08:00
type: closure
topic: boundflow
slug: mr7r-unprofiled-host-recovery-formal-closure
stage: s01
performance_claimed: false
---

# MR7-R Unprofiled Host Recovery 正式关闭

## 1. 结论

MR7-R 按预注册协议完整执行，正式状态为：

```text
VALIDATED_MR7R_HOST_BOUNDARY_OPPORTUNITY
```

五个独立 pair / 十个 fresh process 证明：MR7 unprofiled host ledger 相对原 MR6 diagnostic worker 的
扰动落在冻结范围内，且 `FFI/DLPack/stream + layout/materialization + post-output` boundary 在 5/5 run
均同时超过 `15%` 与 `15 ms`。

本结论只开放 **GC-0/FCR-1 verification graph ABI 与 correctness 预注册**。它不表示 compiled region
已经实现，不表示已经取得 `1.912x` speedup，也不开放 timing、same-solver、query、queue 或 ASPLOS-ready
claim。`performance_claimed=false`。

## 2. 冻结身份与顺序

- source commit：`0a1e79553e216ed5c34604a235d537288fcf8e19`；
- generator commit：`5ba792e55b7d1c5e0c698b8390594fb119310174`；
- protocol hash：`b3f34cc3560cde422a58113a216bb46ddbbe41fcf042000e9a1e7b9c85248d86`；
- summary hash：`28cb5ac21b8227c720ca1c19dc2c0120cb8afad3a2004dcb18ce4d4a607c977d`；
- manifest hash：`24e7378df5d7c2034c2ea4714f2ceb82ca8788a3f807c3cff191d9d9b2375a6a`；
- raw canonical hash：`1cd206fe0bf4840f959e04ce4f17d73362186ed7a82cd2d20ef1678c4ee7526f`；
- run order：`BL/LB/BL/LB/BL`；
- baseline：已冻结 MR6 `diagnostic` worker；
- ledger：已冻结 MR7 `control` worker；
- 10 个 worker 均为独立 subprocess；
- 不启 CUPTI profiler；headline 为 host `perf_counter_ns`；CUDA event 仅核对方向。

artifact：

`artifacts/measurement-recovery/mr7r-unprofiled-host-recovery-v1/`

## 3. Perturbation 与 correctness 门禁

| 门禁 | 冻结要求 | 实测 | 判定 |
|---|---:|---:|---|
| pair/run | 5 pair / 10 fresh | 5 / 10 | PASS |
| semantic | allclose + sign exact | 5/5；max diff=`2.7418137e-6` | PASS |
| lifecycle | 30 forward / 27 backward；cache/module/stream/fallback exact | 5/5 | PASS |
| host closure | error ratio `<=2%` | 5/5；均为 0 | PASS |
| host ratio median | `[0.95,1.05]` | `1.00685785` | PASS |
| host ratio per-run | 每项 `[0.90,1.10]` | min/max=`0.94277477/1.02633524` | PASS |
| host/event direction | 5/5 | 5/5 | PASS |

ledger/baseline host ratio geomean=`0.99060697`。这表示 ledger 没有表现出系统性高于冻结门限的测量成本；
它不是 candidate/provider speedup。

### 3.1 五个 pair

| pair | ledger/baseline host | ledger/baseline event | boundary share | boundary absolute | max semantic diff |
|---:|---:|---:|---:|---:|---:|
| 0 | `1.00685785` | `1.00685655` | `20.6411%` | `23.309416 ms` | `2.741814e-6` |
| 1 | `1.00897758` | `1.00897779` | `20.4273%` | `24.683788 ms` | `7.748604e-7` |
| 2 | `0.97042065` | `0.97043665` | `20.2499%` | `24.779672 ms` | `2.242625e-6` |
| 3 | `1.02633524` | `1.02634896` | `20.3331%` | `24.698236 ms` | `1.609325e-6` |
| 4 | `0.94277477` | `0.94277735` | `20.1984%` | `22.787664 ms` | `2.324581e-6` |

所有 pair 的 host/event 方向一致。

## 4. Host opportunity 门禁

| 门禁 | 冻结要求 | 实测 | 判定 |
|---|---:|---:|---|
| boundary share median | `>=15%` | `20.333052%` | PASS |
| boundary absolute median | `>=15 ms` | `24.683788 ms` | PASS |
| qualifying runs | `>=4/5` | `5/5` | PASS |
| parity target | `1.107412x` | frozen | — |
| required region speedup | finite and `<=10x` | `1.91213674x` | PASS |

`1.91213674x` 是在真实 median share 下，要让旧 bridge 到达 parity 所需的 **region 内加速要求**，不是
本轮已经测得的性能结果。后续 GC-0/FCR-1 correctness 必须先显著减少 57 launch/约 540 crossing、
materialization 和细粒度 host guard；correctness closure 后才能另行预注册 wrapper-inclusive timing。

## 5. Replay 与 tamper

- artifact replay 从 raw 重算所有 5 个 pair、两组 gates、status 和 summary hash；
- 12 类 fully re-signed raw attack 全部拒绝；
- attack 覆盖 host ratio、boundary category、semantic、launch、fallback、module、guard、stream、run order、
  delete run、performance claim 和 source identity；
- tamper report hash：`965ca72535f618a2639ed4e30669a2c23935533ca39b6c0c1a5ce8ed83ee661f`；
- artifact 6 个 manifest-bound payload 无 `/home/` 路径泄漏。

## 6. 测试与静态检查

- MR7-R + MR6/MR7 artifact 相关：`13 passed`；
- 新增/触及五个 Python 文件 Black clean；
- Mypy `--follow-imports=skip`：5 files clean；
- Pylint：`10.00/10`；
- 全量：`1821 passed, 3 skipped, 6 warnings`，耗时 `687.96 s`；
- 3 个 skip 均为既有环境边界：TVM 已存在时跳过 no-TVM smoke，以及两个冻结 VNN-COMP checkout
  unavailable 测试；
- 首轮未激活 conda hook 时有 3 个旧 PR12 collection error（`tvm` 不可见）；按仓库规定执行
  `conda activate boundflow` 后全量通过，因此不是代码回归。

## 7. Claim 边界

本轮可以说：

- unprofiled ledger 已通过低扰动资格；
- host boundary 是正式、稳定、Amdahl 可达的 compiled-region opportunity；
- GC-0/FCR-1 ABI 与 correctness 预注册已开放。

本轮不能说：

- BoundFlow 已经快 `1.912x`；
- 已实现 graph compiler、persistent arena、minimal-saved-state VJP 或 command graph；
- 已达到旧 bridge/provider parity；
- 已形成 same-solver/query/queue/competitor/ASPLOS performance claim。

## 8. 唯一下一步

只允许先写并冻结 `GC-0/FCR-1 verification graph ABI + correctness` 预注册，首个 vertical slice 必须是
closed multi-op region，而不是更薄的逐算子 wrapper：

```text
typed production state
→ compressed α/β lookup
→ relaxation/sign
→ layout normalization
→ fused Conv/Linear bound propagation
→ bias/reduction/epilogue
→ minimal-saved-state VJP
→ persistent arena output/status
→ coarse atomic commit
```

GC-0 只做 schema、analysis-only legality、lowering ABI、arena identity、replay/tamper 和 correctness；
`timing_open=false`。correctness 正式关闭后，才能另行预注册 GC-1/GC-3 timing。
