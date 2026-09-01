---
status: validated-no-go-r3-3-s-isolated-physics
updated: 2026-08-26T05:30:00+08:00
type: closure
topic: boundflow
slug: r3-3-active-beta-timing-formal-no-go
stage: s01
---

# R3-3 Active-β Isolated Timing 正式 NO-GO 关闭

## Verdict

R3-3 S-anchor isolated timing 以 `VALIDATED-NO-GO-R3-3-S-ISOLATED-PHYSICS` 关闭。
固定 sparse Linear TIR schedule 的 6 fresh/180 paired samples 得到：

- paired speedup geomean=`0.6682752922794841x < 1.05x`；
- bootstrap 95% lower=`0.629156535464161x < 1.00x`；
- worst worker=`0.5990888401816539x < 0.98x`。

候选 wrapper 约慢 `1.50x`，三项预注册 latency gate 全部失败。因此停止当前 fixed
schedule 的 R3-4 adjacent-site 扩展；不得改阈值、丢弃慢 worker 或切换成 kernel-only
headline。R3-3 active-β correctness claim 保留。

## 冻结证据

- source=`947eb613928a821ab225d56ea4fe9d2b587e4948`;
- artifact=`artifacts/r3-structured-owner/r3-3-active-beta-timing-v1`;
- protocol hash=`f584e7073cf022b38badf267c4953163c425ac7bdc162c99f7767f2d999a348a`;
- summary hash=`c8050b943d369e01647200a755939420c760e4a695c8fe67f85a1ae36ded9097`;
- manifest hash=`6dffeefcfba5b48b71766fc32f2a935903b826eb2a8dc3f7410d3055a098f2a7`;
- tamper hash=`8d562c4cdf38894750b7495defb19e3d749ff78f297f1f79bffb01a447aea006`;
- template/schedule/module hash 为 `adddcb6a…b9bf56f` / `b8fe0a7d…2350d57` /
  `7f6ab5cb…f842679`，6 个 worker 一致。

## 正式数字

| run | order | PyTorch ms | TIR ms | speedup |
|---:|:---:|---:|---:|---:|
| 0 | AB | `0.807936` | `1.348608` | `0.599089x` |
| 1 | BA | `0.812960` | `1.345408` | `0.604248x` |
| 2 | AB | `0.967760` | `1.387008` | `0.697732x` |
| 3 | BA | `0.908288` | `1.338976` | `0.678345x` |
| 4 | AB | `0.986576` | `1.369600` | `0.720339x` |
| 5 | BA | `0.990272` | `1.372160` | `0.721688x` |

baseline 是 public-PyTorch CUDA dense α/β reconstruction + lower Linear/bias + autograd VJP；
candidate 是 cache-hit sparse TIR custom forward/backward。两侧 wrapper/output allocation/VJP 均在 CUDA event
内，compile、cache miss、H2D 与 IR 构建均在外。AB/BA 均慢，不是单一 order 偏差。

6/6 untimed parity 通过，maximum diff=`6.92903995513916e-07`。candidate module call=
`1 forward + 1 backward`，fallback/eager=`0/0`，forbidden dense workspace=`0`。

## Memory 解读

- maximum absolute allocated ratio=`1.0371094319806284x <=1.05x`；
- maximum reserved ratio=`1.0x <=1.05x`；
- maximum incremental allocated ratio=`10.9375x`。

预注册的 absolute memory 两项通过，但 prepared input/module 占据大部分常驻内存，使 absolute
ratio 掩盖了小 wrapper 的增量分配。`10.9375x` incremental ratio 证明当前 candidate 在该
scope 不具备 memory 扩 site 理由。这些数字都不是 query/system memory claim。

## Tamper 与回归

- 12/12 fully outer-re-signed tamper rejected；
- targeted=`5 passed`；
- full=`1658 passed,3 skipped,6 warnings in 667.12s`；3 个 skip 为已有环境边界。

artifact summary 保持 `PENDING-TAMPER/performance_claimed=false/r3_4_open=false`；本 closure 在 tamper
与全量回归后授予最终 NO-GO。

## 下一步

只开放另行预注册的只读 microphysics attribution/route decision，拆分约 `1.34 ms` candidate
中的 TVM-FFI/DLPack、forward/backward kernel、autograd wrapper、allocation 和 schedule 占比。必须先冻结
share 和新 Amdahl 门禁，才能决定是否存在新 schedule/ABI 路线；R3-4/same-solver 继续关闭。
