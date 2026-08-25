---
status: validated-no-go
updated: 2026-08-26T23:35:00+08:00
type: closure
topic: boundflow
slug: mr6-guard-attribution-formal-no-go
stage: s01
---

# MR6 Hot-Path Guard Attribution 正式 NO-GO Closure

## 1. Verdict

MR6以`VALIDATED-NO-GO-MR6-GUARD-DOMINANCE`关闭。把MR5 bridge每个outer的同步value guard从
`360`减到`60`，只带来full/diagnostic host geomean=`1.0331256409896374x`，低于冻结门槛
`1.10x`；diagnostic相对provider仍只有`0.9030066500186469x`。因此安全device-status aggregation
MR6-B不开放。

这不是说guard没有成本；它只证明guard不是当前约10%—15%剩余差距的dominant原因。三triplet中
full/diagnostic=`1.01909/1.07299/1.00844x`，存在可测收益但不稳定且不足以恢复parity。

## 2. Frozen evidence

- worker source=`fb3c245fc8de1be08471d91b97b026ded9ce204b`；
- formal gate commit=`273e0a7`；
- artifact=`artifacts/measurement-recovery/mr6-hot-path-guard-attribution-v1/`；
- summary hash=`6af1382481b9d50860bb245a60ac4cb7e8bd0b864c94d6f78c019004aefeabdb`；
- manifest/raw/tamper SHA256=`ad0e469c…1fa`/`eb68846a…3d0`/`38b3367c…cf2`；
- 3 triplet/9 fresh Latin顺序=`PFD/FDP/DPF`；
- guard receipt provider/full/diagnostic=`0/360/60`；
- 12/12 fully re-signed attacks rejected。
- focused=`12 passed`；closing full regression=`1808 passed, 3 skipped, 6 warnings`，耗时
  `683.09s`；Black/mypy/Pylint=`10.00/10`与diff check通过。

## 3. Formal result

| Metric | Gate | Result | Pass |
|---|---:|---:|---|
| full/diagnostic geomean | `>=1.10x` | `1.033126x` | no |
| full/diagnostic bootstrap lower | disclosure | `1.008444x` | — |
| full/diagnostic worst | disclosure | `1.008444x` | — |
| provider/diagnostic geomean | `>=0.98x` | `0.903007x` | no |
| provider/diagnostic worst | `>=0.95x` | `0.852613x` | no |
| provider/full geomean | reproduction | `0.874053x` | no |
| host/event direction | `9/9` | `9/9` | yes |
| semantic/module/guard counts | exact | exact | yes |

三方outer/final alpha/module state配对allclose、sign exact，global max diff=
`4.708766937255859e-06`。diagnostic保持相同TIR、30/27 launch、cache/module/stream和两项output-finite
guard，只屏蔽270次输入finite/range与30次handoff content同步。

## 4. 物理解释

MR5 current outer还有下列未被MR6消除的成本：

- `30`次forward + `27`次backward=`57`个独立TVM launch；
- 每个forward `7 source + 2 output`，每个backward `8 source + 2 output`，合计
  `30×9 + 27×10 = 540`个DLPack view/pointer round-trip；
- 每launch进入`use_torch_stream`、读取raw stream并核对device/stream/determinism；
- 每site的incoming/result layout `permute/transpose + contiguous`；
- 至少60个forward result tensor、54个backward gradient tensor与30个incoming-bias zero tensor；
- 仍保留60次output-finite同步。

这些是code-count，不是time share。MR6不能据此挑一个优化；下一阶段必须用互斥NVTX/CUPTI与
unprofiled control给它们计时。

## 5. Route closure

- MR6-B safe guard fusion：关闭；
- MR5 correctness：保留；MR5 current runtime timing NO-GO：保留；
- complete-query、queue、B0/B3 parity与ASPLOS-ready：继续关闭；
- 下一唯一动作：MR7 launch/materialization attribution，不改TIR数学和default runtime；
- 从MR6 geomean反推，diagnostic要达到provider parity还需`1.107412x`，要达到provider上
  `1.15x`研究目标需`1.273523x`。MR7必须使用这两个scope目标，不得只报告局部kernel ratio。
