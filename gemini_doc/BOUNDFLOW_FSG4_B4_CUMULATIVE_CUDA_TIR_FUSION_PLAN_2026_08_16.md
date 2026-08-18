---
status: preregistered-not-implemented
updated: 2026-08-16T00:30:00+08:00
type: plan
topic: boundflow
slug: BOUNDFLOW_FSG4_B4_CUMULATIVE_CUDA_TIR_FUSION
stage: s01
---

# FSG4/B4 cumulative CUDA/TIR 与跨阶段融合计划

## 0. 当前判定

FSG4/B3 已通过 Round 2 外部审计并关闭为
`EXTERNALLY-APPROVED-VALIDATED-REDUCED-B3`。B3 是 B4 的唯一直接基线；B0 original
αβ-CROWN batched executor 继续作为累计公平对照。当前只开放 B4 operator/cross-stage fusion，
B5 JIT/CUDA Graph、B6 runtime batching/streams 和 B7 arena/buffer reuse 继续关闭。

本文只是实现前预注册，不表示 B4 已实现、已正确或已加速。artifact 在正式关闭前继续写
`performance_claimed=false`。

## 1. 目标

在不改变 fixed ResNet2B prop0 的 solver、branching、optimizer 10/9、state ownership、termination
与 post/queue 语义的前提下，逐步把 production lower-bound 数据流从 eager PyTorch 小算子序列改成
BoundFlow 管理的跨阶段 lower-only graph 与可微 CUDA/TIR region，验证以下假设：

1. terminal optimizer 最后一次 CROWN evaluation 可直接导出 lower adjoints，消除重复 backward；
2. optimizer、terminal export 与 KFSB child 中相同的 lower-only ReLU→Affine backward region 可共享
   同一 typed kernel contract；
3. sign selection、α slope、β injection、bias reduction 与相邻 Linear/Conv2d contraction 可在不
   materialize 完整 scaled-A 的情况下融合；
4. 对 α/β 需要梯度的 9 次 update，显式 backward TIR 能保持逐步 mutation parity；对 terminal/KFSB
   无梯度路径则复用相同 forward semantic contract；
5. 累计 B4 相对 B3 产生可传播的 core/query 收益，并持续报告相对 B0 的差距。

## 2. 冻结基线与事实

### 2.1 Source 与 artifact

- B3 externally approved integration source：`f44d656`；
- B3 formal execution source：`36e9069`；
- B3 artifact：`artifacts/fsg4-b3-same-solver-timing/resnet2b-prop0-v1/`；
- B3 exchange：`.docops/exchange/fsg4-b3-formal-timing-20260814/`；
- hardware：RTX 4060 Laptop 8 GiB；
- official solver：αβ-CROWN `e5c7e17`，auto_LiRPA `5a098e8`，solver venv Python 3.11 / torch
  2.11 CUDA；
- direct baseline：B3；cumulative control：B0。

### 2.2 B3 正式结果

- B2/B3 core/query geomean：`1.0716174805930418x / 1.0066228954759742x`；
- B0/B3 query geomean：`0.9100012637918488x`；
- worst B2/B3 core pair：`1.0635877032562384x`；
- B3 无显存收益；
- 36/36 fresh workers、30+6 direct semantic pairs、root replay、10/10 tamper、targeted 114、
  full 1314+3 skipped 均通过。

### 2.3 B3 profile 归因

B3 profile core component geometric share：

| component | wall geomean | core share | B3 query share |
|---|---:|---:|---:|
| optimizer | 112.593 ms | 44.729% | 7.933% |
| typed pre-state | 58.284 ms | 23.154% | 不作为 B4 自动收益 |
| KFSB | 45.922 ms | 18.243% | 与 backward 合并计入下项 |
| atomic commit | 22.476 ms | 8.929% | 不作为 B4 自动收益 |
| terminal backward export | 11.930 ms | 4.739% | 与 optimizer/KFSB 合计 12.010% |

整个 B3 core 占 query `0.17735758999613638`。CROWN backward 调用的固定物理结构为：

- optimizer bound evaluations：10；
- terminal lower/lA export：1；
- KFSB candidate child batches：3；
- 合计：14 次 lower-bound backward；
- forward trace builds：4（optimizer parent 1 + KFSB child 3）。

这些计数来自 B3 raw activation receipts，不从 summary 文案推断。

## 3. Amdahl 与开工公式

令 `s` 为 B3 baseline query 中某候选 region 的 share，`r` 为该 region 的持续加速比，则：

```text
query_speedup(s, r) = 1 / ((1 - s) + s / r)
required_r(s, target) = s / (1/target - (1 - s))
```

B3 回到 B0 parity 需要：

```text
target_B3_to_B0 = 1 / 0.9100012637918488 = 1.0988995727688762x
```

冻结结论：

| candidate region | B3 query share | infinite-speedup upper bound | 回到 B0 parity 所需 r |
|---|---:|---:|---:|
| optimizer only | 0.079331 | 1.086167x | 不可达 |
| optimizer + terminal backward + KFSB | 0.120102 | 1.136495x | 3.989703x |
| whole B3 core | 0.177358 | 1.215595x | 2.030219x |

因此：

- 不允许把 optimizer-only 优化包装成足以完成系统目标；
- B4-0 必须量化 14 次 lower-only CROWN 的 kernel/launch/materialization 覆盖；
- 若 B4 只覆盖其中一部分，必须重算 share 与 required speedup；
- 若 `required_r` 无定义或大于预注册的物理上限，不启动该 TIR candidate；
- 用户报告的 BoundConv `40x` 仍是 `USER-REPORTED`。若未来在这整个 12.010% region 上独立复现
  40x，则 B3 query 的 Amdahl 投影约为 1.133x，而不是直接声称 whole solver 40x。

## 4. B4 与其它层的边界

### 4.1 B4 允许

- lower-only semantic specialization；
- 相邻 ReLU→Linear/Conv2d backward 的 operator fusion；
- α slope、β injection、sign select、bias reduce 与 affine contraction 的单 region lowering；
- terminal optimizer final evaluation→lA export 的 producer/consumer fusion；
- B4 自有 forward/backward TIR 与显式 custom-autograd contract；
- 静态 shape/capability fail-closed dispatch；
- 已编译 exact cache key 的复用，但不把 compile amortization 算作 B4 warm speedup。

### 4.2 继续关闭

- B5：`torch.compile`、Inductor、CUDA Graph capture/replay、conditional JIT、compile amortization；
- B6：跨 query/domain 的 batching、multi-stream、branch pipeline、异步 queue；
- B7：arena、buffer alias/reuse、allocator policy、为制造 OOM 而改变 batch；
- 算法改变：iteration、学习率、branching heuristic、candidate count、timeout、精度、solver flags；
- 用 BoundFlow 自有 solver 替换 same-solver αβ-CROWN host；
- 将 B3 的 1.071617x 与 B4 的增量收益相乘后省略 B0 实测。

## 5. 语义与 autograd 所有权

Production query 的 official post packet 只消费 lower bound，upper 被明确置为 `+inf`；optimizer loss、
terminal export 与 KFSB child decision 同样只消费 lower path。B4 可以 lower-only，但必须保存以下合同：

1. discrete graph/topology/split/history/ordinal exact；
2. 每次 evaluation lower `allclose(atol=rtol=2e-4)` 且 sign exact；
3. 9 次 update 后逐 step α/β parity；
4. terminal lower、六层 lA、三组 72 个 child lower、top-3 candidate 与 final decision parity；
5. official post/queue/accounting/termination exact；
6. 任何 unsupported shape/dtype/layout/grad/state 只能明确拒绝，不能 silent fallback 后计入 B4。

权重、input box 与 external intermediate bounds 为只读；α、β 与上游 coefficient 是 differentiable
inputs。B4 custom backward 必须返回所有影响 optimizer loss 的 gradient，不得只实现 forward TIR 后用
`detach` 绕过。forward/backward kernel、cache key、stream 与 output alias 均进入 Plan/Schedule/manifest。

## 6. 分阶段执行

### B4-0 — read-only kernel 与 materialization attribution

目标：在不改变 B3 默认执行的情况下，对 fresh B3 worker 采集 raw PyTorch/CUDA events，并把事件归属到：

- pre-state import/validation；
- optimizer evaluation、autograd backward、Adam/clamp；
- terminal export；
- KFSB score/topk、三个 child CROWN batch；
- atomic stage/assembly/commit。

输出至少包含：raw event、operator name、kernel name、CPU/CUDA duration、launch count、input shape、
phase parent、stream、allocation delta、14-call ordinal 与 source/code/protocol hash。control/profile 分离，
profile 数字只用于 attribution；正式 latency 只来自无 profiler worker。

准入 B4-A/B：

- raw event 可独立聚合且 phase closure 通过；
- profiler perturbation 被记录，不用于 speedup；
- 14-call lower-only region 的 baseline share、kernel/launch top-N、materialization bytes 与 required-r
  被冻结；
- 若不存在至少一个可覆盖 5% B3 core 或可消除一个完整重复 CROWN call 的 candidate，B4 直接
  `VALIDATED-NO-GO-B4-OPPORTUNITY`。

### B4-A — terminal evaluation/export fusion

在 optimizer 第 10 次、无 update 的 evaluation 中同时生成 terminal lower 与六层 lower adjoints，
`export_rvir_v4_native_backward`只做 typed assembly，不再运行第 11 次 parent CROWN backward。

物理门禁：

- `terminal_export_crown_rerun_count=0`；
- `terminal_lower_adjoint_handoff_count=1`；
- forward trace build 仍为 4；optimizer evaluations 仍为 10；KFSB child batches 仍为 3；
- provider/fallback callback 全 0；
- 5 fresh B3/B4-A correctness pairs全部通过；
- B3/B4-A core geomean至少 1.03x、query pair最差不得低于 0.98x；否则机制可保留但不得累计为
  performance candidate。

### B4-B — differentiable lower-only CUDA/TIR region

只在 B4-0 证明 opportunity 后实现。先冻结一个 production 真实 shape，再扩到六层 exact signature。
region 至少融合：

```text
incoming lower A
  -> sign selection
  -> alpha slope / beta coefficient injection
  -> ReLU intercept + affine bias reduction
  -> Linear matmul or Conv2d transpose-contraction
  -> previous lower A + accumulated bias
```

同时实现 backward TIR，覆盖 incoming A、α 与 β 的必要 gradient。禁止直接复用 PR-12 plain-CROWN
kernel：该 kernel capability 明确为 `requires_grad=false / alpha=false / beta=false / split=false`。

门禁：

- micro semantic：forward、backward gradient 与 eager reference逐项 parity；
- optimizer semantic：10/10 lower、9/9 α/β update、terminal state parity；
- exact activation receipts：eligible region必须全部由 B4 backend执行，fallback=0；
- 编译/load/cold 与 warm 分开；正式 warm timing不含 compile；
- 单一 shape 的 kernel speedup不得外推为 whole-core/query claim。

### B4-C — 14-call cumulative coverage

将 B4-B 合法 kernel扩到：

- optimizer 10 calls（9 grad + 1 terminal no-grad）；
- terminal export 由 B4-A handoff，不重跑；
- KFSB 3 child CROWN calls（no-grad）；
- Linear/Conv2d/residual fanout所有实际 region；unsupported region必须进入显式 coverage ledger。

不得通过 B6 candidate batching 把三个 KFSB call 偷并为一个，也不得通过 B5 CUDA Graph隐藏 launch。

### B4-D — five-fresh correctness 与正式 same-solver timing

先运行 5 fresh counterbalanced B3/B4 pairs；通过后才运行六全排列 B0/B3/B4、36 fresh
control/profile workers。formal artifact必须包含 raw-first/resume、root replay、outer-resigned tamper、
environment gate、semantic parity、activation coverage、profile closure与相对 B3/B0 两组比值。

## 7. B4 分类门槛

所有性能分类均以前置 correctness/environment/measurement/replay/tamper PASS 为条件：

### `VALIDATED-B4`

- B3/B4 core geomean `>=1.20x`；
- B3/B4 query geomean `>=1.03x`；
- worst paired core ratio `>=0.9523809524`；
- B0/B4 query必须实测报告，但不要求本层单独达到最终系统门槛；
- 14-call coverage、custom backward 与全部 semantic gates通过。

### `VALIDATED-REDUCED-B4`

- B3/B4 core geomean `>=1.10x`；
- B3/B4 query geomean `>=1.00x`；
- worst paired core ratio `>=0.9523809524`；
- correctness与机制门禁全部通过，但未达到 full B4门槛。

### `VALIDATED-NO-GO-B4`

- correctness成立但未达到 reduced 门槛，或 opportunity/required-r gate判定物理不可达；
- 后续只能保留机制证据，不得以 tuning 方式继续消耗同一 held-out。

任何 correctness、provider ownership、custom backward、environment、measurement、replay或tamper失败均为
blocker，不得用性能数字覆盖。

## 8. Artifact 与攻击面

artifact至少绑定：

- git source、code blob、TVM/torch/CUDA/GPU identity；
- B3 artifact manifest/hash与外审 closure hash；
- exact B4 feature set、kernel contract、TIR/compiled module hash、cache key；
- raw worker、raw profiler event、activation receipt、semantic tensor digest；
- compile/cold/warm分层、peak allocated/reserved、launch/kernel/materialization ledger；
- B3/B4与B0/B4 paired ratios；
- `performance_claimed=false`直到外审批准。

tamper至少同步重签外层 digest后攻击：phase/ordinal、shape/layout/dtype、lower-only flag、grad flag、α/β
ownership、kernel hash/cache key、activation/fallback count、raw latency、semantic tensor、candidate swap/delete、
B0/B3 baseline identity与summary classification。

## 9. 实现顺序

```text
B4-0 profiler schema + fresh attribution artifact
  -> opportunity/required-r closure
  -> B4-A terminal lower-adjoint handoff
  -> five-fresh B3/B4-A
  -> B4-B one-shape differentiable lower-only TIR
  -> six-layer capability + activation ledger
  -> B4-C 14-call cumulative coverage
  -> five-fresh B3/B4 correctness
  -> B4-D formal B0/B3/B4 timing
  -> replay/tamper/full regression/external audit
```

下一唯一动作是 B4-0。B4-0 closure 前不实现 TIR；B4-A/B/C/D、B5—B7均不得提前混入。

## 10. Validation

- `git diff --check`；
- B4 schema/unit/tamper tests；
- B3 frozen replay与测试持续通过；
- targeted pytest、full pytest、Black、Mypy、Pylint；
- `dol exchange validate`、`dol lint --soft`；
- 正式性能关闭必须交给外部模型从 raw 独立重算。

## 11. 2026-08-16 执行状态

B4-0 schema/runner已实现为`IMPLEMENTED-PENDING-CLEAN-SOURCE-FORMAL-ARTIFACT`。实现包含control/profile
分离、14-call/4-forward marker、真实kernel与CUDA user annotation区分、correlation/temporal归因、
确定性gzip raw、operator/kernel/materialization ledger、root replay所需的多层digest绑定和本机路径拒绝。
worker解释器保留virtualenv symlink，并在创建artifact前独立验证`import boundflow, torch`。
control/profile semantic使用B3冻结`atol=rtol=2e-4`且discrete/sign exact；generate自动执行9类
outer-resigned tamper并将report绑定进最终manifest。

本状态只证明runner实现；正式fresh artifact、raw-derived opportunity、tamper与B4-0 closure尚未完成。
因此上文“不在B4-0关闭前实现TIR”的门禁不变，下一唯一动作是从clean source生成正式artifact。

### B4-0 正式artifact更新

source=`66154e4`已完成fresh formal artifact、root replay与9/9 outer-resigned tamper，内部状态=
`INTERNALLY-VALIDATED-B4-0-OPPORTUNITY-PENDING-EXTERNAL-AUDIT`。CROWN14覆盖14个call、9196个kernel；
B4-A满足消除完整重复terminal export CROWN call，B4-B按冻结share覆盖约67.72% B3 core。外审批准前
仍不实现TIR；批准后下一唯一动作是B4-A terminal lower/lA handoff，B4-B不得与之混跑。

### B4-0 外审关闭更新

Round 1外审AC1—AC7全部PASS，exchange已`closed/approved`，最终状态=
`EXTERNALLY-APPROVED-VALIDATED-B4-0-OPPORTUNITY`。B4-A正式获准入；其production shape必须从
correlation parent CPU operator恢复并绑定lineage，后续exchange必须固定related pytest文件清单。
下一唯一动作是B4-A预注册，B4-B/TIR不得提前混入。

### B4-A预注册更新

B4-A已冻结为`PREREGISTERED-B4-A-NOT-IMPLEMENTED`：第10次optimizer evaluation必须同时产出terminal
lower与六层lA，export只做typed assembly且CROWN rerun=0。lineage绑定terminal state、graph/split、
native/provider topology、producer op ordinal/name、shape/dtype/device/layout/content digest；先过5 fresh
correctness再测B3/B4-A core `>=1.03x`、query worst pair `>=0.98x`。详细合同见
`gemini_doc/BOUNDFLOW_FSG4_B4A_TERMINAL_LOWER_ADJOINT_HANDOFF_PLAN_2026_08_16.md`。下一唯一动作是
实现B4-A typed producer/consumer；B4-B/TIR仍关闭。

### B4-A正式计时关闭更新

source=`46a8493`的v5已按冻结24-process协议完成。correctness/environment/activation/profile、root replay
与14/14 tamper全部通过；core wall geomean=`1.0189949992x < 1.03x`，query worst=
`0.9969470224x >= 0.98x`，故B4-A只保留机制/reduced evidence，不能进入B4 cumulative performance
baseline。当前状态待外审，B4-B/TIR保持关闭。外审批准NO-GO关闭后，B4-B是否启动必须仅依据B4-0已
冻结的67.72% differentiable lower-only opportunity和本计划B4-B门禁另行决定；不得把B4-A的1.9%
收益计入累计候选，也不得修改B4-A阈值重跑。
