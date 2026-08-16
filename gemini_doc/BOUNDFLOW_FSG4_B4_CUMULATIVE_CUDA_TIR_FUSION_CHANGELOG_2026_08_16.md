# FSG4/B4 cumulative CUDA/TIR fusion changelog

## 2026-08-16 — B4-A implementation candidate

- 实现第10次evaluation terminal lower/lA typed producer、one-shot handoff与zero-rerun assembly；
- same-solver仅显式opt-in B4-A，其余B3/KFSB/device/post/queue路径不变；
- content digest与raw float32 export audit移到query后排除计时区；
- 新增5 fresh pair runner/root replay；GPU smoke通过但不形成性能claim；
- 状态=`IMPLEMENTED-B4-A-PENDING-CLEAN-SOURCE-FIVE-FRESH`。

## 2026-08-16 — B4-A preregistration

- 冻结optimizer第10次evaluation同时输出terminal lower/lA、export零CROWN重跑的单变量合同；
- typed lineage绑定state/graph/split/topology、producer op ordinal/name、shape/dtype/device/layout/content；
- 冻结10/9 optimizer、4 forward、3 KFSB、handoff=1、rerun=0及provider/fallback=0计数；
- 固定related pytest文件清单、5 fresh correctness与B3/B4-A `1.03x/0.98x`性能门禁；
- 状态=`PREREGISTERED-B4-A-NOT-IMPLEMENTED`，B4-B/TIR不得混入。

## 2026-08-16 — B4-0 external audit closure

- Round 1外审从raw独立复算AC1—AC7全PASS，无blocker/major；exchange=`closed/approved`；
- 审计方第10类全重签allocation-delta攻击仍被semantic replay拒绝；
- 两项minor转为B4-A硬门禁：shape从correlation parent operator恢复并绑定lineage；exchange固定
  related pytest文件清单；
- 最终状态=`EXTERNALLY-APPROVED-VALIDATED-B4-0-OPPORTUNITY`；只开放B4-A，B4-B不得混入。

## 2026-08-16 — B4-0 internal opportunity closure

- source=`66154e4`生成fresh control/profile正式artifact；semantic discrete/sign exact，lower max diff=
  `4.76837158203125e-07`；
- raw=`270609`events、`35367/35367`kernel closure、14-call/4-forward marker exact；
- CROWN14=`9196`kernels、`32618329 ns`kernel-sum、`57292800 B`累计allocation delta；
- B4-A以消除完整terminal export CROWN call准入；B4-B 14-call覆盖约67.72% B3 core，过5%门槛；
- replay PASS、9/9 outer-resigned tamper rejected；仍`performance_claimed=false`；
- 状态=`INTERNALLY-VALIDATED-B4-0-OPPORTUNITY-PENDING-EXTERNAL-AUDIT`，外审前不启动TIR。

## 2026-08-16 — B4-0 runner candidate

- 实现typed raw profiler schema及control/profile独立worker；
- 以correlation parent为主、temporal marker为显式fallback，保存stream/shape/duration/memory delta；
- 区分CUDA user annotation、phase device total与真实kernel，未归属kernel不再静默丢弃；
- raw以确定性gzip JSONL保存，同时绑定压缩/解压/canonical三层digest与行数；
- 从raw重算exact/root phase、kernel/operator/materialization ledger及Amdahl门禁；
- 修复virtualenv interpreter symlink被`resolve()`展开的问题，并增加worker import preflight；
- control/profile semantic改用冻结B3 typed tolerance，并额外保持discrete/sign exact；
- 自动执行9类outer-resigned raw/semantic/protocol/summary tamper；
- B4 targeted=`15 passed`、B3/B4相关=`54 passed`、full=`1329 passed, 3 skipped`，静态检查通过；
- 状态=`IMPLEMENTED-PENDING-CLEAN-SOURCE-FORMAL-ARTIFACT`，尚无B4性能或opportunity关闭结论。

## 2026-08-16 — v1 preregistration

- 以 externally approved B3为直接基线，B0为累计公平对照；
- 从 B3 raw独立重算 core/query share 与 required-speedup公式；
- 冻结14次 production lower-only CROWN调用的全覆盖目标，而非 optimizer-only局部优化；
- 将 B4拆为 B4-0 attribution、B4-A terminal export fusion、B4-B differentiable lower-only TIR、
  B4-C cumulative coverage与B4-D formal timing；
- 明确 PR-12 plain-CROWN TIR不具备grad/α/β/split capability，禁止静默复用；
- 冻结 semantic/autograd/physical activation/replay/tamper与B3/B0双基线门禁；
- B5 JIT/CUDA Graph、B6 runtime、B7 memory继续关闭；
- 当前状态为`PREREGISTERED-NOT-IMPLEMENTED`，下一唯一动作是B4-0。
