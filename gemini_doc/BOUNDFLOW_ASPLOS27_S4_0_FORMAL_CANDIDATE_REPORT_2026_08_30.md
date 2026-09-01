---
status: validated-s4-0-mutable-state-admission
date: 2026-08-30
type: formal-closure-report
topic: boundflow
slug: asplos27-s4-0-formal-candidate-report
stage: s04
performance-claimed: false
timing-recorded: false
---

# ASPLOS'27 S4-0 mutable-state admission正式候选报告

> **2026-08-30外审关闭修订**：外审以0 blocker/0 major/2 minor判定approve-with-minor-correction；本文已
> 接受并修正F1/F2，状态升级为`VALIDATED-S4-0-MUTABLE-STATE-ADMISSION`，保证等级为
> `E2-DIRECT-LEGACY`。只开放S4-1A implementation/correctness，不开放timing或performance。

## 1. 结论

S4-0已经从local correctness推进并经外审批准关闭：真实alpha-beta-CROWN provider、5个fresh独立进程、
tensor-free receipt、进程内strong-ref lease、63类独立负向门禁和stdlib replay均闭合。现有10类tamper准确
限定为inner+outer-resigned、derived-semantics-inconsistent攻击；它们不能阻止coherent full resign。
当前状态为：

```text
VALIDATED-S4-0-MUTABLE-STATE-ADMISSION
```

这不是性能结果；它只准入S4-1A compressed buffer ownership的implementation/correctness。

## 2. 冻结源与产物

- formal source revision：`b3afde8`；
- runtime基线：`96b6714`；
- formal worker/replay/negative gate初始提交：`5554aa9`；
- exact-call replay修复：`37ce30b`；tamper standalone/binding：`62c2489`、`b3afde8`；
- artifact：`artifacts/asplos27-s4-admission/resnet2b-prop0-v1`；
- replay：`python3 scripts/replay_asplos27_s4_0_admission_stdlib.py --artifact <artifact>`；
- tamper：`python3 scripts/probe_asplos27_s4_0_admission_tamper.py --artifact <artifact>`。

protocol逐文件绑定runtime、worker、artifact generator、stdlib replay、tamper和targeted test的git blob SHA256；
artifact manifest再绑定全部raw、registry、summary、log、README和tamper report。仓库扫描确认无本机绝对路径泄漏。

## 3. 真实provider执行边界

每个fresh进程使用同一个ResNet2B/property/config和冻结外部仓库commit，在真实
`stage_solve.update_bounds_core` exact-call处截获`pre_result`：

```text
provider pre_result
  → strict owner extraction
  → snapshot/topology/R31 plan binding
  → first live capture
  → tensor-free receipt + semantic validation
  → second live capture
  → strong-ref lease publish
  → receipt serialize
  → lease close
  → private sentinel stops before provider core execution
```

真实owner事实为alpha top-level exact `collections.defaultdict(default_factory=dict)`、beta top-level exact dict，
nested alpha exact dict、beta collection exact list、beta entry为SparseBeta、12个value均为exact Tensor。六alpha是
leaf/requires-grad=false，六beta是leaf/requires-grad=true；admission只记录和复验，不改写provider状态。

## 4. 独立重算结果

| 项目 | 结果 |
|---|---:|
| fresh provider processes | 5 |
| mutable slots / paths | 6 / 12 |
| alpha stored / active / preserved elements | 8496 / 4248 / 4248 |
| beta slots / active slots / active elements | 6 / 1 / 6 |
| live tensors / elements per pass / bytes per pass | 12 / 8502 / 34008 |
| content capture passes | 2 |
| logical D2H copies / bytes | 24 / 68016 |
| candidate kernel / CUDA allocation | 0 / 0 |
| provider compute/update callback | 0 / 0 |
| buffer prepare / mutation | 0 / 0 |
| distinct raw / admission hashes | 5 / 5 |

5个admission hash不同是预期行为，但不能全部归因于run ordinal：exact-call identity hash是唯一直接绑定
run ordinal的字段；`production_plan_hash`、`snapshot_hash`、`oracle_mapping_provenance_hash`和
`plan_binding_projection_hash`的逐进程变化来自provider snapshot全量tensor metadata/history的正常数值变化。
外审控制的fresh artifact确认admitted α/β slot及其content hash跨正式/外审共10个进程逐位一致。

## 5. 负向与篡改

原来同一pytest node里连续执行的snapshot/plan、Tensor subclass/object alias、thread/stream和lease攻击已经拆成
独立parameterized case。专项总计78 passed；排除9个positive和6个generic failure-injection seam后，formal
negative registry为63个独立pytest node，全部明确断言stable detail code和
`VerificationRejectionReason`，高于冻结minimum 56。

tamper probe覆盖exact-call hash、copy count、claim flag、slot order、provider content、provider structure、lease
close、provider execute counter、negative registry和worker ordinal。每个变体都重新签署内层receipt/worker及外层
protocol/summary/manifest，但保留至少一处与冻结派生语义不一致，因此semantic replay为10/10 rejected。

外审自建第11类coherent full resign：同步伪造abcrown commit、5行raw source/source hash/raw hash、protocol、
summary与manifest，E0 self-check接受。这是预期的自签闭包边界：现有replay证明artifact内部一致性，不证明物理
执行真实性；本轮真实性由外审独立核验外部source并亲启5个fresh provider进程建立，故等级为
`E2-DIRECT-LEGACY`。S4-4不得继续使用该legacy例外，必须落地challenge+witness。

## 6. 验证与边界

已通过：targeted admission 78、artifact closure 4（合计82）、formal negative 63、5-fresh artifact generation、
stdlib replay、10/10 derived-semantics-inconsistent tamper、mypy clean、pylint 10.00、无本机路径泄漏；全量为
`1966 passed, 3 skipped, 6 warnings in 728.84s`，3个skip均为既有TVM/VNN-COMP环境边界。

明确未证明：process-global query exclusivity、S4-1A buffer ownership、candidate TIR执行、optimizer mutation、
same-solver替换、timing、performance、complete-query和10x。第一个失败的formal生成尝试只暴露replay把
exact-call identity误按字符串直接hash的问题；修复为合同规定的`{"exact_call_id": ...}`投影后重新从干净提交
生成全部5 fresh，失败产物未进入正式artifact。

## 7. 下一步

外审已经批准且F1/F2完成修正。下一动作只开放S4-1A：把六组α与active β转成候选侧persistent compressed
buffers，并验证owner、layout、pointer、version、lease transfer与失败回滚。S4-1A timing/performance、TIR evaluator、
same-solver、complete-query和10x继续关闭。
