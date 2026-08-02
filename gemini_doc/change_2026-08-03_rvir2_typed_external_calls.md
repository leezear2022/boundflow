# 变更记录：RVIR-2 Typed External Verifier Calls

> 日期：2026-08-03
> 分支：`feat/real-verifier-ir-integration-v1`
> 状态：IR closure 与真实 CPU execution closure PASS

## 主要改动

- Bound IR schema 升至 `boundflow.bound_ir/v1.1`，新增
  `EXTERNAL_VERIFIER_CALL` 与强类型 `ExternalVerifierCallAttrs`；α/β/split 版本缺一即
  fail closed；
- Plan IR 新增 `EXTERNAL_VERIFIER` region 和 `EXTERNAL_ABCROWN` backend；
- Task IR 新增 external verifier task/dependency，Schedule IR 仍要求一次 launch、一次 emit，
  backend implementation 固定为 `external_abcrown_exact_call/v1`；
- 新增真实 query 的 Bound→PlanTemplate→PlanInstance→Task→Schedule 编译入口，五层 stable
  hash 均可重算；执行器只允许调用一次原 αβ-CROWN provider method，不实现或替换其算法；
- profiler 的每次原始 `compute_bounds` 现在可经 typed IR 调度，并记录 result hash、完成状态
  与五层 IR hash；嵌套调用按进入顺序预留记录槽并保留 parent lineage；
- adapter v2 显式记录 lower/upper requested outputs，避免继续使用含糊的 `bounds` 标记。

## 真实执行验证

冻结官方 αβ-CROWN `e5c7e17` 的 `simple_mlp.onnx + robustness_mlp.vnnlib`，CPU、BaB、
timeout 30 秒，先执行无 observer baseline，再执行 typed-IR observer：

- solver status：`unknown == unknown`；
- visited domains：`380 == 380`；
- final lower：`tensor(-0.18902308) == tensor(-0.18902308)`；
- query / typed dispatch / completed：`377 / 377 / 377`；
- activation-BaB：343 个，effective method 全部为 αβ-CROWN；
- parent lineage：347 个非根调用均指向先出现的父 query；
- requested outputs：377 个均显式 lower-only；
- exact external provider 保持 semantics owner；`performance_claimed=false`。

## 边界

- typed 编译/hash 是审计路径，当前明显增加开销，不形成任何 speedup claim；
- 历史 PR-14A adapter v1 的 394 个 activation 调用没有记录 parent lineage，也没有拆分
  lower/upper requested flags；后续 artifact 会保留并明确标注这两个 legacy limitation，不能
  伪称历史 identity 完整；
- 本机 NVIDIA 驱动不可通信，本结果仅为 CPU correctness/integration evidence。

## 验证

- typed external-call、nested lineage、profiler 与核心 IR focused tests：PASS；
- 真实在线 observer on/off 对照：PASS；
- 全量回归与自包含 artifact 由后续 RVIR-4 提交冻结。
