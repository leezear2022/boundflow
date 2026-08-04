# BoundFlow External-Seeded Ancestral Refinement v1 计划（NRIR-23）

日期：2026-08-04
分支：`feat/external-seeded-ancestral-refinement-v1`
基线：`main@030ed1f`（PR #33）

## 1. 起因

NRIR-22 已证明：原生父节点 refinement 的最终 intermediate constraints 能够被子节点安全消费，
并在固定七节点、深度二 ResNet 树上显著优于独立重算。但该路径仍从本地 IBP 状态开始，根
lower 为约 `-417/-603`；已有 NRIR-17 使用冻结的 external αβ-CROWN intermediate bounds，
同类 hard clauses 的根 lower 仅约 `-0.45/-0.57`。当前入口禁止 per-child refinement 与
external override 同时出现，因此两条已验证机制尚未连接。

## 2. 目标

建立一条可审计的组合路径：

`external verifier constraint seed -> native root refinement -> native ancestral carry`

其中 external 数据始终保留 external semantics ownership；BoundFlow 只声明对已验 digest、
已绑定本地图与输入的约束进行交集、原生 refinement 和祖先传递，不把 external 生成算法冒充为
本地实现。

## 3. 非目标

- 不声明 external bounds 由 BoundFlow 原生生成；
- 不声明完整 property closure，除非固定树的全部终端节点确实达到 threshold；
- 不声明 CPU/GPU 性能收益；
- 不扩大到多 workload、cuts 或完整 αβ-CROWN BaB；
- 不靠单纯增加 depth/node budget 掩盖 provenance/IR 缺口。

## 4. IR 与运行时契约

### 4.1 External constraint seed IR

新增不可变 typed seed，至少绑定：

- seed ID、provider 和 `semantics_owner=external_verifier`；
- primal graph、input bounds、external ordered bounds digest；
- 绑定到本地 ReLU 名称后的 constraint-content digest；
- source artifact manifest/payload digest；
- source model、property/objective-set identity；
- 明确的消费语义 `sound_constraint_intersection_only`。

所有 digest 均为严格 SHA-256；字段缺失、图/输入不匹配、shape/dtype/device/order/content 不匹配、
约束非有限或区间不可行都必须 fail closed。

### 4.2 Plan/Task/Schedule

- refinement Plan 条件性包含 external seed IR 与稳定 hash；
- seed 与 native parent refinement source 互斥；
- `MATERIALIZE_FORWARD` Task 显式消费 external seed constraints；
- Schedule action trace 的 input hashes 必须绑定 seed IR 和 constraint content；
- 旧的 local、independent 和 ancestral-only payload 在无 seed 时保持字节语义兼容。

### 4.3 Queue lineage

新增 `external_seeded_ancestral_carry_v1`：

- root 只能消费 typed external seed；
- non-root 只能消费已验证父节点的 native refinement execution；
- 每个 queue record 绑定 seed 或 parent 二选一的来源及 plan/semantic/final hashes；
- alpha/beta 仍仅作为 warm state，不得冒充 sound intermediate constraints。

## 5. 固定实验

Workload：VNN-COMP 2021 CIFAR10 ResNet2B property 0，CPU。
Hard clauses：`0, 2, 4`。
共同预算：七节点、深度二、objective branch、adaptive alpha、25 optimizer steps；refinement
预算在 artifact 中冻结。

三组同源对照：

1. `external_baseline`：冻结 external bounds 直接进入现有优化队列；
2. `external_seeded_root_global`：external seed 只生成一次 root native refinement，结果复用于树；
3. `external_seeded_ancestral`：root 用 external seed，子节点消费父节点 native constraints。

逐 clause 报告 root lower、四个 depth-two leaf lowers、worst/best leaf、proof deficit、trace/hash、
seed/parent lineage；时间仅作诊断。

## 6. Acceptance criteria

- **AC1 typed seed**：seed IR 与本地 runtime payload 一一绑定，篡改任一 provenance/content 字段均
  被拒绝；external ownership 保留。
- **AC2 explicit lowering**：Plan/Task/Schedule/action trace 都显式包含 seed dependency，且旧无 seed
  路径稳定。
- **AC3 queue lineage**：root seed 与 child parent-execution 来源严格互斥、逐节点可追溯，父子 hash
  篡改被拒绝。
- **AC4 soundness guards**：raw external constraints 先与本地 forward 区间求可行交集，进入 Plan
  的 effective seed 必须是本地区间子集；每次 native refinement 与祖先传递均保持 lower 非降、
  upper 非升。
- **AC5 fixed evidence**：三 hard clauses、三模式按同一预算生成 replay-grade artifact；replay 重算
  semantic evidence，而非只验 manifest digest。
- **AC6 regression**：focused tests、全量 pytest、Black、mypy、pylint、`git diff --check`、DocOps
  validate/lint 全部通过。

## 7. 判定门禁

- 若 ancestral 的三条 worst-leaf lower 均不弱于 root-global，且至少一条严格改善：关闭为
  `VALIDATED-REDUCED`；
- 若所有 terminal leaves 达到 threshold，才允许对对应固定 clause 声明 bounded-tree closure，
  仍不外推完整 verifier；
- 若 external-seeded ancestral 弱于 root-global，关闭为 `VALIDATED-NO-GO` 并保留根因证据；
- 无论数值结果如何，都不得升级 GPU、性能、多 workload 或 ASPLOS-ready claim。

## 8. 预计提交边界

一个逻辑提交包含 typed seed、queue integration、负向测试、固定 artifact、权威文档和 DocOps 记录；
不修改 third-party 子模块。
