# 变更记录：Plan IR v1 state-validity、legacy assembly 与 IR-2 closure audit

> 日期：2026-07-28
> 分支：`feat/compiler-ir-stack-v1`
> 基线：`29542b4`（IR-2B reference builder/selector/artifact）
> 判定：**IR-2 reference closure = VALIDATED-REDUCED**；C2/ASPLOS-ready 仍未完成

## 1. 本轮关闭的缺口

### 1.1 Query-time state validity

`PlanInstance` 新增 canonical `state_validities`，每项显式记录：

- `state_id`；
- `source_value_id`；
- `state_version`；
- `valid`；
- invalid 时必需的 `invalidation_reason`。

`StateAction` 新增 `REUSE`。跨决策 verifier 与 selector 共同保证：

- 只有 state/value/version 完全匹配且 `valid=true` 才能选 reuse；
- stale/invalid state 必须选 recompute/cache/evict 中的合法替代；
- template 中每个 state group 必须恰好作一个决定；
- 伪造“valid 但版本与 Bound IR 不同”会 fail closed；
- validity evidence 进入 PlanInstance canonical JSON/hash/replay，不藏在 runtime/meta。

### 1.2 Legacy migration 原子 assembly

新增 `boundflow/planner/plan_ir_legacy_assembly.py`：

- 多个 `LegacyPlanMigration` 按 source kind/hash 稳定排序；
- 每组 candidate 原子加入同一 `PlanTemplate`，加入后执行完整 template verifier；
- duplicate ID、orphan reference、capability/storage 等任何失败都拒绝整组，不留下半迁移状态；
- `UNSUPPORTED` 对象保留明确原因；
- 输出 accepted/classified-unsupported/rejected 的 canonical report 和稳定 hash；
- legacy selected candidate IDs 保留在报告中，不假装已经成为新的 query-time PlanInstance。

### 1.3 独立进程 artifact replay

新增：

```bash
python scripts/run_plan_ir_v1_reference_artifact.py generate --out-dir <dir>
python scripts/run_plan_ir_v1_reference_artifact.py replay --artifact-dir <dir>
```

两个命令分别在新进程中确定性重建同一 typed Bound IR/PlanTemplate，生成或核对 immutable
selection artifact。测试要求 generate/replay 的 Bound、Template、Instance 三个 hash 完全相同。

该命令是 reference contract smoke，不是性能 artifact。

## 2. 旧 PR-11/12 原始记录审计

新增：

```bash
python scripts/audit_plan_ir_v1_legacy_records.py --root artifacts
```

本工作区实测：

- JSON/JSONL 文件：58；
- 递归 JSON objects：4,911；
- parse failures：0；
- `boundflow.materialization_plan/v1`：0；
- `boundflow.materialization_placement/v1`：0；
- `boundflow.backend_candidate/v1.0`：0。

结论：仓库/工作区没有序列化的 PR-11/12 planner decision 原始记录。现有 tracked PR-12
kernel-foundation 只含 codegen/latency summary，不是上述计划对象。

因此本轮不能宣称“历史每条记录已迁移”。能够审计的是：

1. 六类旧对象的代码级 adapter/partial/unsupported 覆盖；
2. 多 migration 的原子 template assembly；
3. 对任意外部 artifact root 的 schema inventory 工具；
4. 当前 root 的 raw-record absence evidence。

如果以后找回 ignored/外部 PR-11/12 原始 artifact，可用 audit 工具定位记录，再补
Bound-region/transition/cost context 后批量调用 adapter/assembly；不得从总结数字反推伪造记录。

## 3. IR-2 冻结门禁核对

| 门禁 | 结果 | 证据 |
|---|---|---|
| PlanTemplate/PlanInstance + verifier | PASS | typed schema、完整 candidate accounting、cross verifier |
| 同一输入产生相同 plan hash | PASS | builder/selector/hash tests |
| 旧 PR-11/12 decision 映射或 unsupported | PASS（对象族级） | 六类 adapter + atomic assembly；逐记录因 raw artifact 缺失不可执行 |
| 无语义 decision 藏在 meta/Any | PASS | core AST dependency test；PlanBundle meta 明确 unsupported |
| capability/memory/state-validity 交叉校验 | PASS | capability rejection、budget/deadline、exact reuse/stale recompute tests |
| deterministic replay | PASS | canonical instance replay、immutable artifact、fresh-process CLI |

据此，IR-2 的最小 reference contract 可以关闭为 **VALIDATED-REDUCED**。`REDUCED` 的原因不是
代码门禁失败，而是：

- 没有真实 PR-11/12 raw planner artifact 可做逐记录迁移；
- 当前 artifact 是 CPU/plain-CROWN reference smoke；
- 尚未连接 Schedule IR、生产 runtime/backend 或 non-toy verifier E2E。

## 4. 验证结果

- Plan IR/legacy/artifact/audit 专属：`21 passed`；
- 相邻 Bound IR、PR-11/12、storage/env：`97 passed`；
- 全量：`418 passed, 1 skipped, 6 warnings`，48.17 s；
- Black clean；
- Mypy 0 issues；
- Pylint 10.00/10；
- `git diff --check` 通过。

## 5. 下一阶段

下一步正式进入 **IR-3 Schedule IR v1 + reference executor**：

1. typed allocate/free/launch/event/batch/retry/result-restore nodes；
2. PlanInstance region/backend/storage/state → Schedule IR lowering；
3. dependency、lifetime、stream、fallback/retry、result accounting verifier；
4. 单 host/单 device/同步 reference executor；
5. canonical dump/hash/trace/replay；
6. 与 Bound IR interpreter、Plan IR selection artifact 做最小端到端闭环。

不得把现有 `runtime/scheduler.py` 的 `topo_sort()` 循环或
`FusedCrownExecutionStep` 直接改名为 Schedule IR。
