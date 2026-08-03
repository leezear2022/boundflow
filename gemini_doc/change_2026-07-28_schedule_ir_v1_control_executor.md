# 变更记录：Schedule IR v1 control actions、reference executor 与 trace artifact

> 日期：2026-07-28
> 分支：`feat/compiler-ir-stack-v1`
> 基线：`9322abd`（IR-3A Schedule schema/lowering/verifier foundation）
> 状态：IR-3B control/executor/trace foundation validated；IR-3 尚未完成

## 1. 补齐的控制动作

`boundflow/ir/schedule.py` 新增：

- `BatchLoopAction` + `QueryBatchSlice`；
- `RecordEventAction` / `WaitEventAction`；
- `StateLoadAction` / `StateStoreAction` / `StateInvalidateAction`；
- `RetryAction` / `FallbackAction`；
- `RequestReplanAction`。

这些不是只序列化字段；统一 verifier 已加入对应跨动作不变量。

### 1.1 Query batching

- lowering 根据 selected domain/spec/sample batch capacity 按原序切 query slices；
- 全部 slice 展平后必须与 `ScheduleModule.query_ids` 完全相等；
- query 丢失、重复或重排均 fail closed；
- `EmitResultAction` 再次核对同一 query 顺序和 Bound IR outputs。

### 1.2 Stream/event

- event 只能 record 一次；
- wait 必须发生在 record 之后；
- 每个 value 记录 producer stream；
- 跨 stream consumer 必须先 wait producer event；
- 未显式 happens-before 的跨 stream use-def fail closed。

### 1.3 State

- Plan `REUSE` 降为 exact-version `StateLoad`；
- `CACHE` 降为 `StateStore`，且 source value 必须已定义；
- `EVICT` 降为带 reason 的 invalidation；
- `RECOMPUTE` 不伪造 cache action；
- Schedule state actions 必须与每个 selected Plan state decision 一一对应。

### 1.4 Retry/fallback/replan

- retry 明确绑定一个 launch、fallback action IDs、最大 attempts 和 `retry_on=("oom",)`；
- `max_attempts == 1 + fallback 数量`；
- fallback backend 必须 legal、同 region、兼容 selected representation，且 ladder 不重复；
- orphan fallback、retry-after-launch、非 OOM retry 均拒绝；
- replan 必须保存同一 Bound IR hash，且只能改变 backend/representation/batch/storage/state。

## 2. Reference executor 与 runtime trace

新增 `boundflow/runtime/schedule_ir_executor.py`：

- 顺序执行全部显式 Schedule actions；
- 动态维护 arena live bytes 和 peak；
- launch 只按声明的 bounded ladder 处理 OOM；
- 未声明 fallback 时不做隐藏 retry；
- trace 记录每个 action、每次 launch attempt、backend candidate、OOM/success、live bytes；
- 最终 trace 再核对 peak、query IDs 和 output IDs；
- canonical trace JSON 与 SHA-256 stable hash；
- `replay_schedule_trace()` 通过重新执行拒绝 noncanonical/tampered trace。

`execute_schedule_with_bound_reference()` 在执行 Schedule ledger 后调用独立 Bound IR interpreter
作为数学语义 oracle。当前 smoke 的 scheduled final lower/upper 与直接 interpreter 完全一致。

这仍是 reference executor：它验证 execution/control contract，但尚未逐 region 调用真实
Task/backend kernel。

## 3. Immutable artifact

新增：

- `boundflow/runtime/schedule_ir_artifact.py`；
- `scripts/run_schedule_ir_v1_reference_artifact.py`；
- `tests/test_schedule_ir_v1_artifact_cli.py`。

命令：

```bash
python scripts/run_schedule_ir_v1_reference_artifact.py generate --out-dir <dir>
python scripts/run_schedule_ir_v1_reference_artifact.py replay --artifact-dir <dir>
```

artifact 固定：

- Bound IR；
- PlanTemplate；
- PlanInstance；
- Schedule IR；
- runtime trace；
- 五个 payload 的逐文件 SHA-256；
- Bound/Template/Instance/Schedule/Trace stable hashes。

fresh-process generate/replay 的 Schedule/Trace hash 必须相同；目标目录非空时拒绝覆盖。

## 4. 验证结果

- Schedule control/executor/artifact 专属：`10 passed`；
- 相邻 Bound/Plan/PR-11/12/storage/env：`107 passed`；
- 全量：`428 passed, 1 skipped, 6 warnings`，50.40 s；
- Black clean；
- Mypy 0 issues；
- Pylint 10.00/10；
- `git diff --check` 通过。

专属负路径包括：

- query slice 丢失；
- use-before-def；
- wrong arena size；
- dropped launch/query；
- trace tamper；
- OOM fallback 与 bounded exhaustion；
- cross-stream 缺少 event wait；
- stale state 已由 Plan verifier 拒绝。

## 5. IR-3 尚未关闭的关键原因

当前 `boundflow/ir/task.py` 仍是旧结构：

- `TaskKind` 只有 `INTERVAL_IBP`；
- `TaskOp.attrs`、`BoundTask.memory_plan`、module bindings 仍有 `Any/dict`；
- Schedule lowering 生成的 `task:<region_id>` 只是稳定引用，尚无对应一等 typed Task IR；
- reference executor 的数学结果由 whole-Bound interpreter 提供，不是逐 Task/backend execution。

因此下一切片必须是 **IR-3C Task IR v1 + Plan region lowering + per-task reference execution**：

1. typed TaskKind/TaskOp attrs；
2. task inputs/outputs、parameter/state dependencies、memory effects；
3. backend capability/artifact/reference implementation ID；
4. Plan region ↔ Task ↔ Schedule launch 双向引用；
5. Task reference executor；
6. 与 whole-Bound interpreter 的逐任务最终结果对齐。

在上述完成前，IR-3 不能关闭，现有旧 `BFTaskModule` 也不能直接改名充当新 Task IR。
