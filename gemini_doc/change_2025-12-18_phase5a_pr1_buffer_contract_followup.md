# 变更记录：Phase 5A PR#1 追补（buffer 级 TaskGraph/TaskIO contract）

## 背景

在 PR#1 骨架落地后，收到 review 指出一个关键风险：如果 TaskGraph edge 与 scheduler/env 仍以 value 名为主，Phase 5B（buffer reuse/alias）与 Phase 5E（lowering/后端选择）会被迫返工。

## 本次追补

- TaskGraph edge 升级为 **buffer 级依赖**：
  - `TaskBufferDep(src_value, src_buffer_id, dst_value, dst_buffer_id)`
  - validate 强制对齐 `StoragePlan.value_to_buffer`
- Scheduler/env 升级为 **buffer_id -> IntervalState**，并要求 `input_spec.value_name` / `output_value` 可映射到 buffer。
- 测试补齐分叉+合并（branch+merge）用例，验证 topo 调度在多前驱场景正确。

## 目的

提前把 Edge/Env/Contract 钉在 buffer 抽象上，确保后续 PR#2/PR#3（partition、reuse、lowering）可以直接复用这些接口，不返工。

