# BoundFlow：与大模型协作的工程工作流（模板）

本仓库后续的迭代基本采用“**回合式**（PR-by-PR）”的人机协作流程：你提供目标与约束，大模型负责把目标拆成可执行 PR、落地实现、用测试验证、写变更记录，并在每个回合结束时给出下一回合计划。

下面这份文档把我们已经跑通的协作流程固定成模板，便于你把同样的方式交给其它大模型复用。

---

## 快速版：对话工作流摘要

一句话流程：**输入计划 → 产出计划/DoD → 实现与测试 → 修正 → 总结 → 下一步计划**。

细分为 6 步：

1) 输入目标与约束（阶段/PR、DoD、环境限制）  
2) 大模型产出可执行计划与 DoD（含接口钉子）  
3) 实现最小闭环（先能跑、先对齐）  
4) 运行测试并修正（失败则回到 2/3）  
5) 总结本轮结果（已验证/未验证/风险）  
6) 给出下一步 PR 计划与优先级  

## 0. 固定约束（每回合都必须遵守）

- conda 环境：`boundflow`（验证用 `conda run -n boundflow ...`）。
- 每次改动必须：
  - 新增/更新一份变更记录到 `gemini_doc/`（说明动机、改动、验证方法、踩坑）。
  - 同步更新 `docs/change_log.md`（时间线总账）。
- 尽量保持 PR 小步可回滚：一个 PR 对应一个“最小可验证闭环”。
- 运行时 contract：scheduler 的 env 只认 **physical buffer id**（Phase 5B hardening 的核心前提）。

---

## 1. 回合式协作主流程（PR 循环）

把每个 PR 当成一个完整回合，严格走以下 8 步：

### Step 1：你输入目标（含边界）

你提供：

- 当前要做的阶段/PR（例如 “PR#9：TVMTaskExecutor 对齐 Python”）。
- 明确的 DoD 偏好（例如“先对齐测试、再做性能优化”）。
- 环境/依赖约束（是否允许网络、是否允许改 3rdparty、是否必须写 bench 输出字段等）。

### Step 2：大模型先写“计划 + DoD”

输出内容应包括：

- 拆分后的子任务（能按顺序执行）。
- 关键接口/数据结构的钉子（避免后续返工）。
- DoD（Definition of Done）：哪些测试/bench 指标达到就算完成。

> 经验：如果 DoD 不明确，后续“能跑但不可信/不可复现”的概率会大幅上升。

### Step 3：实现最小闭环（先能跑、先能对齐）

原则：

- 优先实现 reference path（语义正确、易调试），再迭代优化 path（fusion、reuse、VM overhead 等）。
- 对 TVM 相关改动优先保证：IRModule 可 build → 可执行 → 与 Python allclose。

### Step 4：补测试（先对齐、再扩覆盖）

每个 PR 至少有一个“能验收”的测试：

- **正确性对齐**：`PythonTaskExecutor` vs `TVMTaskExecutor` allclose。
- **contract 检查**：verifier fail-fast、env physical-only。
- **可复现性**：determinism（同 program+config 两次产物一致）。

### Step 5：执行验证并修正

固定命令套路：

- 单测优先：`conda run -n boundflow python -m pytest -q tests/<this_pr_test>.py`
- 全量回归：`conda run -n boundflow python -m pytest -q`

如果失败：

- 先定位是 **contract/pipeline** 还是 **数值语义**：
  - contract 失败（比如 buffer id、TaskGraph edge）优先修 contract。
  - TVM pipeline 失败（比如 `alloc_tensor`/`global_symbol`）优先修 pipeline。
  - 数值不一致再看算子语义或 dtype/shape。

### Step 6：写变更记录（gemini_doc）+ 更新总账（change_log）

每个 PR 的变更记录建议固定结构：

- 动机
- 本次改动（按文件/模块列）
- 验证命令
- 已知坑/注意事项（尤其 TVM pipeline、cache key、env 变量）
- 下一步计划（可选）

### Step 7：提交并推送（形成可回滚节点）

- `git commit`（conventional commit message）
- `git push origin main`

### Step 8：回合结束：输出“总结 + 下一步 PR 计划”

总结应包含：

- 这次 PR 实现了什么 DoD
- 暴露了哪些新信号（compile_stats/ir_stats/call_tir 等）
- 下一步 PR 优先级（通常是：先可观测 → 再减 overhead → 再 baseline 对照）

---

## 2. Phase 5D 的特化工作流（TVM/Relax 方向）

在 TVM/Relax 主线下，每个 PR 的典型顺序是：

- 先有 **可观测性**（compile per-pass timing、IR dump、call_tir 统计、VM overhead micro-bench）。
- 再减少 **调用次数与往返**（RELAX_OPS 整 task、fusion pipeline、减少 call_tir）。
- 再减少 **VM overhead**（PackedFunc/VM cache、save_function closure micro-bench）。
- 再做 **baseline 对照**（例如 `relax.transform.StaticPlanBlockMemory()` vs 我们的 StoragePlan reuse）。

---

## 3. 给其它大模型的“输入模板”（你可以直接复制粘贴）

你可以用下面模板开一个新回合：

1) 目标：PR#X 做什么（1 句话）  
2) DoD：必须通过哪些 tests/bench（列命令）  
3) 约束：不能改哪些目录/必须写哪些文档/是否允许网络  
4) 输出：希望你在最后给的总结点（例如“给出下一步 PR 拆分与优先级”）  
