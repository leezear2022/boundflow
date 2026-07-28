# 变更记录：IR—Planner—Schedule—Runtime 架构重置

> 日期：2026-07-20
> 类型：文档/claim/执行路线纠偏；无 runtime 或 backend 代码变化

## 背景

PR-14 已以 `VALIDATED-NO-GO` 关闭真实 verifier replay 门禁。随后复审发现，原下一步如果只做
C1/C2 story freeze，会掩盖更基础的代码事实：Bound IR 仍是占位、Plan IR 是多个局部对象的
集合、Schedule IR 尚不存在。

两份外部设计材料提出 verification-aware runtime、offline/runtime planner 和 cached
specialization。其方向有价值，但不能直接据此扩 runtime：PR-13 已显示 ordinary batching 是主要
收益来源，PR-12J 已显示当前编译摊销困难，状态复用还必须受已有 validity contract 约束。

## 修改内容

1. 新增 `boundflow_ir_planner_schedule_runtime_contract_v1_2026_07_20.md`：
   - 定义 Bound/Plan/Task/Schedule/Query/Runtime 的所有权；
   - 审计现有对象与目标 IR 的差距；
   - 定义 PlanTemplate/PlanInstance 和 offline/runtime planner 边界；
   - 对 JIT、batching、状态缓存设置证据与合法性门禁；
   - 冻结 IR-0—IR-6 实施顺序和完成定义。
2. 更新 `asplos_execution_memo_v1_0.md`：追加 IR-first 路线修订，并明确取代纯 story-freeze
   作为下一工程主线。
3. 更新 `asplos_claims_map.md`：
   - C1 降为 runtime mechanism validated / first-class IR pending；
   - C2 限定为局部机制 validated-reduced / unified Plan/Schedule IR pending；
   - 新增 Schedule IR unimplemented；
   - 保留历史实验数字，不让历史证据自动升级新 claim。
4. 给早期 `boundflow_architecture_review_and_extension.md` 添加历史警告，停止引用其星级评分
   和投稿建议证明当前完成度。
5. 给总体计划、PR-13 后状态和 PR-14 外部审计交接增加路线修订，保留其历史数字但阻止旧
   story-freeze 继续充当当前入口。
6. 根据独立审计补齐 `pr14b_initial_crown_fixed_replay_2026_07_19.md` §6 与
   `pr14_execution_plan.md` §7 的历史/被取代标记，关闭直接打开旧文档时的路线歧义。
7. 更新 `gemini_doc/README.md` 与总变更日志索引。

## 新的下一主线

完成本文档修订后，新分支应为：

```text
feat/compiler-ir-stack-v1
```

执行顺序：

```text
Bound IR v1
  -> Plan IR v1
  -> Task IR + Schedule IR
  -> reference/backend runtime migration
  -> adaptive PlanInstance evaluation
```

## 验证

本次只有 Markdown 修改，验证要求为：

- 所有新增/修改文件均可被 git 跟踪；
- 关键文档引用目标存在；
- Claims Map、执行备忘录和 README 对下一路线表述一致；
- 搜索确认旧 `docs/asplos-c1-c2-story-freeze` 只以历史/被取代语义出现；
- 不运行代码测试，不宣称实现完成或性能变化。
