# 变更记录：2026-04-16 Phase 7A PR-11 到 PR-14 文档同步

**日期**: 2026-04-16  
**类型**: docs / status-sync / planning  
**范围**: 高层工作文档、统一参考、下一步计划

---

## 动机

PR-11 到 PR-14 的细节记录、benchmark 摘要和总账条目已经存在，但若从目录导引、LLM 交接、研发脉络或统一参考切入，读者仍容易停留在“PR-10 + layout-only support 刚完成”的旧口径。

这轮同步的目标不是重写 PR-11 到 PR-14 的细节，而是把高层工作文档更新到同一个状态：

- Phase 7A 当前前沿已经推进到 PR-14
- shared CROWN 的 benchmark / hotspot / ReLU pullback 主线已经成形
- 下一步重点从“继续清 `split_pos_neg_dense`”收敛为“压缩 `relu_relax_pullback()` 内部 dense materialization 成本”

## 主要改动

- 更新：`gemini_doc/llm_briefing_boundflow.md`
  - 当前前沿从 PR-10 同步到 PR-14
  - 补入 PR-11 到 PR-14 的主线摘要、当前可复现结论、Phase 7A benchmark 命令
  - 已知限制从“`split_pos_neg_dense` 热点”改成“`relu_relax_pullback()` 内部 dense materialization 成本”
  - 文档导航改指向 `phase7a_pr11_shared_crown_benchmark_summary.md` 与新的下一步计划

- 更新：`gemini_doc/README.md`
  - 把 `phase7a_pr11_shared_crown_benchmark_summary.md` 与新的下一步计划加入关键交付文档
  - “当前前沿 / 下一步”一节同步到 PR-14 之后状态

- 更新：`gemini_doc/project_evolution_overview.md`
  - 现状判断从“Phase 6 是当前重心”更新为“Phase 7A 是当前真实前沿”
  - 下一步自然路线改为围绕 shared CROWN benchmark、ReLU pullback 与 operator-specific sound 优化展开

- 新增：`gemini_doc/next_plan_after_phase7a_pr14.md`
  - 固化 PR-14 之后的下一步计划
  - 重点明确为：压缩 `relu_relax_pullback()` 内部 dense materialization、继续补 observability、锁定 PR-14 之后的测试/bench contract

- 更新：`docs/reference.md`
  - 模块说明补充 `linear_operator.py` 已承载 ReLU pullback 接口
  - 常用命令补入 Phase 7A shared CROWN benchmark
  - 阶段演进与文档导航同步到 PR-14 之后

- 更新：`docs/change_log.md`
  - 追加本次文档同步总账

## 结果

- 从 `gemini_doc/README.md`、`gemini_doc/llm_briefing_boundflow.md`、`gemini_doc/project_evolution_overview.md`、`docs/reference.md` 四个入口进入，不会再误以为当前还停在“PR-10 刚完成”的状态。
- Phase 7A 的阅读路径现在更清楚：
  - 细节与复跑看 `phase7a_pr11_shared_crown_benchmark_summary.md`
  - 现状总览看 `llm_briefing_boundflow.md`
  - 后续路线看 `next_plan_after_phase7a_pr14.md`

## 影响面

- 不改任何 runtime / solver / benchmark 代码行为。
- 不改变已有 PR-11 到 PR-14 的详细变更记录。
- 主要作用是统一当前状态口径，减少后续协作时重复解释“现在做到哪了”和“下一步该做什么”。

## 验证

已执行：

```bash
rg -n "next_plan_after_phase7a_pr14|phase7a_pr11_shared_crown_benchmark_summary|PR-14" \
  gemini_doc docs README.md

git diff --check -- \
  gemini_doc/llm_briefing_boundflow.md \
  gemini_doc/README.md \
  gemini_doc/project_evolution_overview.md \
  gemini_doc/next_plan_after_phase7a_pr14.md \
  docs/reference.md \
  docs/change_log.md \
  gemini_doc/change_2026-04-16_phase7a_pr11_pr14_doc_sync.md
```

结果：

- 关键高层文档已经指向新的 Phase 7A 摘要与 PR-14 之后计划。
- `git diff --check` 无格式错误。
