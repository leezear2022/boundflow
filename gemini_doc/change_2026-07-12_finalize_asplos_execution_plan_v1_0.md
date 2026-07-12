# 变更记录：定稿 BoundFlow ASPLOS 执行计划 v1.0

## 修改

- 将候选母文档原位升级为执行计划 v1.0，避免维护两份冲突的顶层计划。
- C1 改为具有显式物化语义的 Structured Bound-Operator IR。
- C2 增加 `(G,Q,H,B,R)` 输入、`P=(m,π,f,b,c,r,s)` 决策、目标、约束与 staged heuristic。
- C3 收缩为 multi-spec 与 BaB domain batches；训练降为第二客户端。
- 增加 cache validity matrix 与三层 correctness/soundness 术语。
- PR-10 改为 instrumentation-first，端到端速度门槛移到 PR-12。
- 增加 generic compiler 与 same-solver executor baseline，收缩主 workload。
- 第一次硬 Go/No-Go 提前到 8 月 5 日，8 月 25 日后禁止新增技术功能。
- 同步 `AGENTS.md`、`gemini_doc/README.md`、协作 workflow 与 ASPLOS claims map。

## 验证

- 搜索旧 C1/C2/C3 名称、候选稿状态与旧门禁日期；保留的日期只用于后续里程碑本身。
- `git diff --check` 验证 Markdown 与索引修改无空白错误。

## 边界

本批修改只定稿文档；研究实现从 Gate 0 开始，未跳过门禁启动 PR-10～PR-13。
