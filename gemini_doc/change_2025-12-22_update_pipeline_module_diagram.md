# 修改记录：更新全流程模块图（2025-12-22）

## 变更说明
- 在模块图中补充 `normalize_primal_graph` 步骤。
- 明确 Planner 输出为 `PlanBundle`，并将 Correctness/JSONL 归并到 Bench 阶段。
- 标注 TVM lowering/compile 为执行器内部流程，避免误解为独立阶段。

## 修改文件
- `gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md`

## 影响范围
- 仅影响文档图示，不涉及代码行为。

## 验证
- 未执行自动化验证；可在 Mermaid 预览中检查渲染结果。
