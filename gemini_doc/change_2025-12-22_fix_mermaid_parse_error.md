# 修改记录：修复 Mermaid 解析错误（2025-12-22）

## 变更说明
- 将 Mermaid 节点标签中的 `plan()` 用双引号包裹，避免解析器将括号误识别为形状语法，从而修复报错。

## 修改文件
- `gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md`

## 影响范围
- 仅影响文档中的流程图渲染，不影响代码与流程逻辑。

## 验证
- 未执行自动化验证；可在 IDE/Mermaid 预览中重新渲染该图。
