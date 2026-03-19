# 变更记录：`boundflow_full_pipeline_from_claims_to_ae.md` 移除 Mermaid，改为纯文本流水线图

## 动机

部分 Markdown 渲染器（IDE/笔记软件/静态站点）默认不支持 Mermaid，导致 ` ```mermaid ` 块无法正常渲染或直接显示为原始代码，影响阅读体验。

## 本次改动

- 更新：`gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md`
  - 将 Phase 5/Phase 6 的 Mermaid 流水线图替换为 ` ```text ` 的纯文本图；
  - 保持信息密度与节点命名一致（两条工件链入口、schema_version、runner 位置不变）。

## 验证

- 文档变更（无额外运行时验证）。
