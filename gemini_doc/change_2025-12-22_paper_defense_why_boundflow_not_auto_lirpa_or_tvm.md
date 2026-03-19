# 变更记录：补齐论文辩护（为何不端到端用 auto_LiRPA / 为何不直接用 TVM）

## 动机

论文里需要把两个容易被 reviewer 质疑的问题讲清楚：

1. 为什么不直接用 auto_LiRPA（而是写了 BoundFlow）？
2. 既然用编译优化，为什么不直接“只用 TVM”（而要做 BoundFlow 这一层）？

本次目标是把回答从“口号式”改成“层次准确、可反驳、可指向仓库证据”的论文表述。

## 主要改动

- 更新：`gemini_doc/why_boundflow_not_auto_lirpa_or_tvm.md`
  - 修正/弱化可能被抓漏洞的表述（例如避免“TVM 只能编译前向推理”这类过强说法），改为更精确的分层论证：TVM=张量编译/codegen，BoundFlow=验证语义与规划层。
  - 强化“为什么不用 auto_LiRPA 端到端”的系统论证：缺少稳定 IR、显式 TaskGraph/StoragePlan、跨运行复用粒度与可消融 knobs 等。
  - 新增“仓库证据索引”小节，方便论文/AE 直接引用代码/测试/变更记录作为证据链。

- 更新：`gemini_doc/README.md`
  - 将上述辩护文档加入“关键交付文档”索引，便于交接与检索。

## 验证

- 文档变更（无额外运行时验证）。

