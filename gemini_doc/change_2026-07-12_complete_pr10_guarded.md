# 变更记录：PR-10 以 guarded structured path 完成

## 完成范围

- trace schema、真实结构 workload profile、dense/gradient oracle；
- exact SignSplit operator、structured ReLU coefficient、ephemeral bias reduction；
- deterministic operator dump；
- dense/structured CROWN、α、αβ、BaB correctness；
- 360-row clean GPU guardrail matrix。

## 决策

structured 消除了 persistent dense coefficient，并在 plain CROWN 多数点降低 peak；但 eager
重算使 latency 明显恶化，α/β autograd peak 更高且出现 OOM。因此默认恢复 dense，structured
保留为显式 feature flag，等待 method-aware Planner 与 fused/custom-autograd lowering。

详细数据：`gemini_doc/pr10_dense_structured_comparison_2026_07_12.md`。
