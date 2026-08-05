# 2026-08-05 NRIR47 Single-Pass Target Admission Receipt 预注册

- 从 `main@ca0bcf3` 创建 `feat/single-pass-target-admission-receipt-v1`；
- 唯一变量是把每 child compile/validate 的 exact target selection 从两次收敛为一次 typed
  admission receipt；
- 60 个 node-specific target ledger 保持动态互异，禁止跨节点共享；
- production admission 不重选，显式 full replay 必须从 exact source 重选并逐项比较；
- Phase A compiler ratio `<=0.85`、clauses 2/3 queue ratio `<=0.97`；只有全过才运行 Phase B，
  whole trace/measured ratio 均须 `<=0.98`，且改善大于 pooled MAD；
- 当前没有代码、artifact 或性能 claim，`performance_claimed=false` 与 ASPLOS-ready=NO 不变。
