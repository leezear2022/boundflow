# README 主流水线端到端接通计划变更记录

date: 2026-08-26
status: draft-for-user-review
code-changed: false
performance-claimed: false

## 变更

- 以根README的`Frontend/IR→Global Planner→BoundTasks→TVM`为主轴重新组织下一阶段；
- 把当前旧PlanBundle、新typed IR链、solver runtime三条历史路径识别为主要集成断点；
- 将GC0-1降为Bound IR legality子阶段；其异步外审虽已出现approved产物，但按用户要求暂不处理；
- 冻结E2E-A IBP/CIBC与E2E-B CROWN/RVIR两个纵向切片；
- 明确Planner、Prepared Runtime、same-solver、并行/内存/JIT的准入顺序；
- 保留历史`2.45631x`、`0.910001x`、`1.91214x`等数字作为路由输入，不升级performance claim；
- 给出R0—R7门禁、最终`1.20x queue/1.15x complete-query`目标和十步提交建议。

## 边界

本轮只写草案，不修改README源码、compiler/runtime/tests，不启动外审，不形成执行授权。
