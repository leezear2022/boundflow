# 2026-08-13 — FSG3/B2 Same-Solver Timing 预注册

## 改动

- 新增独立FSG3/B2预注册，冻结B0 original、B1 typed passthrough、B2 whole-call reference replacement；
- 冻结六个全排列block与control/profile交替顺序，共36个fresh独立GPU进程；
- 分离cold total、process-hit query、whole core、GPU event、compile与post-measurement validation；
- 冻结correctness/no-fallback、GPU排他、profile扰动、raw-first replay和同步重签tamper门禁；
- 明确B2变慢不是full-stack NO-GO；FSG3可审计关闭后仍按B3—B7逐层推进。

## 边界

本轮只修改计划和权威状态，不实现runner、不执行性能实验、不形成speedup claim。

## 验证

- 预注册内部顺序、配对数量、指标方向、状态机和已有V4-3 closure一致性人工核对；
- `git diff --check`与DocOps validate/lint在提交前执行。
