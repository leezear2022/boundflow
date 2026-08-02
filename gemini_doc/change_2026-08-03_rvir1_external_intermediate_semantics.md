# 变更记录：RVIR-1 External Intermediate-Bound Semantics

> 日期：2026-08-03
> 分支：`feat/real-verifier-ir-integration-v1`
> 状态：ResNet initial plain-CROWN correctness gate PASS（CPU）

## 主要改动

- Bound IR 的 `ReluRelaxationAttrs` 新增显式 `intermediate_bound_source` 与
  `lower_slope_policy`；默认仍为 `local_forward/zero`，保持旧路径语义；
- 新增 `external_verifier/adaptive` 路径，两个字段进入 canonical JSON 与 stable hash；
- PR-14 capture 现在拥有逐 ReLU pre-activation lower/upper、外部节点 identity 与 aggregate
  SHA256；
- 新增外部→本地图顺序/count/shape fail-closed binding；
- fixed replay runner 使用 external intermediate bounds 与 adaptive lower slope，不再用本地
  IBP trace 伪装 external whole-query semantics。

## 验证

- focused Bound/Plan/Task/Schedule/compiler/PR-14 回归：`89 passed`；
- 冻结 VNN-COMP 2021 ResNet-2B prop0，官方 αβ-CROWN `e5c7e17` / auto_LiRPA
  `5a098e8`，CPU fresh replay：
  - status：`ok`；
  - external intermediate count：6；
  - intermediate hash：`d51615b04dfb205afd67d2c21680ece4ca92f693157da1e32c7f8202a8e08cf1`；
  - lower max diff：`3.09944e-6`；
  - nonnegative：6/6；sign agreement：`9/9`。

## 边界

- 本机 NVIDIA 驱动仍不可用，本轮是 CPU correctness replay，不形成 CUDA 或性能结论；
- external 请求为 lower-only，而现有 BoundFlow runner 仍同时算 lower/upper，所以性能合同
  仍不合规；
- activation-BaB typed external-call IR 尚属下一提交，不能用本结果改写 0/394 的历史 fused
  coverage。
