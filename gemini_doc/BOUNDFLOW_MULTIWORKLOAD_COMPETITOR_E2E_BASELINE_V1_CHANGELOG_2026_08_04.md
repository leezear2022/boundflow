---
status: completed
updated: 2026-08-04T05:05:00Z
type: changelog
topic: boundflow
slug: multiworkload-competitor-e2e-baseline-v1
stage: s01
---

# Multiworkload Competitor E2E Baseline v1 Changelog

## Summary

- NRIR-18 已完成。BoundFlow 首次从三份真实 VNNLIB/ONNX 直接编译 Query/Plan/Task/Schedule
  并运行 bounded complete query；固定 αβ-CROWN 在独立 fresh process 中运行同一 model/property。

## Changes

- 新增 fail-closed VNNLIB box + single-linear-unsafe-disjunct frontend 与 Query IR；三份真实
  property 的 lower/upper/C/rhs 与固定 αβ-CROWN parser 逐字段一致。
- 新增 multiworkload Plan/Task/Schedule IR：3 workload、21 tasks、6 个 fresh-process execution
  action；model/property/CSV/query/compiler/policy/device/timeout 均进入 stable hash。
- 修复 flatten/reshape-first `BoxPerturbation` 没有同步 shape-transform bounds/shape trace 的问题；
  枚举角点 soundness regression 覆盖。
- 本地盘点确认 VNN-COMP 2021 有 10 类 CSV；首批选择 MLP、residual CNN、sequential CNN。
- 新增双后端 generate/replay runner，保存六份完整 stdout log、结构化 result、process return code、
  E2E boundary 与 source-to-IR replay。
- 将上述外部执行原始日志标记为 Git 非文本 diff 输入，避免清理尾随空白时破坏 manifest 绑定的
  原始字节；JSON evidence/manifest 仍保持可审计文本 diff。
- `nvidia-smi` 仍无法连接 driver，因此 GPU 矩阵保留为同一协议的待执行层。

## Validation

- native/αβ-CROWN 状态与 fresh-process E2E：MNISTFC `unknown/verified`、
  `38.644/4.312 s`；ResNet2B `unknown/unknown`、`66.910/64.198 s`；OVAL21
  `unknown/verified`、`31.498/4.527 s`。这些单次、异算法 CPU 数字仅为诊断，不计算 speedup。
- BoundFlow：MNISTFC 完成 9 clauses、3 unresolved；ResNet 在 query deadline 后完成 2 clauses、
  7 pending，root lower=`-543.717/-789.331`；OVAL21 完成 9 clauses、仅 clause 8 unresolved。
- artifact fresh replay evidence hash=
  `473b287bb88e4c52426b405aeb4164aa72a98d7b1bbd74c00471fe1d1451deb0`；Plan/Task/Schedule
  hash=`1ebea324fe28d04d372dc5cf00e094029f940fecebf6060fdb82512caf68897c` /
  `607722045fa06b372fdeda1feab7eaed95222f433c76aff0f8f68809da04a394` /
  `96b04bbd448e065337d67a86f519bca35ff2cf3eb2c59898c8ac64b1720f5b33`。
- focused Query/IR/shape/artifact/tamper `16 passed`；全量 `723 passed, 37 skipped`；新增/相关
  文件 Black、targeted Mypy、Pylint 10.00/10、fresh replay 与 diff check 通过。

## Decisions

- 以多 workload ingest/control/coverage `VALIDATED-REDUCED` 关闭；ASPLOS-ready 仍为 NO。
- CPU competitor 结果只作诊断，禁止报告 speedup。
- ResNet local root 比 external-semantics 历史结果差三个数量级，且 MNISTFC/OVAL21 仍留有
  3/1 个 unresolved clauses；下一主线是 native intermediate-bound refinement，而不是先做
  CUDA timing 或继续堆固定树深。

## Follow-Ups

- 建立 per-ReLU intermediate-bound refinement Plan/Task/Schedule，先在三 workload 上报告
  root/closed-clause 增益与 refinement 成本，再决定 selective policy 和 GPU 执行。

## Links

- plan: `gemini_doc/BOUNDFLOW_MULTIWORKLOAD_COMPETITOR_E2E_BASELINE_V1_PLAN_2026_08_04.md`
- roadmap: `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`
