---
status: blocked-one-environment-gate
updated: 2026-08-05T18:22:00Z
type: changelog
topic: boundflow
slug: nrir49-g0-gpu-opportunity-admission-v1
stage: s01
---

# BoundFlow NRIR49 G0 GPU Opportunity Admission v1 Changelog

## Summary

- 启动 NRIR49 G0，新增 fail-closed admission runner/test/artifact；
- 建成独立 αβ-CROWN 官方锁定环境，并冻结双方整题 `verified` 的公开 solveability 样本；
- 根因定位为 ASUS firmware 禁用 dGPU，enable 已 queued，等待重启；
- G1 仍未准入，未修改 TIR/kernel/math，未新增性能 claim。

## Changes

- 新增 `scripts/run_nrir49_g0_admission.py`：采集 GPU/driver/firmware、BoundFlow/TVM/FFI、竞品
  环境、source-level BoundConv oracle、frontend/solveability、memory/Amdahl 前置状态；
- 新增 manifest digest、semantic replay、derived gate 与 tamper rejection；
- 新增 `tests/test_nrir49_g0_admission.py` 的 Amdahl、fail-closed、symlink interpreter、candidate
  verdict 和 artifact tamper 测试；
- `run_multiworkload_competitor_e2e_artifact.py` 增加显式 `--input-shape`，仅用于把 symbolic batch
  冻结为 batch-one；无 override 时维持旧 fail-closed 行为；
- clone `alpha-beta-CROWN@e5c7e17` 与 `auto_LiRPA@5a098e8`，按官方 `uv.lock` 创建独立 `.venv`；
- sparse clone `vnncomp2021@90419aa` 的 MNISTFC，冻结 `mnistfc:2`；
- 生成并 replay `ga403uv-pre-reboot-20260806-v7`，只剩 GPU infrastructure blocker。

## Validation

- targeted tests：`13 passed`；
- 全量：`1006 passed, 37 skipped`；mypy clean；Pylint `10.00/10`；
- artifact replay PASS；
- competitor import smoke：Python `3.11.15`、Torch `2.11.0+cu130`、auto_LiRPA/abcrown `0.7.2`；
- solveability：BoundFlow=`verified`、αβ-CROWN=`verified`；
- `rg '/home/lee|/tmp/'` 对正式 v7 artifact 无命中；
- GPU benchmark 未运行，`performance_claimed=false`。

## Decisions

- 将 GPU blocker 精确收敛为 `blocked_reboot_required`，不再泛称 driver/透传未知；
- independent competitor env 是公平对照的正式组成，禁止复用不兼容的 BoundFlow Torch env；
- `mnistfc:2` 只负责 solveability admission，不作为性能调参样本；
- 用户 `40x` 源码仍缺失，维持 `NOT-AUDITABLE-SOURCE-MISSING`；
- 重启前不做 G1，更不做 G2/G3/TIR。

## Follow-Ups

1. 用户允许并执行一次正常重启；
2. 重启后用同 runner 生成 post-reboot artifact，关闭六项 CUDA/同 GPU identity smoke；
3. 若 GPU 仍不可见，按 v1.1 的 2-attempt/1-engineer-day timebox 转备用主机，而非无限修环境；
4. infrastructure PASS 后只进入 G1 read-only profiling。

## Links

- plan: [G0 admission plan](BOUNDFLOW_NRIR49_G0_GPU_OPPORTUNITY_ADMISSION_V1_PLAN_2026_08_06.md)
- roadmap: [GPU compiler acceleration research v1.1](BOUNDFLOW_GPU_COMPILER_ACCELERATION_RESEARCH_V1_PLAN_2026_08_05.md)
