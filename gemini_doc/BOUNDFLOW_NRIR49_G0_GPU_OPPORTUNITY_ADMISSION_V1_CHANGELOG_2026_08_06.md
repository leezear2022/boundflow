---
status: ready-for-g1
updated: 2026-08-06T03:50:00Z
type: changelog
topic: boundflow
slug: nrir49-g0-gpu-opportunity-admission-v1
stage: s01
---

# BoundFlow NRIR49 G0 GPU Opportunity Admission v1 Changelog

## Summary

- 启动 NRIR49 G0，新增 fail-closed admission runner/test/artifact；
- 建成独立 αβ-CROWN 官方锁定环境，并冻结双方整题 `verified` 的公开 solveability 样本；
- ASUS firmware 已应用 `dgpu_disable=0`，六项 post-reboot CUDA 门禁全部 PASS；
- G0 已关闭并准入 G1 read-only profiling；未修改 TIR/kernel/math，未新增性能 claim。

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
- 新增 post-reboot 六门 CUDA smoke runner：NVIDIA、BoundFlow Torch、TVM TIR、TVM-FFI stream、
  competitor Torch 与 cross-env identity/digest；blocked 时生成诊断 artifact 后 exit `2`。
- 重启后 dGPU、驱动及双方 Torch CUDA 均恢复；首次六门禁暴露当前 TVM runtime module 不提供
  `type_key` 属性，kernel 已完成执行但元数据采集误报失败；改为不触发动态函数查找的稳定 Python
  type identity，并新增回归测试。
- 保留失败诊断 `ga403uv-post-reboot-20260806-v1`；正式 `v2` 六项 PASS 且 replay PASS，状态为
  `ready_for_g1`。

## Validation

- 本轮 targeted：G0 admission + post-reboot smoke `18 passed`；
- GPU 恢复后全量：`1049 passed, 3 skipped`；
- post-reboot `v2` artifact replay PASS；Black check、mypy clean、Pylint `10.00/10`；
- competitor import smoke：Python `3.11.15`、Torch `2.11.0+cu130`、auto_LiRPA/abcrown `0.7.2`；
- solveability：BoundFlow=`verified`、αβ-CROWN=`verified`；
- `rg '/home/lee|/tmp/'` 对正式 pre-reboot v7 artifact 无命中；
- GPU benchmark 未运行，`performance_claimed=false`。

## Decisions

- GPU blocker 已由 `blocked_reboot_required` 关闭；
- independent competitor env 是公平对照的正式组成，禁止复用不兼容的 BoundFlow Torch env；
- `mnistfc:2` 只负责 solveability admission，不作为性能调参样本；
- 用户 `40x` 源码仍缺失，维持 `NOT-AUDITABLE-SOURCE-MISSING`；
- G0 PASS 后只准入 G1 read-only profiling，仍不直接做 G2/G3/TIR。

## Follow-Ups

1. 冻结 G1 profiling schema、measurement protocol 与量化 go/no-go 公式；
2. 开始 selected-CROWN 的 read-only GPU cost attribution，不改默认配置；
3. G1 数据冻结前不启动 G2/G3/TIR。

## Links

- plan: [G0 admission plan](BOUNDFLOW_NRIR49_G0_GPU_OPPORTUNITY_ADMISSION_V1_PLAN_2026_08_06.md)
- roadmap: [GPU compiler acceleration research v1.1](BOUNDFLOW_GPU_COMPILER_ACCELERATION_RESEARCH_V1_PLAN_2026_08_05.md)
