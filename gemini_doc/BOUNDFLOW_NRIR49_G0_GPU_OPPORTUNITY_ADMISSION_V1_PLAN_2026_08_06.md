---
status: blocked-one-environment-gate
updated: 2026-08-05T18:22:00Z
type: plan
topic: boundflow
slug: nrir49-g0-gpu-opportunity-admission-v1
stage: s01
---

# BoundFlow NRIR49 G0 GPU Opportunity Admission v1

## 结论

G0 已从“GPU session 不可用”推进到一个可重放的 pre-reboot admission artifact。当前不是 CUDA、
PyTorch 或 TVM 没装好，而是 ASUS firmware 将独显设为 `dgpu_disable=1`；`asusd` 已记录
`dgpu_disable=0` 的 delayed apply，因此下一次重启是恢复 GPU 的必要动作。

非 GPU 的两个原 blocker 已关闭：

- 独立 αβ-CROWN 环境已按官方 `uv.lock` 建成并通过 CPU import smoke；
- 已冻结一个双方整题均为 `verified` 的公开 VNN-COMP workload，solveability 有了合法分母。

因此最新 artifact 只剩 `gpu_infrastructure_ready=false`。G1 仍未准入，尚未修改任何 TIR/kernel，
也没有产生 GPU speedup、memory 或用户 `40x` claim。

## Goal

- 在任何性能优化前，把 GPU、竞品、frontend、solveability、memory/Amdahl 前置条件变成 fail-closed
  机器合同；
- 将“环境阻塞”与“研究路线 NO-GO”分开，避免在没有 GPU 数据时误判 bottleneck；
- 为重启后的 G0 CUDA smoke 和后续 G1 read-only profiling 固定同一入口。

## Scope

- 允许：环境诊断、独立 competitor 环境、公开 workload 资格筛选、artifact/replay、测试和文档；
- 不允许：修改 bound math、production 默认 policy、TIR、fused kernel 或性能门槛；
- 用户报告的 BoundConv `40x` 仅作为 `U-40X-01` 线索；本机未找到其源码，当前 verdict 为
  `NOT-AUDITABLE-SOURCE-MISSING`。

## 已冻结的环境事实

| 项目 | 结果 | 判定 |
|---|---|---|
| 主机 | ASUS ROG Zephyrus G14 / GA403UV，kernel `7.1.5-arch1-2` | `MEASURED-CURRENT` |
| BoundFlow | Python `3.12.12`，PyTorch `2.12.1+cu132`，TVM `0.23.dev0` | import 正常 |
| TVM CUDA/FFI | `tvm.runtime.enabled("cuda")=true`，`tvm_ffi` 可导入 | build 能力存在 |
| NVIDIA driver | `nvidia.ko.zst` 存在，但模块未加载，PCI 与 `/dev/nvidia*` 均不可见 | 不是 benchmark 环境 |
| ASUS firmware | `dgpu_disable=1`、`gpu_mux_mode=1` | 独显当前被禁用 |
| delayed apply | `asusd` 日志包含 `Queueing GPU attribute dgpu_disable = 0 for delayed apply` | 需重启 |
| latest artifact | `ga403uv-pre-reboot-20260806-v7` | `blocked`，仅 1 blocker |

不得把 `TVM CUDA enabled` 或 CUDA wheel 的存在解释成 GPU 可执行；四项 CUDA smoke 必须在重启后
同时通过才可关闭 infrastructure gate。

## 独立 competitor 环境

- repo：`Verified-Intelligence/alpha-beta-CROWN@e5c7e17bf0488843acb77b7519f59876717a49f4`；
- submodule：`auto_LiRPA@5a098e8f9fb5786a428a024981d833d303921f2d`；
- lock：官方 `uv.lock` SHA256=`3b5fe60f59e8a48bedbe0fb6b736881261ce76285f8c4dee9c5d45f94fd65d3b`；
- env：Python `3.11.15`、PyTorch `2.11.0+cu130`、auto_LiRPA/abcrown `0.7.2`；
- smoke：`import torch, auto_LiRPA, abcrown` PASS；当前 `cuda_available=false` 与同一 firmware blocker
  一致；
- BoundFlow 与 competitor 不共享 Python site-packages。探针会移除 BoundFlow 的 `PYTHONPATH/TVM_*`
  后启动 competitor interpreter，防止假阳性。

本机 vendored auto_LiRPA 仍为 `9d100ec`，其 `setup.py` 声明 Torch `>=2.0,<2.9`，所以当前
BoundFlow Torch `2.12.1` 不在其声明范围。source oracle 独立记录了 `BoundConv.bound_backward` 的
`OneHotC -> dense`、Tensor `conv_transpose2d` 和 `Patches` 三条路径。

## 公开 solveability 资格样本

冻结 workload：

```text
repo: stanleybak/vnncomp2021@90419aadcf06cf543ce5c1706cae1059dc9fa6cf
model: benchmarks/mnistfc/mnist-net_256x2.onnx
property: benchmarks/mnistfc/prop_2_0.03.vnnlib
workload_id: mnistfc:2
device: CPU
timeout: 30 s
threads: 8
BoundFlow: alpha_steps=5, search_steps=4, max_nodes=1 -> verified
alpha-beta-CROWN: alpha_steps=5, beta_steps=10 -> verified
```

模型 SHA256=`3a5c9730d60bbf1f9b030e731b438436581efd7c00a28ab683c1ec4b6d3449c4`；property
SHA256=`0c36c00722b6c1701f4d5f17b9d28117711f351c6773e450653cc728a2dd224b`。
BoundFlow 九个 clause 全部在 root 完成，整题为 `verified`；αβ-CROWN 也以 initial CROWN
`verified`。该样本角色严格限定为 `solveability_qualification_only_not_performance_tuning`，两侧 CPU
时间不是公平性能数据，也不得传播为 speedup。

筛选过程中的负证据保留：

- 历史 NRIR-18 `mnistfc:000/cifar10_resnet:000/oval21:000` 的 BoundFlow query 均为 `unknown`；
- αβ-CROWN 官方 targeted/robustness MLP fixture 使用单独输出 assertion，超出 BoundFlow VNNLIB v1
  “non-empty OR”语法；
- 官方 disjunctive MLP fixture 可导入，但 7-node 预算下为 `unknown`；
- MNISTFC properties 1/3/5 在 1-node qualification 下为 `unknown`，2/4 为 `verified`；冻结 2 后不再
  用这些结果调 production policy。

## Frontend coverage

当前 replayed 三拓扑的 observed ops 均落在 selected-CROWN 支持集：
`add/concat/conv2d/flatten/linear/relu/reshape`。MNIST FC、CIFAR ResNet2B 和 OVAL CNN 的已观察导入
路径为 `VALIDATED-REDUCED`，但这不等于 G8 的两个 held-out family 已满足。

`AveragePool` 在当前 ONNX frontend 仍无实现，继续 fail closed。符号 batch 只允许 harness 通过显式
`--input-shape` 冻结；新增测试保证没有 override 时仍拒绝，且 override 不得与 ONNX 固定维度冲突。

## Tasks

1. **已完成**：建立 G0 admission runner、语义 replay、digest/tamper 和 Amdahl 公式测试；
2. **已完成**：建立并锁定独立 αβ-CROWN 环境，固定 repo/submodule/lock digest；
3. **已完成**：冻结 `mnistfc:2` 双方非 unknown qualification 样本；
4. **已完成**：重审 observed frontend op coverage 与 AveragePool 缺口；
5. **等待重启**：使 `dgpu_disable=0` 生效，并重新生成 post-reboot artifact；
6. **重启后执行**：`nvidia-smi`、BoundFlow PyTorch CUDA、TVM CUDA build/run、TVM-FFI stream、
   competitor PyTorch CUDA、同 GPU identity 与输入 digest smoke；
7. **只有全部 PASS 后**：冻结 G1 profiling schema/门槛并开始 read-only attribution；仍不直接做 TIR。

### Post-reboot 六项强制 CUDA smoke

`scripts/run_nrir49_g0_cuda_smoke.py` 已冻结以下六项；缺一项都输出 `status=blocked`、退出码 `2`，
不得进入 G1：

1. `nvidia_driver_device`：`dgpu_disable!=1`、PCI/NVIDIA device node 与 `nvidia-smi` 同时可见；
2. `boundflow_torch_cuda`：BoundFlow Torch 在 custom stream 上实算固定 FP32 vector并核对 digest；
3. `tvm_cuda_build_run`：真实编译并运行仓库 CUDA TIR `add_one`，不是只检查 build flag；
4. `tvm_ffi_custom_stream`：Torch current raw stream 与 TVM-FFI raw stream exact；
5. `competitor_torch_cuda`：清除 BoundFlow `PYTHONPATH/TVM_*` 后，在独立 αβ-CROWN env 实算；
6. `cross_environment_identity_digest`：两套 Torch 的 GPU name/capability/total-memory、固定 vector、
   model/property digest一致，TVM 输入与输出 oracle一致。

该 smoke 是功能准入，不采集 latency/peak，不触发 DocOps performance rule。当前同一 boot 的 dry-run
结果为六项 blocked、exit `2`，说明 fail-closed 路径生效；不是 post-reboot 证据。

## Artifact 与 replay

正式 pre-reboot artifact：

```text
artifacts/nrir49-g0-admission/ga403uv-pre-reboot-20260806-v7/
  admission.json
  manifest.json
```

重放命令：

```bash
conda run -n boundflow python scripts/run_nrir49_g0_admission.py replay \
  --artifact-dir artifacts/nrir49-g0-admission/ga403uv-pre-reboot-20260806-v7
```

预期输出：

```json
{"blockers":["gpu_infrastructure_ready"],"g1_ready":false,"status":"blocked"}
```

artifact 内所有性能字段均为 `performance_claimed=false`；GPU 不可用时 memory reachability 与 Amdahl
share/required speedup 强制保持 `NOT-AUDITABLE/null`。

重启后的唯一 generate 命令：

```bash
conda run -n boundflow python scripts/run_nrir49_g0_cuda_smoke.py generate \
  --artifact-dir artifacts/nrir49-g0-cuda-smoke/ga403uv-post-reboot-20260806-v1 \
  --abcrown-root ../alpha-beta-CROWN \
  --abcrown-python ../alpha-beta-CROWN/.venv/bin/python \
  --model ../vnncomp2021/benchmarks/mnistfc/mnist-net_256x2.onnx \
  --property ../vnncomp2021/benchmarks/mnistfc/prop_2_0.03.vnnlib
```

只有输出 `{"blockers":[],"g0_cuda_ready":true,"status":"ready_for_g1"}` 才允许关闭 G0；随后用同一
脚本的 `replay` 子命令复核 manifest、semantic hash 与 derived gates。

## Validation

- `pytest -q tests/test_nrir49_g0_admission.py tests/test_multiworkload_competitor_e2e_artifact.py`
  → `13 passed`；
- post-reboot CUDA smoke contract tests=`8 passed`；三组 G0 targeted 合计=`21 passed`；
- 全量 `pytest -q tests` → `1014 passed, 37 skipped`；37 个 skip 均为现有 CUDA/环境边界；
- mypy 两个 touched runner clean；Pylint touched runner/tests=`10.00/10`；
- admission replay PASS，manifest file SHA 与 semantic hash 一致；
- `mnistfc:2` 两侧结果均为 `verified`，模型/property/repo commit 已嵌入 artifact；
- competitor repo 与 VNN-COMP sparse clone 均保持 clean；
- `git diff --check`、新文档 whitespace/link 检查 PASS；正式 artifact 无用户目录或临时目录绝对路径，
  probe SHA 与最终 runner 一致；
- 尚未运行任何 GPU benchmark，因此 R002 performance evidence 不适用。

## Rollback

- 本分支未改 TIR/math/default policy；回退只需删除 G0 runner/test/docs；
- 外部 competitor `.venv` 与 VNN-COMP sparse clone 不属于仓库内容，可独立移除，不影响 BoundFlow conda
  环境；
- v1—v5 调试 artifact 已移到仓库外的 session temporary holding area，可恢复；仓库只保留无本机
  用户路径且与最终 runner digest 一致的 v7。

## Links

- changelog: [G0 admission changelog](BOUNDFLOW_NRIR49_G0_GPU_OPPORTUNITY_ADMISSION_V1_CHANGELOG_2026_08_06.md)
- roadmap: [GPU compiler acceleration research v1.1](BOUNDFLOW_GPU_COMPILER_ACCELERATION_RESEARCH_V1_PLAN_2026_08_05.md)
