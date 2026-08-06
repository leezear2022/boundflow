---
status: ready-for-g1
updated: 2026-08-06T03:50:00Z
type: plan
topic: boundflow
slug: nrir49-g0-gpu-opportunity-admission-v1
stage: s01
---

# BoundFlow NRIR49 G0 GPU Opportunity Admission v1

## 后续路线说明（2026-08-06）

G0 的环境准入结论与冻结 artifact 保持有效且不作修改。本文的 `ready_for_g1` 和“下一步 G1”是
历史阶段状态：后续 NRIR49A G1 只对 selected-CROWN-only incremental G2/G3 作出
`VALIDATED-NO-GO`，没有关闭 BoundFlow operator→IR→JIT→runtime→memory 的累计全栈路线。
NRIR49A 所得约 `1.0764x` 也只是删除该单一区域的 Amdahl 上限，不是全栈上限。当前路线已由
[Full-Stack GPU Baseline and Attribution v1](BOUNDFLOW_FULL_STACK_GPU_BASELINE_ATTRIBUTION_V1_PLAN_2026_08_06.md)
取代。

## 结论

G0 已关闭：重启后 ASUS firmware 的 `dgpu_disable=0` 生效，NVIDIA driver、BoundFlow Torch、
真实 TVM CUDA TIR、TVM-FFI custom stream、独立 αβ-CROWN Torch，以及跨环境 identity/digest
六项门禁全部 PASS。正式 artifact `ga403uv-post-reboot-20260806-v2` 可重放，状态为
`ready_for_g1`。

非 GPU 的两个原 blocker 已关闭：

- 独立 αβ-CROWN 环境已按官方 `uv.lock` 建成并通过 CPU import smoke；
- 已冻结一个双方整题均为 `verified` 的公开 VNN-COMP workload，solveability 有了合法分母。

这只解除基础设施阻塞并准入 G1 read-only profiling；尚未修改任何 TIR/kernel，也没有产生 GPU
speedup、memory 或用户 `40x` claim。

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
| TVM CUDA/FFI | CUDA TIR `add_one` 实编译/执行，FFI raw stream 与 Torch exact | PASS |
| NVIDIA driver | `610.43.03`，PCI、`/dev/nvidia*` 与 `nvidia-smi` 均可见 | PASS |
| ASUS firmware | `dgpu_disable=0`、`gpu_mux_mode=1` | 独显已启用 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，capability `8.9`，8188 MiB | PASS |
| latest artifact | `ga403uv-post-reboot-20260806-v2` | `ready_for_g1`，0 blocker |

不得把 `TVM CUDA enabled` 或 CUDA wheel 的存在解释成 GPU 可执行；六项 CUDA smoke 必须同时通过
才可关闭 infrastructure gate。

## 独立 competitor 环境

- repo：`Verified-Intelligence/alpha-beta-CROWN@e5c7e17bf0488843acb77b7519f59876717a49f4`；
- submodule：`auto_LiRPA@5a098e8f9fb5786a428a024981d833d303921f2d`；
- lock：官方 `uv.lock` SHA256=`3b5fe60f59e8a48bedbe0fb6b736881261ce76285f8c4dee9c5d45f94fd65d3b`；
- env：Python `3.11.15`、PyTorch `2.11.0+cu130`、auto_LiRPA/abcrown `0.7.2`；
- smoke：`import torch, auto_LiRPA, abcrown` PASS；post-reboot `cuda_available=true`，设备名、capability、
  total-memory 与 BoundFlow 环境一致；
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
5. **已完成**：重启使 `dgpu_disable=0` 生效，并生成 post-reboot artifact；
6. **已完成**：`nvidia-smi`、BoundFlow PyTorch CUDA、TVM CUDA build/run、TVM-FFI stream、
   competitor PyTorch CUDA、同 GPU identity 与输入 digest smoke；
7. **历史下一步（已完成并被取代）**：冻结 G1 profiling schema/门槛并开始 read-only attribution；
   该单区域阶段已关闭，当前执行 Full-Stack 计划。

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

该 smoke 是功能准入，不采集 latency/peak，不触发 DocOps performance rule。首次 post-reboot `v1`
运行保留为失败诊断：真实 kernel 已执行，但读取当前 TVM runtime module 不存在的 `type_key` 元数据时
误报失败。兼容性修复后以 `v2` 重新生成，六项均 PASS。

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

重启后的正式 generate 命令（`v1` 已保留为失败诊断，因此正式证据使用 `v2`）：

```bash
conda run -n boundflow python scripts/run_nrir49_g0_cuda_smoke.py generate \
  --artifact-dir artifacts/nrir49-g0-cuda-smoke/ga403uv-post-reboot-20260806-v2 \
  --abcrown-root ../alpha-beta-CROWN \
  --abcrown-python ../alpha-beta-CROWN/.venv/bin/python \
  --model ../vnncomp2021/benchmarks/mnistfc/mnist-net_256x2.onnx \
  --property ../vnncomp2021/benchmarks/mnistfc/prop_2_0.03.vnnlib
```

实际输出 `{"blockers":[],"g0_cuda_ready":true,"status":"ready_for_g1"}`；同一脚本的 `replay`
子命令已复核 manifest、semantic hash 与 derived gates。

## Validation

- `pytest -q tests/test_nrir49_g0_admission.py tests/test_multiworkload_competitor_e2e_artifact.py`
  → `13 passed`；
- post-reboot CUDA smoke contract tests=`9 passed`；本轮 G0 admission + smoke targeted=`18 passed`；
- GPU 恢复后全量 `pytest -q tests` → `1049 passed, 3 skipped`；剩余 skip 为避免重复 TVM 编译及
  两项缺少冻结 VNN-COMP checkout 的既有边界；
- mypy touched runner clean；Black check PASS；Pylint touched runner/tests=`10.00/10`；
- post-reboot `v2` replay PASS，manifest file SHA、semantic hash 与六项 derived gate 一致；
- `mnistfc:2` 两侧结果均为 `verified`，模型/property/repo commit 已嵌入 artifact；
- competitor repo 与 VNN-COMP sparse clone 均保持 clean；
- `git diff --check`、新文档 whitespace/link 检查 PASS；正式 artifact 无用户目录或临时目录绝对路径，
  probe SHA 与最终 runner 一致；
- 尚未运行任何 GPU benchmark，因此 R002 performance evidence 不适用；G1 才开始只读归因。

## Rollback

- 本分支未改 TIR/math/default policy；回退只需删除 G0 runner/test/docs；
- 外部 competitor `.venv` 与 VNN-COMP sparse clone 不属于仓库内容，可独立移除，不影响 BoundFlow conda
  环境；
- pre-reboot v1—v5 调试 artifact 已移到仓库外的 session temporary holding area，可恢复；正式
  pre-reboot 证据为 v7；post-reboot v1 作为失败诊断保留，正式通过证据为 v2。

## Links

- changelog: [G0 admission changelog](BOUNDFLOW_NRIR49_G0_GPU_OPPORTUNITY_ADMISSION_V1_CHANGELOG_2026_08_06.md)
- current route: [Full-Stack GPU Baseline and Attribution v1](BOUNDFLOW_FULL_STACK_GPU_BASELINE_ATTRIBUTION_V1_PLAN_2026_08_06.md)
- roadmap: [GPU compiler acceleration research v1.1](BOUNDFLOW_GPU_COMPILER_ACCELERATION_RESEARCH_V1_PLAN_2026_08_05.md)
