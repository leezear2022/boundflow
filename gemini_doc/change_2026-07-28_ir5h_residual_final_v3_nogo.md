# 变更记录：IR-5H residual-final-v3（VALIDATED-NO-GO）

> 日期：2026-07-28
> protocol/runner commit：`971a317`
> artifact：`artifacts/ir5/residual-final-v3-20260728`
> 判定：correctness 通过；p90 与双 workload Pareto 门禁失败；停止当前 ASPLOS
> system-performance 路线

## 1. 正式执行与可审计性

v3 在 protocol commit 后第一次消费 fresh final identities `7501/7502`：

- calibration：2 个 chain-CNN × 4 backends = 8 measurements；
- final：2 个 residual-CNN × 4 backends = 8 measurements；
- fixed-single：2；
- from-forward-trace batched-original：2；
- contexts：8；
- policies：6；
- outcomes：48；
- CUDA warm samples：9；
- manifest 精确绑定
  `971a3175af3cadee7eff1138837354740dbff026`；
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU；
- PyTorch/CUDA：2.12.1+cu132 / 13.2。

目录级 SHA-256 integrity replay 与 fresh semantic replay 均通过。

## 2. Correctness 门禁

全部通过：

- 8/8 compiler candidate `semantic_allclose=true`；
- 2/2 legacy from-forward-trace baseline 与 typed reference allclose；
- 2/2 fixed-single 与 batched first query final bounds allclose；
- 两组 single input 都与 batch query zero `torch.equal`；
- input center max diff 均为 `0.0`；
- Global 8/8 contexts feasible；
- 全部 48 outcomes 已生成。

fixed-single 的 final-bound max diff：

- gray：lower `1.2207e-4`，upper `2.4414e-4`；
- color：lower/upper `0.00390625`；

在冻结 allclose 口径内通过。这些差异来自 batch/single CUDA kernel 浮点路径，不是输入
identity 漂移。

## 3. 正式性能结果

### 3.1 Policy summary

| Policy | feasible | regret p50 | p90 | max |
|---|---:|---:|---:|---:|
| fixed-single | 8/8 | 3.809× | 4.053× | 4.053× |
| ordinary typed batching | 8/8 | 1.000× | 1.008× | 1.008× |
| batched-original from trace | 8/8 | 1.100× | 1.123× | 1.123× |
| local greedy | 8/8 | 22.208× | 1341.820× | 1341.820× |
| Global | 8/8 | 1.00385× | **1.26160×** | 1.26160× |
| Oracle | 8/8 | 1.000× | 1.000× | 1.000× |

Global p90 的冻结门槛为 `≤1.20×`，实际为 `1.26160×`，明确失败。

失败 context 是 `final-residual-color-v3:warm-single`：

- Global 因 TVM artifact 已缓存而选择 `compiler:tvm_fused_tir`；
- TVM median `0.53146 ms/query`；
- dense median `0.42577 ms/query`；
- Oracle regret `1.26160×`。

这说明 chain-CNN calibration model 对 residual-CNN 的 cached-backend 相对速度泛化错误；
不是 compile/setup 计费错误。

### 3.2 逐 workload steady median / peak

| workload/backend | median ms/query | measured peak |
|---|---:|---:|
| gray reference | 0.42897 | 12,346,368 |
| gray dense | 0.43818 | 12,017,152 |
| gray chunked | 0.45610 | 12,848,128 |
| gray TVM fused | 0.41543 | 11,105,280 |
| gray legacy from trace | 0.48041 | 10,825,216 |
| color reference | 0.43316 | 17,348,096 |
| color dense | 0.42577 | 16,586,240 |
| color chunked | 0.45543 | 18,704,384 |
| color TVM fused | 0.53146 | 14,716,416 |
| color legacy from trace | 0.47253 | 13,829,632 |

### 3.3 Pareto 与多预算选择

- color compiler frontier 有两个点：dense 更快、TVM 更省显存；
- gray 中 TVM 同时比其他 compiler candidates 更快且更省显存，frontier 只有一个点；
- 因而 `compiler_latency_memory_pareto_all_workloads=false`；
- high/low memory 的 Global plan 在两组 workload 上都为 dense；
- `any_multi_budget_global_switch=false`。

## 4. 冻结门禁判定

| Gate | 结果 |
|---|---|
| architecture families disjoint | PASS |
| compiler/baseline correctness | PASS |
| exact single input identity | PASS |
| Global feasible all contexts | PASS |
| Global p90 regret ≤1.20× | **FAIL：1.26160×** |
| compiler Pareto on both workloads | **FAIL：gray 无 tradeoff** |
| any multi-budget Global switch | FAIL / 未出现 |

IR-5D remediation 成功把原 `70.263×` host-overhead 灾难降到接近公平 baseline，但没有让
calibration-only Global selector稳定优于 ordinary batching，也没有形成跨 workload 的
memory tradeoff。因此 IR-5 最终仍为 **VALIDATED-NO-GO**。

## 5. 路线关闭

按 IR-5C3/IR-5E/G 预先冻结的止损规则：

- 不再旋转 seed、workload 或 final split；
- 不按 `7501/7502` 数据重拟合 selector；
- 不启动 IR-6 cached specialization；
- 不把 gray/color 的局部收益写成系统级 Planner claim；
- 当前 ASPLOS system-performance 路线停止。

仍然成立的成果是 IR-1—4 的 typed Bound/Plan/Task/Schedule/runtime
validated-reduced 机制与 residual correctness；不成立的是 C2 paper-level adaptive
performance claim。后续若重启研究，必须先提出新的、独立的研究假设和数据划分，而不是
在本 final 上继续调参。
