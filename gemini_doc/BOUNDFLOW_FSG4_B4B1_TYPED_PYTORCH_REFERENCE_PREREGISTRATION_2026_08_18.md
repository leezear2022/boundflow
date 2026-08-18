---
status: preregistered-b4b1-typed-pytorch-reference
updated: 2026-08-18T06:24:00Z
type: plan
topic: boundflow
slug: BOUNDFLOW_FSG4_B4B1_TYPED_PYTORCH_REFERENCE_PREREGISTRATION
stage: s01
---

# FSG4/B4-B1 Typed Pure-PyTorch Reference 预注册

## 1. 准入与目标

B4-B0 已在 DocOps exchange `fsg4-b4b0-five-fresh-20260818` Round 2 独立外审批准并关闭。
B4-B1 只建立两个冻结 production 锚点的 typed differentiable lower-region IR 与不依赖 TVM 的
pure-PyTorch reference。B4-B2 CUDA/TIR、性能、显存与 ASPLOS-ready 仍未准入。

冻结锚点不变：

- S-anchor：`semantic-active-beta-gemm-14`，`31/Gemm_14`，active beta；
- P-anchor：`performance-conv-8-candidate`，`25/Conv_8`，empty beta。

## 2. B4-B0 到 B4-B1 的证据缺口

B4-B0 raw 已足以证明现场 capture ownership，但不足以作为自包含的 B4-B1 numerical oracle。
从 v2 run 0 独立重算：现有 incoming A、pre-bounds、native α/β 和 weight 可重建两个锚点的
output A，最大误差分别约 `3.73e-8` 与 `2.98e-8`；但把进入 region 前的 lower bias 与 affine
operator bias 假设为零时，output bias 最大误差分别约 `0.5505364` 与 `1.1097491`。

此外，现有 `loss_seed` 是 whole optimization objective 的最终种子，不等价于中间 region 的
output adjoint。若没有 `dLoss/d(output_lower_a)` 与 `dLoss/d(output_bias)`，不能从局部 reference
独立重放 production native α/β/incoming-A gradients。

因此禁止：

- 从捕获的 target output bias 倒推 incoming bias；
- 从 target native gradient 反解 output adjoint；
- 把缺失 operator bias 伪造成零而不记录 presence；
- 只比较 forward，却把 production gradient 宣称为已重放；
- 调用 TVM 或复用待替换 backend 作为 reference oracle。

## 3. B4-B1a：capture sufficiency amendment

创建新 schema，不改写已外审关闭的 B4-B0 v1/v2 artifact。每个锚点的新 raw capture 必须增加：

1. `incoming_lower_bias`：进入 ReLU step 前的 `state.b_l`，shape=`[6,1]`；
2. `operator_bias_present` 与可选 `operator_bias`：存在时保存真实 tensor，不存在时保持 absent；
3. `output_lower_a_gradient` 与 `output_bias_gradient`：在 live output 上 `retain_grad` 后，由同一次
   production loss backward 产生；
4. α sparse mapping raw：compressed α、feature shape、全部 feature indices、可选 spec lookup；
5. β sparse mapping raw：value、location、sign、可选 bias/update mask；
6. affine input/output logical shape、weight、stride/padding/dilation/groups 与显式 output padding；
7. source/capture/topology/lineage、device/stream/layout/alias/requires-grad 与 presence bitmap。

所有新增 tensor 都必须 raw-first、canonical digest、source/code/protocol-bound。旧 B4-B0 capture
仍只读 replay；B4-B1 不得通过修改旧 schema 或旧 artifact 获得准入。

### B4-B1a 门禁

- S/P 各 5 fresh process，10/10 capture 新字段齐全且离散结构 exact；
- `output_lower_a_gradient`、`output_bias_gradient` finite、shape exact；
- S-anchor：operator bias presence 按 raw，native β/output adjoint/α gradient 必须存在；
- P-anchor：operator bias presence 按 raw，production β empty，禁止伪造 non-empty β gradient；
- 从 sparse raw 单独重建 dense α/β，与 captured native dense 值 allclose、sign exact；
- outer-resigned 负例至少覆盖 incoming bias、operator-bias presence/value、output adjoint、layout raw；
- 任一缺口无法在显式 opt-in observer 中稳定捕获，则状态=
  `BLOCKED-B4-B1-CAPTURE-SUFFICIENCY`，不得实现 B4-B2。

## 4. Typed IR 边界

新增 `DifferentiableLowerRegionIRV1`，stable canonical serialization/hash 至少包含：

- anchor/start-node/producer ordinal、name、kind 与 lower-only polarity；
- domain/spec/logical input/output shapes；
- 每个 tensor 的 role、shape、dtype、layout、stride、requires-grad 与 presence；
- sparse α selection/reconstruction layout；
- sparse β location/sign/value scatter layout；
- ReLU relaxation/sign-select/intercept semantics；
- affine kind、weight/bias presence 与 Linear/Conv attributes；
- 顺序固定为：β scatter → α reconstruction → lower sign-select → intercept/bias reduction →
  β pre-add → Linear matmul 或 Conv transpose-contraction → incoming bias carry；
- fanout=`single-consumer`、stream=`current-default`、alias=`none`；
- source capture hash、lineage hash 与 instance input hashes。

Plan/instance/receipt 分离：静态 IR 不携带 tensor payload；instance 绑定 raw input hashes；reference
receipt 绑定 IR/instance、output/gradient hashes、计数与 tolerance。任何 hash 或 presence 不一致均
fail closed。

## 5. Pure-PyTorch reference

reference 放在新的独立模块，只使用公开 `torch` / `torch.nn.functional` 运算，不调用 TVM，且不
调用 `crown_ibp.py` 的 production private helper。它必须从 sparse raw 开始重建 native α/β，执行
lower-only ReLU + affine region，并输出：

- `output_lower_a`；
- `output_bias`；
- 以 captured output adjoints 构造的 local vector-Jacobian product；
- incoming A、native α，以及 eligible native β 的 gradients。

S-anchor production incoming A 原本 `requires_grad=false`，production parity 不伪造其 gradient；但
micro gate 必须用显式 requires-grad clone 比较 reference 与独立 eager decomposition 的 incoming-A
gradient。P-anchor empty β 必须保持 absent-gradient 语义，不得用零 tensor 冒充。

## 6. Admission 与负向测试

admission 必须逐项拒绝：

- start-node/anchor/producer identity；
- 非 lower-only、upper 或混合 polarity；
- Patches、未知 coefficient representation 或 fanout；
- shape/dtype/device/layout/stride/requires-grad/presence；
- α feature index 重复、越界、lookup 不唯一；
- β location 重复、越界、sign/value/presence 不一致；
- Linear/Conv attributes、groups、output padding、operator-bias presence；
- alias、非默认 stream、NaN/Inf；
- source/topology/lineage/input hash 漂移。

每个拒绝族至少一个专用测试，并断言具体错误类别；禁止以宽泛异常代替合同门禁。

## 7. Correctness/gradient gate

固定 `atol=2e-4`、`rtol=2e-4`，离散结构 exact，finite nonzero gradient sign exact：

- 2 anchors × 5 fresh captures 的 sparse reconstruction、forward A/bias 全过；
- production output adjoint 驱动的 native α、S native β、P incoming-A gradients 全过；
- S incoming-A requires-grad clone micro parity 全过；
- P empty β absent-gradient ownership 全过；
- root replay 从 raw 重建 IR、instance、reference receipt 与全部派生字段；
- coordinated outer-resigned identity/input/adjoint 改写必须语义拒绝；
- targeted、B4-B0/B3 related、full pytest 与 Black/Mypy/Pylint/diff/DocOps 全过。

通过后状态最多为 `VALIDATED-B4-B1-TYPED-PYTORCH-REFERENCE-PENDING-EXTERNAL-AUDIT`。
外审批准前不开放 B4-B2。

## 8. 提交顺序与停止条件

1. `docs: preregister B4-B1 typed PyTorch reference`；
2. `feat: extend B4-B1 production reference capture`；
3. `feat: add B4-B1 typed differentiable lower IR`；
4. `feat: add B4-B1 pure PyTorch reference`；
5. `bench: close B4-B1 five-fresh reference parity`；
6. 独立外审；批准后才另行预注册 B4-B2。

若 capture sufficiency、sparse reconstruction、forward 或 gradient 任一门禁失败，记录 raw 与
NO-GO/BLOCKED 证据并停止；不得以放宽 tolerance、删掉 active-beta S-anchor 或改用 synthetic-only
workload 继续。
