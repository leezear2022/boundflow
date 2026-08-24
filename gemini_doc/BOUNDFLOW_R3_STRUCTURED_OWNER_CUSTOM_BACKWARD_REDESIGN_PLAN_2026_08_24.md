---
status: validated-no-go-r3-1-m0-r3-1b-preregistration-open
updated: 2026-08-25T06:55:00+08:00
type: plan
topic: boundflow
slug: r3-structured-owner-custom-backward-redesign
stage: s01
---

# BoundFlow R3 结构化所有权与自定义反向重设计

## 0. 结论与当前边界

> **2026-08-25 R3-2A closure / R3-2B preregistration**：P-anchor 5对fresh的10/9 optimizer
> trajectory correctness、ownership、memory与12/12 tamper已通过，只开放R3-2B。R3-2B冻结为
> host-wall wrapper-inclusive 3 warmup/30 samples×5 fresh pair；GO仍为geomean>=`1.20x`、worst>=
> `0.98x`、memory<=`1.0x`。见 `BOUNDFLOW_R3_2A_OPTIMIZER_TRAJECTORY_FORMAL_CLOSURE_2026_08_25.md`
> 与 `BOUNDFLOW_R3_2B_WRAPPER_INCLUSIVE_TIMING_PLAN_2026_08_25.md`。

> **2026-08-25 R3-1b3 closure / R3-2A preregistration**：compiled P-α custom VJP 的5对fresh
> correctness与memory门禁已关闭为 `VALIDATED-R3-1B3-COMPILED-FIVE-FRESH`，只开放R3-2A。
> R3-2A现已冻结P-anchor 10 evaluation/9 Adam mutation、逐步动态rebind、five-fresh、memory与
> replay/tamper合同；仍不计时。见 `BOUNDFLOW_R3_1B3_FIVE_FRESH_FORMAL_CLOSURE_2026_08_25.md`
> 和 `BOUNDFLOW_R3_2A_OPTIMIZER_TRAJECTORY_CORRECTNESS_PLAN_2026_08_25.md`。

> **2026-08-25 R3-1 M0 Python rematerialization NO-GO**：5对独立native/candidate的final lower
> 与compressed dα语义全过，saved dense A=0、custom backward mandatory成立；但peak allocated=
> `1.1181179x`且没有compiled bounded-arena module，违反§8/R3-1硬门禁。R3-2A保持关闭。下一只
> 允许预注册R3-1b真正的compiled recurrence，不允许把Python prototype接入optimizer。见
> `BOUNDFLOW_R3_1_M0_PYTHON_REMATERIALIZATION_FORMAL_NO_GO_CLOSURE_2026_08_25.md`。

> **2026-08-25 R3-0 compressed-alpha v2 closure**：v1 fixture 的 dense-alpha admission 已撤销；
> clean source=`8941e66`的v2以P-anchor production compressed alpha `[2,1,6,86]`重建，replay
> 逐字节一致且12/12 tamper拒绝。现在重新只开放R3-1 mandatory custom-backward correctness；
> R3-2A/timing/performance仍关闭。见
> `BOUNDFLOW_R3_0_COMPRESSED_ALPHA_V2_FORMAL_CLOSURE_2026_08_25.md`。

> **2026-08-25 reprioritization**：CIBC R1-A 已按冻结协议关闭为
> `VALIDATED-NO-GO-R1A-ATTRIBUTION`，R1-B/R1-C/R1-D/R2不再开放。该 formal closure 满足本文
> §14 所需的显式 reprioritization 记录，因此现在只开放 R3-0 合同和静态验证器；R3-1及其后续、
> production接入和timing仍关闭。见
> `BOUNDFLOW_CIBC_R1_A_FORMAL_NO_GO_CLOSURE_2026_08_25.md`。

> **2026-08-25 R3-0 closure**：clean source formal artifact/replay与12/12 fully re-signed tamper
> 已通过，状态=`VALIDATED-R3-0-CONTRACT`。现在只开放R3-1 `25/Conv_8`一个evaluation的mandatory
> custom-backward correctness；optimizer mutation、timing和R3-2A及以后仍关闭。见
> `BOUNDFLOW_R3_0_STRUCTURED_OWNER_FORMAL_CLOSURE_2026_08_25.md`。

可以重新设计，但这不是修补或复活 B4-C2。新路线暂命名为
**R3-SO-CVJP（Structured Owner + Custom VJP）**，核心是：

1. lower-bound 系数在 Python、IR 和 autograd 边界始终是结构化 DAG handle；
2. 一个闭合 lower region 只向 PyTorch autograd 暴露最终 `[batch, spec]` lower；
3. dense A 只允许在一次 kernel/region 执行内部以 tile 或至多两个复用 scratch buffer 短暂存在；
4. 自定义 backward 不保存逐层 dense A，而是从压缩 α/β、bounds、weight 和静态 plan 重算；
5. correctness shadow 只能在独立 worker 运行，正式 control path 不得 native+candidate 双算。

本文只冻结设计与可证伪门禁，状态是
`PREREGISTERED-DESIGN-REVIEW-ONLY`。它不开放实现、不形成 correctness/performance/memory claim，
也不改变当前工程顺序：先完成 R0 审计卫生与 R1 协议/目标冻结，再执行 G1
optimized-graph attribution。

## 1. 基础问题与已证实的失败

### 1.1 原始任务

在 α-CROWN 的反向 bound propagation 中，系数矩阵 A 从输出 specification 逆向经过
Linear/Conv/ReLU/Add/reshape 等算子，最终在输入扰动集合上 concretize。α/β 优化需要对最终 lower
反传梯度，所以该计算同时涉及：

- 结构化线性算子组合；
- ReLU 中依赖 A 正负号的 slope/intercept 选择；
- start-node keyed 压缩 α/β 和 split/history；
- 一个 evaluation 的 forward lower 与随后的一阶 autograd backward；
- 10 次 evaluation / 9 次 optimizer mutation 的真实生产轨迹。

局部 CIBC 风格 differentiable TIR 已达到相对 PyTorch `4.89834x`，说明单 anchor 的 CUDA/TIR
物理机会存在；但 B4-C0/C1/C2 的累计结果分别约为 `0.940x`、`0.948x` 和 `0.337—0.349x`，
B4-C2 peak allocated 又升到 `1.3401x`。因此失败发生在表示、所有权和 autograd live set，而不是
“没有一个更快的局部 kernel”。

### 1.2 B4-C2 在代码中具体做错了什么

当前 native structured 路径会在 ReLU 后返回 `SignSplitLinearOperator`，继续延迟组合。B4-C2 的
`dense_lower_once=True` 分支却执行：

```text
bias_A_l = state.A_l materialize
selected_alpha_l = where(bias_A_l >= 0, alpha_l, alpha_u)
dense_lower = bias_A_l * selected_alpha_l (+ dense beta pre-add)
lower = DenseLinearOperator(dense_lower)
```

这一动作在六个 frontier `31/28/25/23/19/17`、每个 optimizer 10 次，共发生 60 次。它把本应是
轻量 operator tree 的状态变成带 autograd history 的系数 Tensor，并跨后续层存活。

与此同时，现有 `_CIBCDenseExactTIRFunctionV3.forward` 使用 `ctx.executor = executor`。executor
强引用 `incoming_lower_a`、preactivation bounds、native α、bias、weight、输出和 gradient buffer，
所以即使 kernel 本身很快，autograd context 仍把大对象图保留到 backward。

这两个问题必须同时消除：只去掉 `dense_lower_once` 会回到 native structured baseline；只改
`ctx.save_for_backward` 仍会逐层产生 dense A。R3 必须改变 region 的语义边界。

### 1.3 不能从 B4-C2 继承的内容

R3 明确禁止：

- 复用或改名 `B4C2MaterializationFrontierObserverV1` 作为新 provider；
- 在六个 ReLU 分别返回 `DenseLinearOperator`；
- 一个 ReLU 配一个 autograd Function，并让下游 Function 保存上游 dense 输出；
- `ctx.executor`、`ctx.tensor = ...` 或任何间接持有 Tensor 的 context 对象；
- 把 C0/C1/C2 的 timing 当作 R3 baseline，或降低门槛让旧结果变 PASS；
- control worker 内运行 native shadow、reference、diagnostic capture 或隐式 fallback。

局部 B4-B2 TIR 只可作为算子语义和 CUDA ABI 的参考，不携带累计性能 claim。

## 2. 设计目标、非目标与不变量

### 2.1 设计目标

R3 v1 必须同时实现：

- lower coefficient 的 first-class structured ownership；
- closed-region、single-owner execution；
- 对真实压缩 α/β、active/empty beta、split/history 的一阶精确 VJP；
- 不跨层保存 dense A；
- 可审计的 buffer/lifetime/allocator receipt；
- 单 site → active-beta site → 双 site → residual DAG → 六 site 的逐级证伪；
- 六 site no-regression 后才允许重开 same-solver B4-D。

### 2.2 非目标

R3 v1 不承诺：

- higher-order gradient；
- training model weights、input center 或 forward bounds；
- 通用动态 shape、多 GPU、混合精度或所有 ONNX op；
- 把 upper path 一起接管；
- 一开始就实现单 kernel 覆盖整个 ResNet；
- 自动证明比 auto_LiRPA、αβ-CROWN、BaB 或 complete query 更快。

### 2.3 必须保持的语义不变量

对每个 start node 和 optimizer evaluation：

- source/topology/lineage、op attributes、shape/dtype/device/stream identity 不变；
- terminal lower、sign、α gradient、β gradient 与 native reference 满足冻结 tolerance；
- α/β 的 path、lookup/index、location/sign、active/empty 状态逐项相同；
- split/history、optimizer mutation count/order、final α/β state 相同；
- 同一个 semantic lower 在 control path 只有一个 production owner；
- candidate 不得改变 branch、termination、timeout 或 bound quality。

## 3. 选择的架构：region-level，而不是 layer-level autograd

### 3.1 被拒绝的粒度

“每层一个 custom Function”看似能控制 `save_for_backward`，实际仍会：

- 创建六段 autograd 边和六个大系数输出；
- 让每层输出成为下一层输入，dense A 继续跨层存活；
- 若每层都从结构化根重算，产生近似 O(L²) 的重复工作；
- 在 residual/fanout 上重复子树或错误拆分 bias ownership。

因此它不是 R3 的主设计。

### 3.2 选定粒度

R3 的 production 单位是一个 **closed lower region**：从输出 spec seed 开始，包含一组
ReLU/Conv/Linear/Add/reshape/slice/bias 节点，直到 input concretization 或另一个显式 typed
consumer。region closure 必须证明中间 lower A 没有外部 consumer。

```text
spec seed + compact α/β + forward bounds + immutable weights
                         │
                         ▼
              StructuredLowerRegionTemplate
                         │  bind current evaluation
                         ▼
               StructuredLowerRegionInstance
                         │
              one custom-autograd boundary
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
     forward region executor    custom VJP executor
     output: final lower        output: dα / dβ
     no dense A escapes         recompute, no A saved
```

upper bound 暂时保持 native structured path。R3 只替换 lower owner，避免把两个方向一起扩大风险。

## 4. First-class Structured Lower IR

### 4.1 Template 与 Instance 分离

`StructuredLowerRegionTemplateV1` 是不可变、可哈希的静态对象：

- node DAG、root、region inputs/outputs；
- op kind、shape、Conv attrs、start-node identity；
- α/β layout schema、split/history schema；
- region closure、fanout/post-dominator 和 liveness plan；
- forward/VJP module receipt、schedule receipt、scratch budget；
- source/code/topology/lineage hashes。

它不得持有本次 evaluation 的 Tensor。

`StructuredLowerRegionInstanceV1` 只绑定本次 evaluation：

- compressed α leaf tensors；
- compressed β leaf tensors（empty beta 是合法零长度，不伪造全零 dense tensor）；
- preactivation lower/upper 和 split/history；
- weights/bias、input center/radius、spec seed；
- storage pointer、shape、stride、dtype、device、version counter 和 current stream；
- plan-owned scratch buffers。

Template 可跨 10 次 optimizer evaluation 缓存；Instance 必须逐次 rebind 并核对 α/β version，不能把
旧 instance 静默复用到 mutation 之后。

### 4.2 节点集合

v1 只开放生产图已经需要的节点：

| 节点 | 输入/输出语义 | 关键验证 |
|---|---|---|
| `SpecSeed` | 生成初始 objective coefficient | spec/start-node identity |
| `ReluLowerTransform` | sign split、α slope、β/intercept bias | A=0 端点、active/empty beta |
| `LinearRight` | A·W 与 bias contraction | transpose/layout |
| `Conv2dRight` | conv-transpose 型 coefficient propagation | stride/pad/dilation/groups |
| `Add` | residual/fanout DAG 合并 | child ownership、无重复子树 |
| `Reshape` | flatten/view 语义 | numel/contiguity |
| `Slice` | concat 逆向切片 | start/stop/axis |
| `BiasSplit` | fanout 时一次性拆分 bias token | 和保持、不得重复累计 |
| `InputConcretize` | center/radius 上得到最终 lower | perturbation contract |

`ReluLowerTransform` 逻辑上输出 `(coefficient_expr, bias_expr)`，但不产生 PyTorch dense coefficient
Tensor。`bias_expr` 也保留在 DAG 中，直到 region terminal 一次性求值，避免每层 `[batch,spec]`
autograd 节点和重复 reduction。

### 4.3 DAG 而不是 tree

Residual/add 和 fanout 必须通过 stable node ID、hash-consing 与 consumer count 保持共享；禁止递归复制
子树。admission 需要验证：

- `node_count <= c * source_op_count`，v1 冻结 `c=4`；
- 每个 node 的 producer ordinal 早于 consumer；
- region 内所有 lower consumers 都被覆盖；
- `BiasSplit` 的子 token 之和恰为父 token；
- 没有 escaped dense/materialized value。

若 DAG 节点数超线性增长，或不能证明 region closure，直接 fail closed，不退回 B4-C2。

### 4.4 与当前 `LinearOperator` 的关系

现有 `SignSplitLinearOperator`、`Conv2dLinearOperator`、`AddLinearOperator`、reshape/slice/right-matmul
提供了语义原型，但它们的 `to_dense()`、`contract_input()` 和部分 row reduction 会递归 materialize。
R3 不直接给这些方法打补丁，而是新增显式 `StructuredCoefficientHandleV1`：

- 只包含 `(template_builder, root_node_id, shape metadata)`；
- `conv2d_right/add/reshape/slice/sign_split` 只追加或复用 DAG node；
- production `to_dense()` 永远抛出 `StructuredDenseEscapeError`；
- 只有 `InputConcretize` 等 typed terminal consumer 可以关闭 region；
- debug/reference materialize 仅在独立 correctness worker、独立进程中开放。

## 5. Lowering、临时 dense A 与 buffer 纪律

“不保存 dense A”不等于 kernel 永远不能计算 A。ReLU slope 选择依赖 A 的符号，Conv/Linear 也需要
真实系数传播。R3 的准确约束是：

- Python/IR/autograd 边界没有 dense A；
- dense A 不作为 region forward output；
- dense A 不进入 `ctx.save_for_backward` 或任何 context-owned object；
- 不为六层分别保留 dense buffer；
- region executor 最多使用两个 plan-owned ping-pong coefficient buffers，或更小的 tile/local/shared
  storage；buffer 在每个 node 后复用；
- 所有 scratch 必须从 PyTorch allocator 预分配并通过 DLPack 零拷贝交给 TVM，v1 禁止隐式 TVM
  runtime allocation；
- forward 返回后 scratch 可由 plan 复用，但不得被 autograd context 强引用为语义状态。

第一实现不要求一个 monolithic kernel。允许一个闭合 region 内使用有限 kernel sequence，只要：

- 中间值都在同一 bounded arena 中复用；
- launch/module/scratch receipt 完整；
- 没有 Python-visible coefficient tensor；
- CUDA Graph capture 前完成编译和 allocation。

后续可基于真实 profiler 决定 vertical fusion、tile fusion 或 kernel segmentation，不能预设“一个大
kernel 一定更快”。

## 6. 自定义 autograd 与 VJP 合同

### 6.1 唯一 autograd 边界

概念 API：

```python
final_lower = StructuredLowerRegionFunction.apply(
    *compact_alpha_leaves,
    *compact_beta_leaves,
    *forward_bound_leaves,
    *immutable_weight_bias_leaves,
    input_center,
    input_radius,
    spec_seed,
    immutable_plan_handle,
)
```

只有 α/β 是本路线允许返回梯度的 optimizer leaves。weights/bias、input、forward bounds 和 spec seed
必须 `requires_grad=False`；否则 admission 拒绝，不静默返回错误梯度。

### 6.2 context 保存规则

Function 必须：

- `ctx.set_materialize_grads(False)`；
- 所有 backward 所需 Tensor 只通过 `ctx.save_for_backward(...)` 保存；
- `ctx` 上只存没有 Tensor 字段的 immutable plan key、input ordinal 和 schema version；
- 禁止保存 executor、PlanInstance、DLPack view cache 或 closure，因为它们可能间接持有 Tensor；
- α/β 的 PyTorch version counter 必须在 backward 前保持一致；
- 使用 once-differentiable/显式门禁拒绝 higher-order gradient。

保存 weights/bounds 是保存已有 storage 的引用，不是复制；receipt 同时报告 logical bytes 与 unique
storage bytes，防止重复计数。无论如何，coefficient-shaped dense A 都不能出现在 saved tensor 集合。

`ctx.plan_key` 只可查到 immutable Template、compiled module 与一个受 liveness plan 约束的空 scratch
arena；registry 不得保存本次 Instance、input Tensor、DLPack input view 或任何有语义的 A 内容。
forward 结束后不允许依赖 arena 中的旧值；backward 必须从 `saved_tensors` 重新构造 Instance，借用
arena并完整重算。这样 bounded scratch 不是藏在 registry 里的 dense checkpoint。

### 6.3 默认 backward：零 dense checkpoint 重算

默认 `M0-rematerialize`：

1. backward 从静态 region plan 和保存的 compact leaves 重放 coefficient recurrence；
2. 只使用两个 ping-pong scratch 或 tile-local state；
3. 同一次 VJP 中累计所有 α/β gradient；
4. 返回与 Function 输入 ordinal 一一对应的 gradients；
5. absent beta 返回 `None`，不构造伪零 tensor；
6. output adjoint、current stream、device、pointer/version 全部 fail closed。

重算会增加 FLOPs，但把 O(Σ layer dense-A bytes) 的 autograd live set 改成
O(max layer dense-A bytes + compact state)。单-site wrapper-inclusive 门禁会直接判断这个交换是否
值得，不能因为理论内存更好就跳过 timing。

### 6.4 可选但未开放的 M1 variant

若 M0 正确但 backward 重算成为唯一瓶颈，可另行预注册 `M1-bitpacked-sign-certificate`：forward
保存逐层 1-bit sign certificate，而不是 float32 A。它理论上约为 dense A 的 1/32，但仍增加跨层
state，必须独立证明：

- A=0 端点语义相同；
- bitpack/unpack 成本与 buffer 可见；
- peak memory 仍 `<=1.0x` baseline；
- M1 相对 M0 的累计 timing 有净收益。

M1 当前是 CLOSED；float16/float32 dense checkpoint 在 R3 v1 中始终禁止。

## 7. Start-node keyed α/β、split/history 所有权

R3 不能只在 P-anchor 的 empty beta 上通过。每个 `ReluLowerTransform` 必须绑定：

- start node ID 与 native preactivation ID；
- production compressed α path、index/lookup 和 source shape；
- production β path、location/sign、active/empty 状态；
- split mask、history hash、domain identity；
- optimizer evaluation ordinal 与 mutation ordinal。

顺序要求：

1. P-anchor `25/Conv_8` 先证明 dense Conv recurrence、compressed α 与 empty beta；
2. S-anchor `31/Gemm_14` 再证明 active beta、location/sign 和非空 β gradient；
3. 双 site 后才接 residual/add DAG；
4. 六 site 需要 10 evaluation/9 mutation 最终 α/β state 对齐。

任何 path/lookup/location/sign 变化都必须在 kernel launch 前拒绝。

## 8. Liveness 与内存证明

### 8.1 三层证据

1. **autograd saved-tensor ledger**：用 `torch.autograd.graph.saved_tensors_hooks` 记录 pack/unpack、
   shape/dtype/device/storage ID/version/bytes/producer ordinal；
2. **PyTorch allocator evidence**：记录 allocated/reserved peak，并用 CUDA memory history/snapshot
   核对大 allocation 的创建和释放；
3. **external/runtime ledger**：TVM scratch 必须来自已登记的 PyTorch buffer；若出现无法被 PyTorch
   allocator 看见的 CUDA allocation，v1 直接拒绝，而不是只依赖 NVML 差值猜测。

### 8.2 禁止项

正式 candidate 必须满足：

- saved tensor 中 coefficient lineage 的 float/half dense A 数量 `=0`；
- `ctx` 直接 Tensor 属性 `=0`，递归可达 Tensor 的 executor/context 属性 `=0`；
- Python-visible intermediate coefficient tensor `=0`；
- implicit `to_dense`/native shadow/fallback/eager recompute `=0`；
- per-layer persistent coefficient buffers `=0`；
- plan scratch buffer count `<=2`，大小不超过预注册 max-live shape；
- warm execution dynamic CUDA allocation count `=0`。

### 8.3 必须报告的 raw 字段

- template/instance/module/schedule/source/topology/lineage hashes；
- region node/edge/root/closure/consumer counts；
- evaluation/mutation ordinal；
- saved tensor logical bytes、unique storage bytes、coefficient bytes；
- scratch count/bytes、high-water mark、reuse events、alloc/free ordinals；
- PyTorch allocated/reserved peak 和 baseline ratio；
- forward/VJP launch count、fallback/eager/shadow/materialization count；
- per-output numeric diff、sign、α/β gradient/state、split/history；
- cold compile、warm execution、CUDA Graph capture/replay 与 copy 口径；
- control/profile 环境与 thermal/power admission。

## 9. 分阶段实现 DAG 与 kill gate

每一级必须在独立提交中先预注册、再实现、再生成 formal artifact。未过上一级，不开放下一级。

### R3-0：合同和静态验证器

只实现 IR schema、Template/Instance、closure/liveness validator、receipt schema 和 negative tests；不接
production、不计时。

通过条件：

- node/shape/topology/start-node/beta/fanout tamper 全部拒绝；
- `to_dense()` production escape 专用负例通过；
- context recursive tensor-reachability checker 和 saved-tensor ledger 单测通过；
- `performance_claimed=false`。

### R3-1：P-anchor 单 site、正确性 only

只接 `25/Conv_8`，一个 evaluation，native reference 在独立 worker。optimizer state 固定，禁止
`optimizer.step()`，每个 worker 恰好执行一次 candidate forward 和一次 **必须存在的 custom
backward**；比较 final lower、`dα`、empty beta、receipt 和 scratch。纯 `no_grad` forward 只可作为
smoke，不能关闭 R3-1，因为它无法证明 M0 VJP、saved-state 和 autograd lifetime 合同。

通过条件：

- five fresh、独立 oracle、max diff/sign 达冻结 tolerance；
- saved dense A=`0`，scratch `<=2`，peak `<=1.0x` native；
- forward/custom-VJP exactly once，`dα` 与独立 oracle 达冻结 tolerance，fallback/eager/shadow=`0`；
- optimizer mutation count=`0`、α/β version 不变；
- 未计时，不形成 performance claim。

### R3-2A：P-anchor 10/9 mutation 轨迹正确性

在 R3-1 的 mandatory-backward correctness 不变的基础上运行 10 evaluation/9 optimizer mutation；
本阶段只关闭完整轨迹正确性，不读取 latency、不形成 performance claim。

通过条件：

- 5 fresh，candidate control path 无 native shadow/fallback/eager；
- 逐 evaluation terminal lower、`dα`/`dβ`（P-anchor beta absent）、mutation input/output 与独立
  native worker 达冻结 exact/allclose/sign 规则；
- final α/β、split/history、optimizer mutation count/order 全部一致；
- saved dense A=`0`、scratch `<=2`、allocated/reserved peak `<=1.0x` native；
- formal raw/replay 能从 mutation 0 重放到 mutation 9，任何中间轨迹漂移 fail closed。

### R3-2B：P-anchor 10/9 wrapper-inclusive 本地物理门禁

只有 R3-2A 通过后才开放计时。baseline 必须是**同一个 P-anchor、同一 10 evaluation/9 mutation
轨迹、native single-owner wrapper**；candidate 是 R3 region owner/custom VJP。两侧 optimizer、输入、
stream、warmup、mutation 次数和 correctness capture 开关完全对称，不得用 no-grad baseline 或局部
kernel latency代替 wrapper-inclusive worker。

GO：

- 5 fresh geomean `>=1.20x`，worst pair `>=0.98x`；
- final α state、terminal lower、optimizer mutation exact/allclose；
- allocated/reserved peak `<=1.0x` native。

否则该实现 kill；不得靠扩大 region 掩盖单-site wrapper/autograd 成本。

### R3-3：S-anchor active-beta correctness

只开放 `31/Gemm_14`，证明非空 β、location/sign 和 β gradient。先 correctness，只有通过后才可另行
预注册 timing。

额外门禁：β gradient 非伪零、unowned native gradient 恰为零、empty/active 两类路径不能共享错误
specialization。

### R3-4：两个相邻 site

具体 site pair 不在本文凭名字猜测；R3-0 必须从 topology、post-dominator 和 escaped-consumer 分析
选出并预注册。要求一个 closed two-site region，而不是两个独立 Function。

GO：

- 5 fresh cumulative geomean `>=1.00x`，worst `>=0.98x`；
- peak allocated/reserved `<=1.0x`；
- node count 线性、scratch `<=2`、saved dense A=`0`。

### R3-5：Residual/fanout DAG

选择最小真实 residual diamond，验证 `Add`、`BiasSplit`、hash-consing 与 consumer ownership。

GO：同 R3-4，另要求 shared node 无重复执行/重复 bias、DAG node count 不超线性门禁。

### R3-6：六 frontier lower region

覆盖 `31/28/25/23/19/17`，10 evaluation/9 mutation，6 fresh interleaved control/candidate。

GO：

- terminal lower、α/β gradients、final α/β、split/history 全部通过；
- candidate control 无 native shadow、dense escape、fallback 或 dynamic allocation；
- peak allocated/reserved `<=1.0x` B3；
- cumulative core geomean `>=1.05x`，worst pair `>=0.98x`；
- cold/warm/break-even 分开披露。

低于 `1.00x` 为 NO-GO；`1.00—1.05x` 只能 `VALIDATED-REDUCED`，不能开放 B4-D 性能路线。

### R3-7：same-solver 传播（原 B4-D 的新前置）

只有 R3-6 GO 才开放，比较必须是同一 αβ-CROWN solver 内 original executor 与 RVIR adapter +
BoundFlow executor；B3 为累计 typed baseline，同时保留 B0 原执行器公平对照。

此阶段重新测 complete query、queue、bound quality、branch、TTV/solved 和 memory。局部 `4.898x`、
R3-2B 或 R3-6 数字均不能代替系统证据。

## 10. 统一 kill 条件

出现任一项立即停止当前 variant：

- dense A 进入 Function output、saved tensor、ctx/executor 或跨层 persistent buffer；
- control path 存在 native+candidate 双算、fallback 或 implicit `to_dense()`；
- semantic/sign/αβ/split/history/optimizer mutation 漂移；
- scratch 超过两个 max-layer buffers，或 warm execution 新增隐式 allocation；
- DAG node count/执行次数随层数超线性增长；
- R3-2B 单 site wrapper-inclusive `<1.20x`，或双 site `<1.00x`；
- 任一正式 stage worst pair `<0.98x`；
- peak allocated/reserved `>1.0x` native；
- 为通过门禁而修改 target、tolerance、worker subset 或 baseline。

失败后只能提出新的 representation/checkpoint/schedule variant并重新预注册，不能返回 C2。

## 11. 建议文件边界（实现时才创建）

```text
boundflow/ir/structured_lower_region.py
boundflow/planner/structured_lower_region_plan.py
boundflow/backends/tvm/structured_lower_region.py
boundflow/runtime/structured_lower_region.py
boundflow/runtime/structured_lower_autograd.py
boundflow/runtime/structured_lower_liveness.py
tests/test_structured_lower_region_ir.py
tests/test_structured_lower_region_autograd.py
tests/test_structured_lower_region_liveness.py
scripts/run_r3_structured_lower_*.py
```

对 `crown_ibp.py` 的 production 接入必须保持 opt-in，且旧 B4-C2 provider 不能成为实现依赖。

## 12. 现有代码与证据入口

GitHub：

- 仓库：<https://github.com/leezear2022/boundflow>
- 当前 PR：<https://github.com/leezear2022/boundflow/pull/60>
- 当前分支：<https://github.com/leezear2022/boundflow/tree/feat/rvir-v4-production-state-ownership-v1>
- 设计所依据的代码基线：<https://github.com/leezear2022/boundflow/commit/f87f737cebffaf10827957682e3196063e4c78ed>
- ReLU structured/C2 dense 分叉：<https://github.com/leezear2022/boundflow/blob/f87f737cebffaf10827957682e3196063e4c78ed/boundflow/runtime/crown_ibp.py#L1401-L1538>
- production reverse traversal：<https://github.com/leezear2022/boundflow/blob/f87f737cebffaf10827957682e3196063e4c78ed/boundflow/runtime/crown_ibp.py#L1945-L2293>
- 当前 structured operator 原型：<https://github.com/leezear2022/boundflow/blob/f87f737cebffaf10827957682e3196063e4c78ed/boundflow/runtime/linear_operator.py#L722-L1092>
- B4-C2 frontier owner：<https://github.com/leezear2022/boundflow/blob/f87f737cebffaf10827957682e3196063e4c78ed/boundflow/runtime/fsg4_b4c2_materialization_frontier.py#L1-L110>
- 当前 TIR autograd/context：<https://github.com/leezear2022/boundflow/blob/f87f737cebffaf10827957682e3196063e4c78ed/boundflow/runtime/fsg4_b4b3_cibc_dense_tir.py#L20-L283>
- exact-call 集成：<https://github.com/leezear2022/boundflow/blob/f87f737cebffaf10827957682e3196063e4c78ed/boundflow/runtime/fsg4_b4b3_cibc_exact_call.py#L35-L419>

仓库内证据：

- `gemini_doc/BOUNDFLOW_FSG4_B4C2_MATERIALIZATION_FRONTIER_KILL_CHANGELOG_2026_08_24.md`
- `gemini_doc/BOUNDFLOW_FSG4_B4C1_PROVIDER_OWNED_LOWER_FORMAL_CLOSURE_2026_08_24.md`
- `gemini_doc/BOUNDFLOW_FSG4_B4C0_CUMULATIVE_CORE_FORMAL_CLOSURE_2026_08_24.md`
- `gemini_doc/BOUNDFLOW_FSG4_B4B2_V2_MANUAL_TIR_FORMAL_CLOSURE_2026_08_24.md`
- `artifacts/fsg4-b4c2-materialization-frontier-pilot/resnet2b-prop0-v1/`

一手参考：

- PyTorch, Extending autograd：<https://docs.pytorch.org/docs/stable/notes/extending.html>
- PyTorch, Autograd mechanics：<https://docs.pytorch.org/docs/stable/notes/autograd.html>
- PyTorch, saved tensor hooks API：<https://docs.pytorch.org/docs/stable/autograd.html>
- PyTorch, CUDA memory snapshots：<https://docs.pytorch.org/docs/stable/torch_cuda_memory.html>
- Apache TVM, MetaSchedule：<https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html>

PyTorch 官方文档要求 backward 所需 Tensor 通过 `save_for_backward` 保存，并指出把 Tensor 直接放在
`ctx` 会让其在 autograd graph 生命周期内保持存活；CUDA memory snapshot 只覆盖 PyTorch allocator，
所以本设计禁止不可见的 TVM 动态 allocation，而不是误把 snapshot 当完整设备账本。

## 13. 外部评审必须回答的问题

1. region-level single Function 是否能完整表达 α-CROWN lower 依赖，是否遗漏了 escaped consumer？
2. `ReluLowerTransform` 的 sign/zero endpoint 和 β location/sign VJP 是否数学正确？
3. M0 重算在六层上的计算复杂度与最小 scratch 是否合理，是否存在更好的 bounded-memory VJP？
4. 保存 bounds/weights 的引用但不保存 A，是否满足 PyTorch version/lifetime 语义？
5. residual/fanout 的 DAG、BiasSplit 和 post-dominator closure 是否充分？
6. `<=2` scratch、node-count `<=4x`、memory `<=1.0x` 与 timing gates 是否过松或过严？
7. 哪个 stage 最可能最早证伪整条路线？
8. 是否存在比该方案更简单、同时满足“不跨层保存 dense A”的实现？

## 14. Rollback 与后续动作

- 设计评审不改 production code，所以 rollback 是删除/修订本预注册文档；
- R3 实现必须新开独立分支和 DocOps exchange；
- R1-A formal NO-GO closure 已留下显式 reprioritization 记录，R3-0 现已开放；
- R3-0 只实现合同、静态验证器、negative tests 与 formal replay，不接 production、不计时；
- R3-1 只有在 R3-0 formal artifact 通过后才开放，不能以 design review 或单元测试提前接入。

## Links

- changelog: `BOUNDFLOW_R3_STRUCTURED_OWNER_CUSTOM_BACKWARD_REDESIGN_CHANGELOG_2026_08_24.md`
- external review prompt: `BOUNDFLOW_R3_STRUCTURED_OWNER_EXTERNAL_REVIEW_PROMPT_2026_08_24.md`
- parent recovery plan: `BOUNDFLOW_FAILED_GATES_DIAGNOSIS_AND_RECOVERY_PLAN_2026_08_24.md`
