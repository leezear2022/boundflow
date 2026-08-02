# 真实 Verifier IR 集成契约 v1

> 日期：2026-08-03
> 分支：`feat/real-verifier-ir-integration-v1`
> 基线：`d457b22` / tag `ir5-final-validated-nogo`
> 性质：correctness/integration 新路线；不继承 IR-5 性能 claim

## 1. 起因与目标

PR-14A 在真实 αβ-CROWN workload 上记录了 540 个 `compute_bounds` 调用，其中
`activation_bab_bound` 为 394 个，但原 fused capability eligibility 为 `0/394`。PR-14B
whole-query replay 又在 VNN-COMP ResNet-2B prop0 上出现 lower max diff `796.765`、符号
仅 `3/9` 一致，因此 compiler 入口保持 No-Go。

本路线只回答两个 correctness 问题：

1. initial plain-CROWN 能否在保留外部 verifier 中间界语义时通过 Bound/Plan/Task/Schedule IR
   得到等价结果；
2. activation-BaB 调用能否作为 typed external-verifier operation 进入同一编译与调度栈，
   同时让 α/β/split/cuts/termination 的算法所有权继续属于 αβ-CROWN。

## 2. 已独立定位的 ResNet 根因

同一冻结模型、property、box 与 `C` 在 CPU 上复现正式 external lower。逐层对照发现：

- 第一组 ReLU pre-activation 的 BoundFlow IBP 界与 external 基本一致；
- 第二组开始，external 使用递归 CROWN intermediate bounds，而现有 whole-query 路径只使用
  本地 IBP trace；
- 最后一组 ReLU 的本地 IBP width max 为约 `394.717`，external 为约 `2.141`；
- 只改 lower slope 为 adaptive 不能修复，max diff 仍约 `810.805`；
- 注入 external 六组 ReLU pre-activation bounds，并使用 external 的 adaptive lower-slope
  后，max diff 降为 `2.15e-6`，符号 `9/9` 一致。

因此根因是丢失了 external intermediate-bound semantics，加上 relaxation policy 未进入 IR；
不是 ONNX 导入错误，也不是 eager/chunked backend 分叉。

## 3. 所有权边界

| 对象 | 所有者 | IR/运行时责任 |
|---|---|---|
| input box、linear spec、requested bounds | VerificationSpec / query payload | 稳定 identity、shape、dtype、content hash |
| intermediate pre-activation bounds | external verifier dynamic payload | Bound IR 标注来源；runtime 做 count/shape/content 校验 |
| ReLU lower-slope policy | Bound IR | `zero` 与 `adaptive` 必须显式，不允许 runtime 猜测 |
| α/β/split state identity | Bound IR external-call inputs/attrs | typed role、版本/hash、失配 fail closed |
| αβ-CROWN 算法与 termination | external exact backend | 本路线不重写、不声称 compiled kernel |
| backend/region/batch/storage 选择 | PlanTemplate / PlanInstance | 外部 exact backend 也必须是显式 candidate |
| 调用顺序与结果提交 | Task IR / Schedule IR | 一次 query 一次 launch，一次 emit，禁止隐式 fallback |

## 4. 两类合法执行路径

### 4.1 External-semantics plain-CROWN

外部 verifier 提供 input box、`C`、逐 ReLU intermediate bounds、requested outputs 与
relaxation policy。BoundFlow 构建普通 CROWN Bound IR，并将以下事实写入稳定 IR：

- `intermediate_bound_source=external_verifier`；
- `lower_slope_policy=adaptive`；
- 每个 ReLU preactivation 的 primal value identity。

运行时 payload 仍携带 tensor，Bound IR/Plan cache 不嵌入 tensor 内容，但 query identity 必须
包含其 aggregate content hash。ResNet 门禁为 lower allclose 且 property sign 全一致。

### 4.2 Activation-BaB external exact call

α/β/split query 不得伪装成 plain-CROWN graph，也不得 fallback 到无状态 executor。Bound IR
使用显式 external-verifier call，Plan 中只允许 `abcrown_exact` capability；Task/Schedule
必须包含对应 launch 与 emit。执行时调用原 external method，observer 只负责 typed
编译/调度、顺序、identity 与结果证据。

这会把历史 `0/394` 从“fused kernel 不可执行”分解为：

- fused replacement coverage 仍可为 `0/394`；
- typed IR admission/dispatch coverage 应提升到可审计值；
- external exact backend 不得计入 BoundFlow kernel speedup。

## 5. 门禁

### RVIR-1：ResNet semantics

- external intermediate count/shape 必须与本地 ReLU 拓扑一一对应；
- 任一缺失、顺序或 shape 失配均 fail closed；
- frozen ResNet lower max diff `<= 2e-4`，sign `9/9`；
- MLP 等价与旧 zero-policy 行为不回归。

### RVIR-2：IR closure

- relaxation policy/source 出现在 Bound IR canonical JSON 与 stable hash；
- real verifier query 可生成并验证 BFBoundModule、PlanTemplate、PlanInstance、TaskIRModule、
  ScheduleModule；
- α/β/split state identity 不得只藏在 `dict[Any]` 或文档中。

### RVIR-3：execution closure

- external exact schedule 只有声明的 backend 能执行；
- observer on/off 的 solver status、visited domains/final lower 保持一致；
- query/result 数量、顺序与 parent lineage 无丢失或重复。

### RVIR-4：artifact closure

- 冻结 manifest、typed IR hashes、coverage、correctness comparison 与 replay；
- 全量 `pytest tests` 无回归；
- GPU 不可用时允许先形成 CPU correctness artifact，但必须显式写环境边界，不能补写
  CUDA/performance 数字。

## 6. 提交顺序

1. `docs: freeze real verifier IR integration contract`
2. `feat(ir): preserve external intermediate-bound semantics`
3. `feat(ir): type external verifier activation calls`
4. `bench: freeze real verifier IR correctness artifact`
5. `docs: close real verifier IR integration route`

每一步都必须有 focused tests 和修改记录；未过 correctness 前不启动任何性能分支。
