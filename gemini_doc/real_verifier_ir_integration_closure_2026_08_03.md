# 真实 Verifier IR 集成路线关闭审计

> 日期：2026-08-03
> 分支：`feat/real-verifier-ir-integration-v1`
> 起点：`d457b22` / tag `ir5-final-validated-nogo`
> 代码与 artifact 基线：`e03b3d2`
> 最终判定：**VALIDATED-REDUCED（CPU correctness/integration）**

## 1. 为什么启动

PR-14A/B 暴露两个不能靠文档绕过的问题：394 个 activation-BaB 调用对原 fused capability
为 0 eligible；VNN-COMP ResNet-2B prop0 的 BoundFlow local whole-query lower 对 external
max diff 为 `796.765`、sign 仅 3/9。IR-5 系统性能路线随后又因 p90/Pareto 门禁失败关闭。

本路线没有重启性能调参，也没有重写 αβ-CROWN；目标是让真实 verifier 语义以明确所有权进入
已经完成的 Bound/Plan/Task/Schedule 栈。

## 2. 实现闭环

### RVIR-1：external intermediate semantics

- ReLU intermediate-bound source 与 lower-slope policy 进入 Bound IR canonical JSON/hash；
- capture 拥有逐 ReLU pre-activation bounds 与 aggregate hash；
- count/order/shape 任一失配均 fail closed；
- ResNet 使用 6 组 external bounds + adaptive slope 后 lower max diff `3.09944e-6`，sign
  9/9，门禁 PASS。

### RVIR-2：typed external exact call

- Bound IR v1.1 新增 `EXTERNAL_VERIFIER_CALL`；
- Plan 新增 external region/backend，Task 新增 external-call unit/dependency；
- Schedule 约束一次 launch、一次 emit，只允许 `external_abcrown_exact_call/v1`；
- α/β/split identity 缺失 fail closed，算法与 termination 所有权仍属于 αβ-CROWN；
- profiler v2 显式记录 requested lower/upper、query/result order 和 parent lineage。

### RVIR-3：真实 execution closure

官方 αβ-CROWN `e5c7e17` simple MLP CPU BaB 执行无 observer baseline 与 typed observer：

| 项目 | baseline | typed observer |
|---|---:|---:|
| solver status | unknown | unknown |
| visited domains | 380 | 380 |
| final lower | -0.18902308 | -0.18902308 |
| query/dispatch/completed | N/A | 377/377/377 |

其中 activation 343 个，effective method 全为 αβ-CROWN；377 个调用均显式 lower-only，347 个
非根 query 均指向已出现的父 query。

### RVIR-4：artifact closure

`artifacts/rvir/rvir-cpu-correctness-v1-20260803` 内嵌 394 个历史 activation query 的完整
identity 与五层 IR hash。fresh process 不依赖 ignored 历史目录即可逐行重新编译和比较：

- typed admission：394/394；
- workload：simple MLP 343、ResNet 51；
- effective method：αβ-CROWN 394；
- artifact integrity + semantic hash replay：PASS。

## 3. 验证总账

- focused Bound/Plan/Task/Schedule/adapter tests：72 passed；
- artifact + integration focused tests：11 passed；
- targeted mypy：6 source files，Success；
- artifact runner/test pylint：10.00/10；
- 全量：`452 passed, 37 skipped`；
- `git diff --check`：PASS。

## 4. 不得升级的主张

1. 历史 fused replacement coverage 仍为 `0/394`；394/394 是 external exact-call typed
   admission，不是 BoundFlow kernel 执行；
2. 历史 adapter v1 全部 394 行缺 split tensor values、requested polarity 与 parent lineage；
   artifact 明确写入三项 limitation；
3. 当前 adapter v2 真实在线执行只覆盖官方 simple MLP CPU；
4. 本机 NVIDIA 驱动不可通信，没有 fresh CUDA 证据；
5. typed validation/hash 带来明显开销，且 external lower-only 的公平性能合同尚未建立；
6. 因此不形成 speedup、GPU、完整 VNN-COMP E2E 或 ASPLOS-ready claim。

## 5. 最终状态与后续准入

RVIR-1—4 已在约定的 CPU correctness/integration 范围内做完，路线关闭为
VALIDATED-REDUCED。IR-5 保持 VALIDATED-NO-GO，IR-6 仍不启动。

若未来开启性能路线，必须使用新分支、新 protocol 和 fresh GPU evidence，并先满足：精确
lower-only 同输出合同、observer/audit 成本与 production timed path 分离、至少一个 non-toy
完整 verifier workload、预注册停止规则。不得直接把本 artifact 改写成性能结果。
