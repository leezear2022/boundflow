# PR-14A 真实 Query Coverage 审计

> 状态：**VALIDATED-PARTIAL**
> 决策：当前 backend 对 activation-BaB 核心查询为 **NO-GO（0/394）**；只允许继续评估
> incomplete/initial plain-CROWN 子阶段，不新增 α/β/split kernel 来强行提高覆盖率。
> 原始工件：`artifacts/phase7a-pr14/pr14a-real-query-trace-20260719-v2/`（本地 ignored，按
> manifest 重生成）

## 1. 审计问题与口径

本次运行未替换 αβ-CROWN 的 branch/split/α/β/cuts/termination，只用可撤销 observer 包装上游
`BoundedModule.compute_bounds`。每次实际调用先转换为 PR-13 `BoundQuery`，再派生
`VerificationQueryProfile`。

`backend_eligible=true` 的严格含义是：

1. ONNX 能通过当前 BoundFlow frontend；
2. query 内至少存在一个当前 PR-12 fused capability 可处理的 affine→ReLU region；
3. method、grad、α/β/split、device/dtype/layout/stage 均通过 capability contract。

它不等于整个 verifier query 已能被 BoundFlow 端到端替换，也不构成性能 claim。

## 2. Workload 与来源

| Workload | 来源 | model SHA256 | property SHA256 | 定位 |
|---|---|---|---|---|
| official simple MLP | αβ-CROWN integration fixture | `2c412204...b91b01d` | `c6eb6a7e...3985e06` | MLP activation-BaB 接线 |
| official simple CNN | αβ-CROWN integration fixture | `e93babc6...4236cd5` | `0968861b...afa9c2` | CNN fail-closed 接线 |
| VNN-COMP 2021 ResNet-2B prop0 | 官方 CIFAR-10 ResNet benchmark | `791aa24d...dc4a6d` | `89edf066...c3769ff` | 2 residual blocks、5 Conv + 2 Linear |

Verifier 固定为 αβ-CROWN commit `e5c7e17bf0488843acb77b7519f59876717a49f4`，其
auto_LiRPA submodule 为 `5a098e8f9fb5786a428a024981d833d303921f2d`。ResNet 来自
VNN-COMP 2021 commit `90419aadcf06cf543ce5c1706cae1059dc9fa6cf`。

## 3. 结果

| Workload | status | query | phase | method | eligible | frontend |
|---|---:|---:|---|---|---:|---|
| simple MLP CUDA activation-BaB | unknown / all nodes split | 377 | init 34；activation 343 | CROWN 369；α 1；αβ 7 | 33 / 377（8.75%） | supported |
| simple CNN CUDA | verified at initial CROWN | 1 | init 1 | CROWN 1 | 0 / 1（0%） | **AveragePool unsupported** |
| VNN-COMP ResNet-2B prop0 CUDA | timeout/unknown | 162 | init 111；activation 51 | CROWN 160；α 1；αβ 1 | 110 / 162（67.90%） | supported |
| 合计（仅作 micro count） | — | 540 | init 146；activation 394 | CROWN 530；α 2；αβ 8 | 143 / 540（26.48%） | — |

真正决定路线的是分 phase 结果：

| Phase | eligible / total | 结论 |
|---|---:|---|
| `alpha_crown_initialization` | 143 / 146（97.95%） | plain-CROWN 初始化值得进入窄化 PR-14B |
| `activation_bab_bound` | **0 / 394（0%）** | 当前 fused backend 不可进入核心 BaB query |

主要拒绝计数为：`optimization_stage_unsupported=396`、
`split_state_unsupported=394`、method/grad/alpha 各 10、beta 8，以及 CNN
`onnx_frontend_unsupported_op:AveragePool=1`。这些失败均保留，未 fallback 后删除。

## 4. Observer-on/off 基线

同一 model/property/config/seed/timeout 各运行 original 与 profile-on：

| Workload | status match | visited domains | final BaB lower |
|---|---|---:|---|
| MLP | PASS（unknown/unknown） | 508 / 508 | `-0.18902308` / `-0.18902308` |
| CNN | PASS（verified/verified） | N/A | initial CROWN solved |
| ResNet-2B prop0 | PASS（unknown/unknown） | 192 / 192 | `-0.31085062` / `-0.31085050` |

ResNet lower 差约 `1.2e-7`，属于两次独立 FP32 solver run 的微小数值差；当前未记录逐 split
lineage，不能据此声称 branch sequence 完全相同。profile 内部 query ID 连续、query/profile
一一对应，0 duplicate/loss。

## 5. 环境与限制

- 机器：RTX 4060 Laptop GPU，CUDA capability 8.9；
- BoundFlow：Python 3.12、Torch 2.12.1+cu132；
- 上游 αβ-CROWN 声明 Python 3.11、Torch 2.11。本次未降级项目 Torch，只补齐其 Python
  依赖，因此证据是 integration compatibility，不是上游正式支持矩阵；
- observer 会 hash tensor 并检查 Python stack，禁止使用 profile-on wall time 作性能数据；
- 当前 external query 只有 identity/profile，没有 tensor payload 和 parent lineage，尚不能做
  PR-14B fixed replay；
- CNN 的 verifier 执行成功，但 BoundFlow frontend 对 `AveragePool` fail closed，所以不能把其
  initial CROWN 记为 backend coverage。

## 6. Go/No-Go 与下一步

1. **NO-GO：** 不做 activation-BaB fused replay，不新增 α/β/split kernel，不把 26.48% micro
   coverage 写成核心 BaB 覆盖；
2. **NARROW GO：** PR-14B 只针对 supported MLP/ResNet 的 initial plain-CROWN，冻结真实调用
   payload，比较 original batched 与现有 PR-12 candidate；
3. PR-14B 若不能在该真实 phase 保持 bounds/property 并获得稳定 latency/peak-memory 收益，则
   C3 正式降级为基础设施，论文主贡献保留 C1+C2；
4. parent/split lineage 只在未来确实需要 activation fixed replay 时补；当前 0% capability 下不应
   为了完整性先扩 observer 或造新 backend。

## 7. 可复现来源

- αβ-CROWN：[official repository](https://github.com/Verified-Intelligence/alpha-beta-CROWN/tree/e5c7e17bf0488843acb77b7519f59876717a49f4)
- official API/fixtures：[complete verifier tests](https://github.com/Verified-Intelligence/alpha-beta-CROWN/tree/e5c7e17bf0488843acb77b7519f59876717a49f4/complete_verifier/tests/fixtures)
- VNN-COMP ResNet：[official benchmark](https://github.com/VNN-COMP/vnncomp2021/tree/90419aadcf06cf543ce5c1706cae1059dc9fa6cf/benchmarks/cifar10_resnet)
