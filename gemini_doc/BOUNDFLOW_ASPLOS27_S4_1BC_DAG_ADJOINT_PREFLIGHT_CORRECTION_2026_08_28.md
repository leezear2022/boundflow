---
status: design-correction-required-before-implementation
date: 2026-08-28
type: diagnosis-and-corrected-plan
topic: boundflow
slug: asplos27-s4-1bc-dag-adjoint-preflight-correction
stage: s04
execution-authority: false-pending-s3-external-audit
code-change-open: false
gpu-correctness-open: false
timing-open: false
performance-claimed: false
---

# ASPLOS'27 S4-1B/1C：DAG coefficient-adjoint预检与设计纠正

## 0. 直接结论

S4原蓝图中“用一张普通selected-primal图生成六个`pre_i`，再统一计算
`dα_i=A_i×pre_i`”的假设，已经被真实ResNet2B production state证伪：六个site中五个通过，site19在其
production compressed ownership上出现：

```text
max abs diff = 0.0011564247542992234
gradient sign mismatch = 9
owned width = 132 per domain
```

这不是coefficient arena或输入端点符号错误。现有compiled coefficient pass与generic CROWN在input A上：

```text
max abs diff = 5.029141902923584e-08
sign mismatch = 0
```

因此不得直接实现原S4-1B selected-primal graph。正确owner应改为：

> 对实际CROWN coefficient schedule做精确adjoint replay，令`V_i`成为post-ReLU transformed coefficient
> state的VJP adjoint；随后用同一时刻的incoming `A_i`发射compressed dα/dβ。

原有两块coefficient arena、六张sign bitmap、两个residual stage scratch、37,464-element value arena、
compressed output ABI与terminal lA handoff都继续复用；被否定的是value的推导方式，不是这些物理资产。

## 1. 本轮只读证据

固定输入：仓库冻结的ResNet2B property-0 pre-state、CUDA、六个production topology site、同一objective、
同一α/β/split。探针没有修改production代码，也没有形成性能claim。

### 1.1 compressed α投影结果

| site | owned width/domain | max abs diff | sign mismatch | 判定 |
|---:|---:|---:|---:|---|
| 17 | 164 | `7.3341652751e-09` | 0 | PASS |
| 19 | 132 | `1.1564247543e-03` | 9 | **FAIL** |
| 23 | 121 | `4.06289473176e-08` | 0 | PASS |
| 25 | 86 | `4.47034835815e-08` | 0 | PASS |
| 28 | 178 | `8.19563865662e-08` | 0 | PASS |
| 31 | 27 | `1.63912773132e-07` | 0 | PASS |

比较两侧为：

1. control：完整provider-independent PyTorch CROWN对dense α求`d(-sum(lower))/dα`后投影到production
   compressed indices；
2. candidate：现蓝图的单一selected endpoint/ReLU/primal Conv/Add/Linear图，再乘incoming A。

site19误差远大于冻结`2e-5` gradient gate且有符号错误，所以不能以“其余五site通过”放宽门禁。

### 1.2 active β交叉检查

site31唯一active β在production六个location上仍通过：

```text
max abs diff = 1.1920928955078125e-07
sign mismatch = 0
```

这说明local `dβ=-V×split_sign`公式可保留，但`V`必须来自精确coefficient-program adjoint，不能只凭
“普通selected primal等价”类比。

### 1.3 terminal lA语义复核

provider `BoundRelu.bound_backward`在应用slope/β变换前执行`self.lA = last_lA`；BoundFlow native export也在
ReLU step入口记录lower coefficient。因此terminal lA是incoming `A_i`，不是transform后的coefficient。

ordinal 9每site的固定顺序必须为：

```text
incoming A ready
  → gradient emitter读取A与V
  → terminal模式复制同一个incoming A
  → 以[D,S,*feature] typed view发布lA
  → ReLU coefficient transform / residual stage继续
```

formal当前`S=1`，value arena可物理保存`[D,F]`，但handoff必须恢复`[D,1,*feature_shape]`视图并绑定
spec-axis identity，不能仅以元素数相同证明ABI相同。

## 2. 为什么原假设不够

原图把`V_i`定义为按全局coefficient sign运行原始primal DAG得到的`pre_i`。这个等价在链式或已验证的局部
anchor上成立，但真实CROWN程序还包含：

- residual fanout后的coefficient复制与再汇合；
- staged residual内部A的生命周期；
- ReLU transform后β coefficient injection；
- bias accumulator与input box concretization；
- per-site coefficient state与全局累计state的不同切点。

site19反例证明：仅从原始primal拓扑和全局sign bitmap推导，尚不足以证明每个site的VJP owner。这里不能靠继续
增加特判或另写site19 kernel解决；那会重新制造per-site旁路。

## 3. 修正后的数学owner

对site `i`，定义：

```text
A_i = ReLU transform前的incoming lower coefficient
T_i = A_i * selected_slope_i + beta_add_i
V_i = d lower / d T_i
```

则production loss `L = upstream · lower`下：

```text
dα_i = upstream × A_i × V_i
       （仅lower direction、ambiguous、A_i>=0、owned index）

dβ_i = upstream × V_i × d(beta_add_i)/dβ_i
     = upstream × (-V_i × split_sign_i)
```

关键变化是`V_i`由**coefficient program的精确VJP**定义。普通selected-primal `pre_i`只能在经过逐site
等价证明后作为优化lowering，不能反过来充当规范。

## 4. 修正后的三pass结构

```text
Pass A: coefficient/lower + exact branch/sign trace
  - 两个existing coefficient arena
  - residual11/residual6 stage scratch
  - Ainput/A18/A20/A24/A26/A29 sign/version

Pass B: coefficient-schedule adjoint replay
  - 对Pass A已冻结的typed action sequence做VJP
  - 显式处理fanout duplicate、add accumulate、bias与box concretization
  - 输出六个V_i slot，共37,464 float32 / 149,856 bytes

Pass C: coefficient recompute + compressed emit
  - A_i到达时读取同site V_i
  - 六dα + site31 active dβ
  - ordinal 9在transform前复制incoming A_i为terminal lA
```

Pass B可以由手写typed adjoint schedule或TVM Relax/TIR可审计VJP实现；第一版不得依赖不透明autograd runtime，
也不得退回六个Python wrapper。它仍属于现有CROWN schedule的派生lowering，不新增顶层solver IR。

## 5. 新增S4-1B0门禁

S4-1B实现前插入`S4-1B0 DAG-adjoint reduction closure`：

1. 为coefficient schedule的每类action冻结primal/VJP规则：seed、Linear/Conv right、ReLU transform、β add、
   residual duplicate/accumulate、bias accumulator、box concretization；
2. 用full PyTorch autograd对六site dense α/β作为control；
3. 用不调用candidate module的coefficient-adjoint replay作为candidate；
4. 投影到production compressed indices/location；
5. five fresh process，逐run绑定state、action sequence、A/sign、module和layout hash；
6. 六dα与active dβ均须`max abs/rel <=2e-5`且sign exact；
7. site19必须显式关闭本轮`1.156e-3 / 9 sign mismatch`反例；
8. 近零A位置不得用除法构造oracle，必须直接比较最终gradient；
9. 未通过时S4-1B/1C/1D、Adam trajectory和timing继续关闭。

## 6. 物理设计哪些保留

保留且继续作为implementation输入：

- coefficient arena=`2`；cross-layer saved dense A=`0`；
- residual11 stage1 scratch=`A26/site25 incoming`；
- residual6 stage1 scratch=`A20/site19 incoming`；
- six sign bitmap=`55,296 bytes`；
- six V slots=`37,464 float32 / 149,856 bytes`；
- compressed dα total=`4,248 float32`，active dβ=`6 float32`；
- terminal lA total=`37,464 float32`且handoff count=`1`/rerun=`0`；
- value/lA phase-safe alias只在本siteV已消费且terminal copy完成后成立。

需要改名：`effective primal arena`改为`coefficient-adjoint arena`。它仍不是dense coefficient A，但receipt必须
分别披露A、V、terminal lA的phase和bytes。

## 7. 新增fail-closed/tamper覆盖

至少新增：

1. `ORDINARY_PRIMAL_SUBSTITUTED_FOR_COEFFICIENT_ADJOINT`；
2. `DAG_FANOUT_ADJOINT_OWNERSHIP_MISMATCH`；
3. `RESIDUAL_ACCUMULATION_VJP_MISMATCH`；
4. `SITE19_REDUCTION_COUNTEREXAMPLE_NOT_CLOSED`；
5. `TERMINAL_LA_POST_TRANSFORM_COPY`；
6. `TERMINAL_LA_SPEC_AXIS_IDENTITY_MISMATCH`。

S4-4 formal minimum由64类扩为68类fully outer-resigned tamper；新增64之后的四类为：普通primal替换、fanout/
residual VJP provenance删除、site19错值重签、terminal lA pre/post-transform或spec-axis身份篡改。

## 8. 当前门禁

```text
S3 exchange = ready_for_audit
S4-0/S4-1A implementation = closed pending S3
S4-1B0 = design-corrected, implementation closed
S4-1B/S4-1C/S4-1D = closed behind S4-1B0
S4 timing/performance = closed
```

本报告纠正设计，不升级correctness/performance claim。
