---
status: draft-implementation-blueprint
date: 2026-08-28
type: implementation-plan
topic: boundflow
slug: asplos27-s4-1c-compressed-gradient-emitter
stage: s04
depends-on: validated-s4-1b-six-site-effective-values
execution-authority: false-pending-s3-external-audit-s4-0-s4-1a-s4-1b
code-change-open: false
gpu-correctness-open: false
timing-open: false
performance-claimed: false
---

# ASPLOS'27 S4-1C：通用compressed α/β gradient emitter实施蓝图

## 0. 直接结论

当pass C的incoming coefficient `A_i`和S4-1B的effective preactivation `V_i`同时可用时，六site dα已经
不再依赖原始operator是Conv、Linear还是Residual：

```text
dα_i = upstream × A_i × V_i
```

再应用lower direction、ambiguous ReLU、`A_i>=0`、compressed feature ownership与clamp endpoint语义即可。

因此S4-1C只需要一个layout-parameterized α emitter模板，formal实例化六次；site31同一边界再发射唯一active dβ。
不应串六个B4-B2 wrapper，也不应materialize dense gradient或保存跨层A。

## 1. 通用数学模板

通用逻辑轴：

```text
D = domain count
S = spec count
F = site logical feature count
W = compressed α width
Q = compressed β entries per domain
```

当前formal为`D=6/S=1`；schema保留S轴，不能把spec=1写进通用模板。

### 1.1 lower-α VJP

对compressed ordinal `k`映射到feature `f=index[k]`：

```text
dα[d,k] = Σ_s upstream[d,s] * A[d,s,f] * V[d,s,f]
```

只有以下条件全真才采用该值，否则为0：

- production lower direction；
- `lower[d,f] < 0 < upper[d,f]`；
- `A[d,s,f] >= 0`，即lower slope被选择；
- `0 <= α[d,k] <= 1`，与PyTorch clamp端点梯度=1一致；
- feature index属于当前slot且唯一。

S4-1A active α shape为`[D,W]`，emitter输出同shape；不再产生full `[2,1,D,W]` gradient。preserved direction
不属于optimizer ABI，也不需要zero-filled output。

### 1.2 sparse β VJP

β在ReLU coefficient中的语义是：

```text
relu_lower_A = incoming_A * selected_slope - beta * split_sign
```

对`location[d,q]`：

```text
dβ[d,q] = Σ_s upstream[d,s] * (-V[d,s,location[d,q]] * split_sign[d,q])
```

B4-B2局部模板写作`-adjoint_relu * split_sign`；在full composition中，`V`是该coefficient的selected-primal
adjoint。这个等价关系必须由site31 B4-B2 sparse Linear、full PyTorch autograd和float64公式三方证明，不能仅凭
符号推导升级claim。

当前active metadata：location=`[17,17,31,17,17,31]`、sign=`[1,1,1,-1,-1,-1]`、shape=`[6,1]`。
通用模板允许`Q>=0`；`Q=0`直接返回S4-1A empty token且launch count=0。

## 2. pass C插入点

reverse order固定：

| 顺序 | site | incoming coefficient到达点 | emitter后动作 | 当前物理资产 |
|---:|---:|---|---|---|
| 0 | 31 | Linear16 right之后、ReLU31 transform之前 | ReLU31→Linear14 | explicit arena |
| 1 | 28 | Linear14 right之后、ReLU28 transform之前 | ReLU28→residual11 | explicit arena |
| 2 | 25 | residual11 stage1输出A26 | emitter→stage2 | D1C scratch `[6144]` |
| 3 | 23 | residual11 stage2输出A24、ReLU23之前 | ReLU23→residual6 | explicit arena |
| 4 | 19 | residual6 stage1输出A20 | emitter→stage2 | D1C scratch `[6144]` |
| 5 | 17 | residual6 stage2输出A18、ReLU17之前 | ReLU17→Conv0 | explicit arena |

每个emitter与后续transform在同一non-default CUDA stream排序，不需要host synchronize。A只存在于现有两个
coefficient arena或现有stage scratch；emitter完成后即可覆盖。

## 3. TIR模板

建议新增：

```text
boundflow/ir/asplos27_s4_compressed_gradient.py
boundflow/backends/tvm/asplos27_s4_compressed_gradient.py
boundflow/runtime/asplos27_s4_gradient_emitters.py
```

这里的IR文件只是TIR template/schedule descriptor，不是solver execution IR。

### 3.1 α PrimFunc ABI

```text
incoming_A[D,S,F] float32
effective_pre[D,S,F] float32
lower[D,F] float32
upper[D,F] float32
active_alpha[D,W] float32
alpha_indices[W] int32
upstream[D,S] float32
→ compressed_dalpha[D,W] float32
```

一个thread负责一个`(d,k)`，S为compile-time reduction。formal S=1，但模板/receipt保留axis identity。

### 3.2 site31 combined α/β ABI

site31可在同一module中导出α与β两个PrimFunc，correctness第一版允许两个launch；只有profile证明launch成本后才考虑
融合成一个kernel。β ABI：

```text
effective_pre[D,S,F]
beta_location[D,Q] int32/int64 normalized
beta_sign[D,Q] float32
upstream[D,S]
→ compressed_dbeta[D,Q]
```

location在prepare阶段normalize为int32并hash绑定；production source int64仍由S4-0 receipt保留。转换只发生一次。

### 3.3 schedule

第一版冻结simple 1-D CUDA schedule：

- α：fuse `(D,W)`，128 threads/block；
- β：fuse `(D,Q)`，当前formal仅6元素，单block；
- no global workspace；
- output caller-owned；
- no atomic、no dynamic allocation、no cooperative group；
- higher-order gradient unsupported。

S4-1C是correctness阶段，不在这里autotune。

## 4. persistent output与metadata

复用S4-1A：

| 输出 | 数量 | elements | bytes(float32) |
|---|---:|---:|---:|
| dα | 6 buffers | 4,248 | 16,992 |
| active dβ | 1 buffer | 6 | 24 |
| empty dβ | 5 tokens | 0 | 0 |
| 合计 | 7 physical + 5 token | 4,254 | 17,016 |

额外prepare metadata：

- compressed α indices总708 int32，2,832 bytes；
- active β normalized locations 6 int32，24 bytes；
- active β signs 6 float32，24 bytes。

这些是logical bytes，不是allocator peak。selected-ReLU使用的dense alpha map属于S4-1B/static metadata；gradient
emitter只需要compressed indices，不再读dense map。

## 5. emission state machine

每个slot的pass C phase：

```text
COEFFICIENT_READY
  → GRADIENT_LAUNCHED
  → GRADIENT_CONSUMED_BY_STREAM_ORDER
  → COEFFICIENT_TRANSFORMED_OR_ARENA_REUSED
```

ordinal 9增加：

```text
GRADIENT_CONSUMED
  → TERMINAL_LA_COPIED_TO_VALUE_SLOT
  → VALUE_SLOT_PHASE = TERMINAL_LA
  → COEFFICIENT_TRANSFORMED
```

即S4-1B effective-value slot只在本site gradient读取完成后才允许改写为terminal lA。六slot shape与对应A完全一致，
因此无需第三套arena。phase tag、generation和one-shot lease必须进入receipt。

非terminal ordinal禁止lA copy；ordinal 9每site恰一次，总terminal lA copy count=6。copy可在后续profile中与emitter
融合，但correctness第一版保持显式。

## 6. 一个logical evaluation的结构账

S4-1C完成后，单evaluation为：

```text
pass A: coefficient/lower + six sign bitmap
pass B: six effective pre values
pass C: coefficient recompute + six dα + one dβ
```

receipt必须区分：

- logical evaluation count=`1`；
- coefficient pass count=`2`；
- effective graph count=`1`；
- α emitter count=`6`；
- physical β emitter count=`1`；
- empty β emitter count=`0`，empty token count=`5`；
- cross-layer saved dense A=`0`；
- gradient output allocation=`0`；
- warm DLPack/Python dispatch=`0`；
- provider/native shadow/fallback=`0`。

不能把两次coefficient pass伪写成一次，也不能把六emitter称为“一个kernel”。

## 7. correctness closure

### 7.1 oracles

每个site至少：

1. production captured/native optimizer autograd gradient；
2. provider-independent full PyTorch autograd；
3. no-autograd float64 closed formula；
4. site25与existing R31B2 P kernel交集；
5. site31与B4-B2 sparse Linear α/β交集。

### 7.2 gates

- six dα key/order/shape=`[D,W_i]` exact；
- active dβ=`[D,Q]` exact，五empty token exact；
- max abs/rel diff `<=2e-5`；
- gradient sign exact；
- unowned dense projection位置全0；
- α clamp端点0/1行为与PyTorch exact；
- five fresh process，逐run绑定A/V/state/module/schedule identity；
- site31 location/sign逐domain exact；
- no terminal lA on ordinal 0；ordinal9 six lA与existing B4-A handoff比较。

S4-1C不接Adam、不计时；只允许单evaluation与terminal-mode correctness fixture。

## 8. negative gates

至少覆盖：

1. A/V/state version不一致；
2. incoming coefficient site错配；
3. effective-value slot错配；
4. α index重复/越界/乱序；
5. active α不是`[D,W]`或full-source escape；
6. stable/non-ambiguous位置产生非零gradient；
7. `A<0`位置产生lower-α gradient；
8. clamp endpoint 0/1错误拒绝；
9. β location越界/重复；
10. β sign/history漂移；
11. empty β触发physical launch；
12. residual stage emitter插在错误stage；
13. emitter后arena在同stream完成前被覆盖；
14. saved dense A或dense gradient出现；
15. warm DLPack/Python per-site dispatch；
16. dynamic output allocation；
17. ordinal非9产生terminal lA；
18. effective slot未消费就覆盖；
19. terminal lA缺slot/重复copy/lease复用；
20.全重签receipt后修改count/bytes/claim；
21. provider/native fallback；
22. timing/performance flag提前为true。

## 9. fail-closed detail code

新增runtime detail至少：

```text
COEFFICIENT_EFFECTIVE_VERSION_MISMATCH
COEFFICIENT_SITE_PHASE_MISMATCH
COMPRESSED_INDEX_IDENTITY_MISMATCH
ALPHA_GRADIENT_SEMANTICS_MISMATCH
BETA_LOCATION_SIGN_IDENTITY_MISMATCH
EMPTY_BETA_LAUNCH_FORBIDDEN
GRADIENT_OUTPUT_POINTER_DRIFT
GRADIENT_GENERATION_MISMATCH
DENSE_GRADIENT_OR_COEFFICIENT_ESCAPE
RESIDUAL_STAGE_INSERTION_MISMATCH
TERMINAL_LA_BEFORE_GRADIENT_CONSUMED
TERMINAL_LA_ORDINAL_MISMATCH
TERMINAL_LA_INVENTORY_INCOMPLETE
CLAIM_FLAG_TRUE_BEFORE_FORMAL
```

均映射到现有GC0 legality/runtime reason，不扩展solver IR。

## 10. 提交与门禁

仅在S3 approved+closed且S4-0/1A/1B依次validated后开放：

1. `feat(tvm): add generic compressed alpha gradient emitter`；
2. `feat(tvm): add sparse beta emitter and staged insertion points`；
3. `test(tvm): close six-site gradient and terminal-lA gates`；
4. `docs: close S4-1C and open S4-1D evaluator closure`。

S4-1C通过只证明single-evaluation gradients和terminal lA phase correctness；不证明10/9 trajectory或性能。

当前状态：

```text
S3 exchange = ready_for_audit
S4-0/S4-1A/S4-1B/S4-1C implementation = closed
S4 timing/performance = closed
```
