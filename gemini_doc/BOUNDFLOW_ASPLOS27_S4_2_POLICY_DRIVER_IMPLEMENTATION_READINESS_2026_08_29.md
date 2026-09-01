# ASPLOS'27 S4-2：production policy driver 实施就绪审计

status: design-readiness-only
date: 2026-08-29
execution-authority: false
code-change-open: false
performance-claimed: false
same-solver-claimed: false

## 0. 结论

> 2026-08-29施工级修订：本稿完成source/live/readiness诊断，但后续施工复核又关闭了opaque capability消费、
> evaluator family re-arm、version拆分、terminal-best/lA一致性和fresh顺序平衡问题。实现以
> `BOUNDFLOW_ASPLOS27_S4_2_IMPLEMENTATION_CONSTRUCTION_PACKAGE_2026_08_29.md`为准。下文20-worker/
> `25.59 MiB`为已被取代的readiness估算；新formal为24 workers，mandatory transition-tensor floor至少
> `60,550,896 B`，且仍不等于完整artifact。

S4-2 的主方向成立：应把 production α/β 优化循环抽成 sealed、无任意 callback、与 evaluator
representation 无关的 host policy driver，再分别接 native dense oracle 与 compiled compressed evaluator。
但旧蓝图在开工前还需要关闭四个实现歧义：

1. `472,758 B` 只加了 Adam `m/v`，漏掉 step、compressed keep-best、初始比较基准和
   validate-before-commit shadow；在2026-08-29进一步扣除重复计算的residual arena slice后，修正后的已知
   logical subtotal 是 **`491,774 B`**；
2. evaluator、Adam、clamp 或 scheduler 失败后不能承诺恢复成“从未执行”；一旦 evaluator transaction
   已开始或 mutable state 开始提交，run 必须进入 **`POISONED_NO_RETRY`**；
3. 固定 `iteration=10, early_stop_patience=10` 时，源码条件 `patience > 10` 不可达；该分支只能用显式
   synthetic policy variant 测试，不能写成 ResNet fixed-policy 事实；
4. production checkpoint predicate 对当前参数实际在 ordinal `0,6,7,8,9` 检查，不是 `0,5..9`。

本稿只做源码、live observer、functional Adam 与 artifact 的实现就绪诊断。S3 exchange 尚未有批准结果，
所以 S4-0—S4-2 production 代码、formal、timing 和 performance claim 继续关闭。

## 1. 证据边界

### 1.1 pinned source

```text
BoundFlow source:
  b1e73aadd641e1d94b943ca25c6be0d3d6e01bb0

alpha-beta-CROWN:
  e5c7e17bf0488843acb77b7519f59876717a49f4

auto_LiRPA:
  5a098e8f9fb5786a428a024981d833d303921f2d

production import:
  /home/lee/Codes/alpha-beta-CROWN/complete_verifier/auto_LiRPA/
  auto_LiRPA/optimized_bounds.py

frozen trajectory:
  artifacts/rvir-v4-optimizer-step/resnet2b-core-step-trace-v1
```

live probe 使用 αβ-CROWN 自身 `.venv`：

```text
Python: /home/lee/Codes/alpha-beta-CROWN/.venv/bin/python
torch:  2.11.0+cu130
device: RTX 4060 Laptop GPU
```

### 1.2 本稿没有证明什么

- 没有实现 sealed driver；
- 没有执行 compiled evaluator 的 10-step trajectory；
- 没有形成 S4 correctness closure；
- 没有测 timing、显存 peak 或 same-solver speedup；
- live observer 数字只约束当前 fixed workload，不自动推广到 held-out workload。

## 2. production 源码的精确控制顺序

`optimized_bounds.py::_get_optimized_bounds` 的每个 ordinal 顺序是：

```text
set terminal need_grad if i == iteration - 1
→ evaluator / compute_bounds
→ optional prune + recover
→ stop predicate + masked loss
→ update best lower
→ patience = 0 or patience + 1
→ conditional α/β checkpoint
→ stop_all / patience / timeout / no-gradient exits
→ zero_grad
→ backward + Adam update                    # i != iteration - 1
→ β nonnegative projection
→ α [0,1] projection
→ scheduler.step()
→ pruner.next_iter()
```

循环退出后：

```text
pruner.update_best(...)
→ restore checkpointed best α/β
→ restore/update intermediate state according to production policy
→ return best result
```

这里有两个必须写进 driver 状态机的条件语义：

- `stop_all`、`patience > early_stop_patience`、timeout、no-gradient 都在 scheduler 之前退出；
- terminal ordinal 9 只有在没有提前退出时才执行第 10 次 scheduler call。

因此“scheduler 永远调用 10 次”不是通用 policy 语义，只是当前 fixed live path 的观测结果。

## 3. live production policy observer

### 3.1 实际 param-group ABI

live optimizer 有两个 param group：

| group | physical parameters | LR | `batch_dim` |
|---|---:|---:|---:|
| α | 6 | `0.01` | `2` |
| β | 1 | `0.05` | `0` |

共同 defaults：

```text
betas                  = (0.9, 0.999)
eps                    = 1e-8
weight_decay           = 0
amsgrad                = false
maximize               = false
foreach                = null
capturable             = false
differentiable         = false
fused                  = null
decoupled_weight_decay = false
```

旧蓝图未绑定 `batch_dim`。它虽然不是标准 Adam 数学参数，但属于 live param-group ABI，prepare receipt 必须
逐 group 记录并 fail closed；不能只比较 LR。

### 3.2 step scalar

9 次更新后，7 个 physical parameter 各有一个 step tensor：

```text
value   = 9
dtype   = float32
device  = cpu
bytes   = 7 * 4 = 28 B
```

所以 memory receipt 必须区分 CUDA persistent state 与 CPU optimizer scalar；不能把 step 无声排除。

### 3.3 fixed live decision trajectory

live observer 捕获 10 次 policy decision，得到：

| ordinal | improved domains | checkpoint | patience after | stop-all | pruning active |
|---:|---|---|---:|---|---|
| 0 | none（初始化 best） | true | 1 | false | false |
| 1 | 0—5 | false | 0 | false | false |
| 2 | 0—5 | false | 0 | false | false |
| 3 | 0,2,3,4,5 | false | 0 | false | false |
| 4 | 0—5 | false | 0 | false | false |
| 5 | 0—5 | false | 0 | false | false |
| 6 | 0—5 | true | 0 | false | false |
| 7 | 0—5 | true | 0 | false | false |
| 8 | 0—5 | true | 0 | false | false |
| 9 | 0—5 | true | 0 | false | false |

所有 ordinal 的 timeout 都是 false，production physical pruning 始终 inactive，preserve mask / next preserve
mask 都是 `None`。这不允许删除 pruning branch；只说明 fixed formal workload 需要证明 identity/no-prune path，
active pruning 由 synthetic fixture 覆盖。

### 3.4 checkpoint predicate 的精确索引

源码条件是：

```text
i < 1
or i > int(iteration * start_save_best)
or deterministic
or stop_final
or patience == early_stop_patience
or time_spent > max_time
```

代入 `iteration=10, start_save_best=0.5`：

```text
int(10 * 0.5) = 5
i > 5
checkpoint ordinals = [0, 6, 7, 8, 9]
```

ordinal 5 不满足条件。

### 3.5 scheduler 轨迹

fixed live path 的 10 个 consumed-before LR 从：

```text
ordinal 0: α=0.01,                   β=0.05
...
ordinal 9: α=0.008337477621301496,   β=0.04168738810650749
```

到 terminal scheduler 后：

```text
post α=0.008170728068875466
post β=0.040853640344377336
```

固定轨迹的 cardinality 因而是：

```text
evaluation_count             = 10
parameter_mutation_count     = 9
consumed_lr_transition_count = 9
scheduler_step_call_count    = 10
```

## 4. functional Adam 可实现性

live probe 在每次 production `torch.optim.Adam.step()` 之前，以相同 parameter、gradient、m、v、step 和
param-group defaults 调用 pinned `torch.optim._functional.adam`，再比较 production step 之后的状态。

结果：

```text
updates                     = 9
physical parameters/update  = 7
comparisons                 = 63
max parameter diff          = 0
max m diff                  = 0
max v diff                  = 0
max step diff               = 0
all bit exact               = true
```

这只证明 pinned torch/environment 下 functional transition 可以 exact 复刻 live Adam；正式实现仍必须固定
调用参数与 source hash，不能把 private functional API 当成跨版本稳定 ABI。

## 5. representation-neutral keep-best owner

### 5.1 production checkpoint inventory

live production 最终持有：

| 对象 | logical bytes |
|---|---:|
| full best α | 33,984 |
| best β（含5个empty） | 24 |
| best intermediate bounds | 299,712 |
| best lower | 24 |
| `ret_0` checkpoint comparison reference | 24 |

### 5.2 compressed candidate 的合法缩减

candidate optimizer 只拥有：

- 6 个 active lower α：`16,992 B`；
- 1 个 active β：`24 B`；
- 合计 current/checkpointable parameters：`17,016 B`。

preserved α direction 由 S4-0/S4-1A immutable live-source lease 提供，不进入 optimizer，也无需每个 best
checkpoint 复制。只有在下列门禁全部成立时，compressed best checkpoint 才能替代 full production checkpoint：

1. preserved source object/storage/version 在 S4-0→S4-3 不变；
2. active direction 的 per-domain checkpoint mask 与 production exact；
3. restore 后 dense→compressed round-trip exact；
4. full external projection与 production visible α/β exact或满足冻结容差；
5. empty β 继续是 typed zero-element token，不伪造 nonempty tensor。

### 5.3 fixed intermediate bounds 不应复制

当前 policy `fix_intermediate_bounds=true`。candidate evaluator 把 `relu_pre_lower/upper` 当 immutable prepared
input；它们不是 S4-2 mutable policy state。因此 candidate 不应照搬 production 的 `299,712 B`
`best_intermediate_bounds` clone，而应：

- prepare 时绑定 object/storage/version/content hash；
- 每次 evaluate 前做 O(1) identity/version guard；
- S4-3 precommit 再从 current provider mapping 复核；
- 任何变化 fail closed。

这是一项 representation/ownership 改写，不是删掉 production 语义；formal 必须比较最终 external
intermediate container 与 official post visible state。

## 6. 修正后的 logical memory ledger

S4-1D 的 `389,574 B` 已包含 current compressed parameter/gradient、lower、terminal lA、dense bridge、coefficient
arena内residual scratch views、tokens/metadata，但不包含 policy driver state。旧`438,726 B`把两个arena slice误作
独立storage，已被取代。

其中每次S4-1D evaluate的Pass C结构已进一步冻结为nonterminal 17-action或terminal 23-action；S4-2只负责在
evaluation之间做受控state re-arm/version transition，不能在driver内改变Pass C顺序、偷偷增加第11次CROWN或把
terminal copy移到dβ31之前。

### 6.1 persistent policy state

```text
S4-1D evaluator state                         389,574 B
current Adam m/v                               34,032 B
current step scalars (CPU)                         28 B
compressed best α/β checkpoint                 17,016 B
best lower                                         24 B
ret_0 comparison reference                         24 B
```

### 6.2 validate-before-commit shadow

为了在修改 stable prepared buffers 前完成 finite、Adam equation、clamp 和 scheduler-next-state 校验，S4-2
必须用 out-of-place transition shadow：

```text
next compressed parameters                     17,016 B
next m/v                                       34,032 B
next step scalars (CPU)                            28 B
shadow subtotal                                51,076 B
```

prepared evaluator 的 stable parameter pointers 不能每步 pointer-swap；validated shadow 需要 copy-commit 到
stable current buffers。任何 mid-copy/hidden-state failure 都进入 poison，不承诺回滚 `_version`。

### 6.3 subtotal

```text
389,574 + 34,032 + 28 + 17,016 + 24 + 24 + 51,076
= 491,774 B
```

设备分解：

```text
known CUDA logical bytes = 491,718 B
known CPU logical bytes  =      56 B
known cross-device total = 491,774 B
```

若后续 S4-3 继续使用已冻结的 full candidate + rollback `68,016 B`：

```text
known S4-3 subtotal = 491,774 + 68,016 + persistent upper/depths 48 = 559,838 B
```

其中upper为CUDA 24 B、depths为CPU 24 B；working-β location/sign 72 B是external retained liveness，不重复计入new
allocation。详见2026-08-29 S4-3 implementation-readiness审计。

这些都不是 `torch.cuda.max_memory_allocated()` 或 reserved-memory claim；policy masks、Python objects、allocator
metadata、module/cuDNN/TVM workspace、model/fixed inputs仍需实测披露。

## 7. 事务与失败状态机

### 7.1 状态

```text
PREPARED
  → EVALUATING
  → POLICY_DECIDED
  → TRANSITION_STAGED
  → TRANSITION_VALIDATED
  → COMMITTING
  → READY_NEXT | TERMINAL_READY

任意 post-begin failure
  → POISONED_NO_RETRY
```

### 7.2 清洁拒绝边界

只有下列发生在 evaluator begin 和 mutable staging 之前的 admission/preflight 错误可以
`REJECTED_CLEAN`：

- policy/program/evaluator/source hash 不匹配；
- param-group ABI、dtype/device/shape/order 不匹配；
- prepared pointer/version guard 不匹配；
- run 已消费或 ordinal 不合法。

### 7.3 为什么不能承诺 rollback

- evaluator transaction begin 后可能已经改变内部 arena/generation；
- stable parameter/moment copy-commit 会增加 PyTorch `_version`；
- scheduler、optimizer 或外部库可能有 tensor 之外的 hidden state；
- 把数值 copy 回去不能恢复“从未发生”的版本与别名观察。

因此：

- evaluator post-begin 失败：poison；
- functional Adam shadow 计算/验证失败：run poison，stable current state不得继续消费；
- stable buffer commit 中途失败：poison；
- clamp/scheduler next-state 验证失败：poison；
- 不允许 retry、resume 或把同一 prepared run 返回 `READY`。

实现可以在 debug artifact 中保存 before/shadow/partial-commit raw，但不能把它宣传为 production rollback。

## 8. early-exit 与 synthetic fixture

### 8.1 fixed patience 分支不可达

源码从 `patience=0` 开始，每次 evaluation 最多加 1，并在更新后检查 `patience > 10`：

```text
10 evaluations → maximum patience = 10 → no early-stop
11 evaluations → ordinal 10 reaches patience = 11 → early-stop
```

所以旧蓝图的“patience超过10”不能用 fixed `evaluation_limit=10` fixture 触发。

### 8.2 合法 synthetic policy variants

driver 必须支持 sealed、枚举式 test-only program，不接受任意 callback：

- `evaluation_limit=12, early_stop_patience=10`：触发 `patience > 10`；
- 或更短 test-only patience，但必须用不同 program hash；
- ordinal 3 stop-all；
- scripted test clock 触发 timeout；production formal 只能用 sealed monotonic clock；
- partial preserve mask；
- 不同 domain 在不同 ordinal 取 best；
- ordinal 0 no-gradient early exit。

这些 fixture 只证明 control state machine，不进入 ResNet performance artifact，也不能与 fixed production policy
共用 program hash。

## 9. formal artifact 冻结（历史readiness口径，施工包已取代）

### 9.1 fresh-process 拓扑修正

旧“five-fresh”虽然解释为五对、每个成员独立process，但无法平衡正反顺序，现修正为：

```text
A/B: 6 pairs = 12 fresh worker processes  # AB/BA各3
B/C: 6 pairs = 12 fresh worker processes  # BC/CB各3
total         = 24 fresh worker processes
```

每对顺序预注册并严格平衡；任一worker缺失、重复或失败，整组作废，不允许resume。

### 9.2 mandatory transition-tensor floor修正

旧candidate-only估算每ordinal保存parameter/gradient/after、m/v before/after、step与lower，但漏掉functional Adam
unprojected parameter shadow和最终restore state，也把A/B dense raw按candidate尺寸外推。施工包按path修正为：

```text
A = 10*(parameter before/after) + 9*gradient
    + 10*(m/v before/after) + 10*(step before/after) + 10*lower
    + 9*unprojected shadow + restored state + terminal lA

B/C同式，但terminal evaluator derivative也保存，所以gradient count=10。
```

A/B/C per-run mandatory tensor floor=`2,837,288/2,871,296/1,511,936 B`；6A+12B+6C合计
`60,550,896 B = 57.74583435058594 MiB`。这仍排除policy projection、receipt、source和容器开销；正式manifest
必须给raw实际bytes，而不是只给hashes。

### 9.3 replay 必须独立重算

- functional Adam 逐元素 transition；
- clamp、LR 和 scheduler transition；
- keep-best/checkpoint/patience/stop/prune decisions；
- 10/9/10 fixed-path cardinality；
- terminal restore 与 dense round-trip；
- source/code blob/module/ABI hashes；
- 全重签 tamper 在外层 digest 更新后仍被语义拒绝。

## 10. 对旧蓝图的修正清单

1. memory subtotal：历史`472,758 → 540,926 B`，2026-08-29 arena-slice owner复核后再纠正为`491,774 B`；
2. downstream S4-3 subtotal：历史`540,774 → 608,942 → 608,990 B`，同次复核后纠正为`559,838 B`；
3. Adam ABI 增加 per-group `batch_dim` 与 step CPU/float32；
4. checkpoint ordinals 冻结为 `0,6,7,8,9`；
5. terminal scheduler 改为 fixed live path 事实，不写成所有 exit 的无条件行为；
6. mutation/evaluator failure 改为 `POISONED_NO_RETRY`，删除虚假 rollback 承诺；
7. patience synthetic fixture 使用独立 sealed program；
8. 历史readiness曾冻结A/B与B/C各五对；施工复核发现3/2顺序不平衡，现以各六对、共24 fresh workers取代；
9. formal raw 明确最低 payload 预算；
10. fixed production pruning physical path标为 inactive，但 active branch仍由 synthetic fixture覆盖。

## 11. 实施顺序（S3 外审批准后才可执行）

1. `feat(runtime): capture and validate live policy and optimizer ABI`；
2. `feat(runtime): add sealed policy program and representation-neutral driver`；
3. `feat(runtime): add functional Adam shadow transition and poison semantics`；
4. `test(runtime): close native A/B policy parity and synthetic branches`；
5. `feat(runtime): bind compiled compressed evaluator`；
6. `test(runtime): close B/C 10/9/10 all-state trajectory`；
7. `artifact: freeze 24-worker balanced raw replay and fully re-signed tamper`；
8. `docs: close S4-2 and only then open S4-3 implementation`。

## 12. 当前停止点

```text
S3 exchange/audit                          ready_for_audit / not approved
S4-2 source/live/functional diagnostics   complete
S4-2 implementation-readiness design      complete
S4-0..S4-2 production code                closed
S4-2 formal/timing/performance             closed
S4-3/S4-4/S4-P                            closed
```

下一外部动作仍是审计
`.docops/exchange/asplos27-s3-optimizer-runtime-20260828/request.md`及其
`r001/delivery.md`；本稿不改变 `.docops/s.md` 的 stage、blocker 或 next。S1/S2历史交接的仓库内真实文件名为
`gemini_doc/BOUNDFLOW_ASPLOS27_S1_S2_COMBINED_EXTERNAL_AUDIT_HANDOFF_2026_08_28.md`，不应使用不存在的
`...AUDITOFF...`路径。
