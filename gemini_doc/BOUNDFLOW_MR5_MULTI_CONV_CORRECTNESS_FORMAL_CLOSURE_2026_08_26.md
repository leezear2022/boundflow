---
status: validated
updated: 2026-08-26T20:40:00+08:00
type: closure
topic: boundflow
slug: mr5-multi-conv-correctness-formal
stage: s01
---

# MR5 Multi-Conv Production Bridge Correctness 正式 Closure

## 1. Verdict

Formal artifact已输出：
`VALIDATED-MR5-MULTI-CONV-PRODUCTION-BRIDGE-CORRECTNESS`。

它证明BoundFlow可以在同一个真实αβ-CROWN outer optimized exact call中，按C2→C1→C0接管三条
lower ReLU+Conv路径，同时保留逐site lower、compressed α VJP、10/9 optimizer mutation、owner state、
termination与atomic rollback语义。

这不是性能结果。timing、complete query、queue、B0/B3 parity与ASPLOS-ready仍无新增claim。

## 2. Frozen provenance

- worker source=`3e1a70933910c009019c59de4f44d233a75f7950`；
- formal gate commit=`d2ed121`，lossless-xz amendment=`72155ed`；
- artifact=`artifacts/measurement-recovery/mr5-multi-conv-production-bridge-v1/`；
- summary hash=`293c5c8b902cb7bfa813da03f3b4519a81dab6059b166264f3d98f68f616a718`；
- manifest file SHA256=`15ba6b30f5b62d13d5f04ab3cef3641df6b8decafa5f51f6ca7765ea0dfff443`；
- 三个外部repo、model/property identity与MR4/MR3冻结值一致；
- raw以`raw.json.xz`无损保存，解压后约150 MB，仓库对象约18 MB。

2026-08-26 timing amendment：为正式计时在已验证bridge上追加可选prewarmed cache入口和只读
timing receipt，并补相应负向测试；correctness raw、summary、source commit及所有数值未修改。旧manifest
按fail-closed设计拒绝了code blob漂移，随后以当前code revision重签并成功replay，summary hash仍为
`293c5c8b…a718`。该amendment只恢复correctness证据链，不改变correctness或performance claim。

## 3. Five-pair semantic closure

- 5 pair/10 fresh，顺序=`PB/BP/PB/BP/PB`；
- provider/candidate verdict均`verified`、visited domains均`6`；
- candidate forward/backward=`150/135`；
- general element逐pair重算，global max diff=`5.0067901611328125e-06`；
- optimizer element逐step重算，global max diff=`2.562999725341797e-06`；
- 五pair allclose/sign exact；
- 每evaluation逐site region、inner/outer result、target α、完整module state、final clip均纳入比较；
- 前9次compressed α gradient、Adam exp_avg/exp_avg_sq、pre/post clamp、step/lr均纳入比较。

## 4. Three-site compiler/runtime closure

| Site | Signature hash | Forward/backward per worker | Cache | Workspace adjoint |
|---|---|---:|---:|---|
| C0 | `46151b76…bd377` | `10/9` | `1 miss+9 hit` | `[6,1,8,16,16]` |
| C1 | `aab0a910…0f311` | `10/9` | `1 miss+9 hit` | `[6,1,16,8,8]` |
| C2 | `191debcf…5274a` | `10/9` | `1 miss+9 hit` | `[6,1,16,8,8]` |

- 三个unscheduled/scheduled/device-source receipt跨5个candidate fresh process exact；
- C0/C1真实使用stride-2/output-padding-1，不是复制C2 stride-1 ABI；
- 每site β=`10 × [6,0]`、numel=`0`；
- handoff content=`10/10`，pending/fallback/eager/native shadow=`0/0/0/0`；
- 三site independent PyTorch forward/VJP focused tests通过。

## 5. Atomic、replay与tamper

- evaluation 5、C1 forward后注入失败；
- owner tensor=`12`，content/storage pointer before/after exact；
- version delta=`[1,6]`，staged/commit/rollback=`0/0/1`；
- artifact replay从压缩raw重算全部pair metric，summary hash一致；
- 21/21 fully re-signed attacks rejected，覆盖semantic、optimizer、site order、launch、β、pending、
  signature/module/workspace/fresh drift、rollback、run order与source provenance。

## 6. Validation

- focused=`23 passed`；
- Black、mypy、pylint=`10.00/10`、diff check通过；
- clean post-xz full regression=`1787 passed, 3 skipped, 6 warnings`，耗时`678.83s`；
- 首轮并发运行曾有`1785 passed,2 failed,3 skipped`，两失败均为pytest进程启动后raw从JSON切为
  XZ导致其缓存旧reader；新进程artifact tests 4/4及上述clean full均通过，不是代码回归。

## 7. 唯一合法下一动作

full regression已通过，只开放一个独立预注册的MR5 multi-site outer exact-call timing：

1. baseline=同一αβ-CROWN native provider；candidate=相同solver/state下三site bridge；
2. compile与dummy warm显式排除，headline包含完整outer exact-call wrapper/runtime/materialization；
3. 6 pair/12 fresh PB/BP；host wall为headline，CUDA event与peak memory为诊断；
4. 语义必须继续通过，且不得借用CIBC独立IBP graph的`2.45631x`；
5. 未过预注册parity gate则以NO-GO关闭当前三独立site runtime形态。
