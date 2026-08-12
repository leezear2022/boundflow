# RVIR-v4 V4-2 Optimizer Mutation Replacement 预注册计划

日期：2026-08-13

## 1. 目标与非目标

V4-2的目标是让BoundFlow从V4-0冻结的production pre-state独立执行与αβ-CROWN相同的optimizer
mutation，并产出等价的lower、post α、post SparseBeta与原子copy-out receipt。候选执行期间
provider optimizer/core callback与fallback必须为`0/0`。

本阶段不替换整个`update_bounds_core`，不比较性能，不准入B2，不把V4-1的post-state复算冒充
optimizer replacement。

## 2. 冻结输入

- source capture：`artifacts/rvir-v4-production-state/resnet2b-core-capture-v2`；
- capture SHA256：`99c2c0766621bc0c7db77e1aa4f9f262baa07ff2f9d64d742984a03de000df1e`；
- capture manifest SHA256：`8706e1176a9d29a232fcc8d455a88c7889920f34ad70c8fb75fd0c711142d255`；
- V4-1 topology/state基线：`9be36162…bca35` / `8f8cd55d…793fe`；
- workload：VNN-COMP CIFAR10 ResNet2B property 0，1个production core、6个child domain；
- mutable paths：6个start-node `/49` alpha与6个SparseBeta value，共12个receipt；正式post中
  6个alpha和1个非空beta发生变化，合计7个changed receipt。

## 3. Production 策略事实

| 字段 | 值 |
|---|---|
| iteration | `10` |
| alpha learning rate | `0.01` |
| beta learning rate | `0.05` |
| polarity | lower=`true`，upper=`false` |
| intermediate bounds | fixed=`true` |
| deterministic optimization | `false` |
| stop criterion | `auto_LiRPA.utils.stop_criterion_batch_any.<locals>.<lambda>` |

provider的`optimized_bounds.py`执行`range(iteration)`：共10次bound evaluation，仅前9次执行
backward/Adam step，最后一次只评价和保存best。正式call tree对应1个outer optimized call和10个
depth-1 backward calls。

## 4. 当前实现差距

1. `NativeAlphaBetaOptimizerPolicy`只有统一`lr`，不能表达`0.01/0.05`；
2. `run_alpha_beta_crown_mlp`执行`steps+1`次evaluation和`steps`次更新，与production的
   `iteration`语义不同；
3. native optimizer重新执行IBP取得`relu_pre`，尚不能锁定production external intermediate bounds；
4. V4-1的compressed-alpha/layout/SparseBeta映射封装在post-state evaluator内，尚未成为可复用的
   pre-state initializer与post-state projector；
5. 当前capture只有每次nested call的result digest与pre/post metadata，没有逐step tensor payload，
   不足以独立比较mutation trajectory；
6. native best-of、early stop、Adam state、parameter clipping与production copy-out尚未形成同一事务合同。

## 5. 执行切片

### V4-2A — Policy 与迭代语义

- native policy显式拥有alpha/beta learning rate；旧统一lr调用保持hash兼容；
- production模式定义`evaluation_count=iteration`、`update_count=max(iteration-1,0)`；
- 参数组、iteration=0/1/10、非法学习率全部有正负测试；
- 不宣称数值parity。

### V4-2B — Step Trace Artifact

- 在不修改vendored auto_LiRPA的前提下，作用域内观察outer optimizer的10个depth-1 calls；
- 保存每step lower、alpha、SparseBeta value及Adam step ordinal的原始tensor payload和digest；
- replay重算snapshot/digest、call lineage、10 evaluation/9 update与mutable path集合；
- state、step ordinal、result或policy被篡改且外层manifest同步重签后仍fail closed。

GPU正式生成门禁：`torch.cuda.is_available=true`且NVML/driver probe无错误。当前机器的
`cudaGetDeviceCount error 803`与NVML driver/library mismatch只允许记录blocker，不允许生成CPU替代工件。

### V4-2C — Pre-state Native Initializer

- 把V4-1 topology/layout逻辑抽成共享mapper；
- pre α按`alpha_indices`恢复dense domain slopes，SparseBeta按location散射；
- round-trip回compressed coordinates逐path exact schema、finite、数值`2e-4`；
- external intermediate bounds和split/history hash必须与capture exact。

### V4-2D — 10-step Mutation

- 使用独立alpha/beta Adam parameter groups；
- 10 evaluation/9 update、lower-only、fixed intermediate、batch-any stop与clipping逐项复刻；
- 每step lower、alpha、beta与provider trace比较；离散结构exact，finite float按`2e-4`；
- 任一步超差即NO-GO，不只比较最终lower。

### V4-2E — Atomic Copy-out

- candidate只写私有clone；全部step和post-state验证通过后一次性构造新immutable snapshot；
- 12个mutable path必须一次提交，read-only/history/layout不可变化；
- NaN、shape/path drift、stop-policy drift或超差时pre snapshot保持byte/hash不变，callback/fallback=`0/0`；
- post α/post beta/final lower与production通过后，V4-2才关闭。

## 6. Formal Acceptance

V4-2必须同时满足：

1. source/model/topology/policy identity exact；
2. 1 core、6 domains、10 evaluation、9 update、12 receipt、7 changed的正式结构一致；
3. 逐step与final lower、post alpha、post beta在`atol=rtol=2e-4`内，sign exact；
4. callback/fallback=`0/0`；
5. atomic commit与失败回滚负向测试通过；
6. artifact original replay和重签名tamper probes通过；
7. focused、全量、mypy、Pylint、DocOps lint全部通过；
8. `performance_claimed=false`，B2仍关闭。

任一项失败：V4-2保持NO-GO，保留artifact与failure rows，不进入V4-3或B2。

## 7. 当前唯一下一动作

先实现V4-2A的双学习率/production iteration typed合同；同时准备V4-2B capture schema。GPU恢复前
不得生成formal step-trace artifact，也不得以V4-1 post-state artifact替代。

## 8. V4-2A Closure

状态：`VALIDATED-POLICY-CONTRACT`；只关闭V4-2A，不关闭V4-2。

- `run_alpha_beta_crown_mlp`新增可选beta LR；未设置时继续走原单一Adam group，旧payload/hash不变；
- 显式设置时alpha/beta进入独立Adam parameter groups；正式策略映射为`0.01/0.05`；
- `ProductionMutationPolicyV4`把production iteration=`10`冻结为evaluation/update=`10/9`，并对
  lower-only、fixed intermediate、batch-any stop以及iteration>0 fail closed；
- 非正、NaN、Inf beta LR均拒绝；旧统一LR兼容测试通过；
- focused=`17 passed`；full=`1100 passed, 39 skipped`；mypy 4个相关文件clean；新增/typed policy
  模块Pylint=`10.00/10`；
- 这只证明策略表达与循环基数，不证明任何step tensor或final mutation parity，
  `performance_claimed=false`，B2继续关闭。

下一动作改为V4-2B step-trace schema与capture runner。formal GPU run当前由
`cudaGetDeviceCount error 803`及NVML driver/library mismatch阻塞；schema、replay与CPU负向测试可继续。

### 完成性修正：完整 optimizer controls

V4-2B开工审计确认V4-0 snapshot中的8字段policy只是核心子集；逐step等价还受以下正式默认配置影响：
optimizer=`adam`、lr decay=`0.98`、keep-best=`true`、loss reduction=`sum`、early-stop patience=`10`、
start-save-best=`0.5`、last-iteration fp64=`false`、pruning-in-iteration=`true`、threshold=`0.2`、
alpha/beta enabled、init-alpha、shared-alpha、output constraints、direct optimization、input tightening与
cuts。对固定协议继续沿provider赋值链复核后，beta core的live `init_alpha=false`：α已由pre-state
attach，不应在optimized call内重新初始化；`max_time=60.0 s`，来自alpha-CROWN默认比例`1.0`乘本协议
`bab/timeout=60`后被beta配置继承。此前草案中的`init_alpha=true/max_time=1e9`不是该production call
的live值，已修正，并加入相反配置的fail-closed测试。

因此V4-2A的`VALIDATED-POLICY-CONTRACT`精确解释为“双学习率与loop cardinality子合同”，不是完整
optimizer policy ownership。V4-2B先实现上述controls的live-boundary capture、canonical hash与
missing/tamper fail-closed，再定义step trace；formal run仍等待GPU恢复。

V4-2B controls schema第一切片已实现：18项controls进入canonical payload/hash，live mapping要求字段
全集，replay parser要求exact字段集合与严格bool/list/numeric类型；cuts、output constraints、direct
optimization等当前路线未准入配置fail closed。focused=`10 passed`，mypy clean，typed policy模块
Pylint=`10.00/10`。尚未接入provider step capture，也未生成formal artifact。

## 9. V4-2B Step Trace 与 Capture Runner 实现状态

状态：`IMPLEMENTED-CAPTURE-READY / FORMAL-ARTIFACT-BLOCKED`；不是V4-2B正式关闭，更不是V4-2关闭。

- 新增typed `ProductionOptimizerStepV4/TraceV4`：每个step保存core/call/parent lineage、evaluation与
  observed Adam step ordinal、是否update、当步α/β实际learning rate、24个raw α/SparseBeta tensor及
  lower raw tensor；canonical payload/hash明确包含`performance_claimed=false`；
- 固定workload门禁要求每步`6 alpha + 6 beta value + 6 beta location + 6 beta sign`，mutable为
  alpha/value，location/sign为稳定copy-in；相邻step必须恰有7个mutable tensor改变；
- trace把production LR `0.01/0.05`与scheduler decay `0.98**ordinal`逐步绑定，不再只根据调用数量
  推断更新；最后一次evaluation没有Adam step，前9次必须各观察到一次真实`Adam.step()`；
- production observer只在一个active `update_bounds_core`内、一个outer optimized call下捕获10个
  depth-1 beta calls；同时从live `BoundedModule.bound_opts`捕获18项controls。额外Adam parameter group、
  调用/step错序、缺失policy、提前/重复update均fail closed；作用域退出后恢复原provider methods；
- 新增正式artifact runner，worker固定真实CUDA、三仓commit、模型/property digest、24-call phase tree、
  core policy与trace lineage；replay从raw tensor重建typed trace，并复核manifest/code/file digest；
- CPU测试通过一个真实PyTorch Adam + ExponentialLR的嵌套10-evaluation执行验证observer，确认不是静态
  拼装；raw lower、copy-in、ordinal、LR schedule、mutation count、source identity与parameter group
  tamper均有拒绝路径；
- 定向V4-2/V4-0回归=`31 passed`；首轮及最终文件冻结全量复跑均为
  `1108 passed, 39 skipped`；mypy五文件clean；Pylint五文件=`10.00/10`；
- 当前运行内核`7.1.5`加载NVIDIA module `610.43.03`，而已安装内核`7.1.8`及用户态NVIDIA library
  `610.57.04`；GPU probe因此仍为NVML driver/library mismatch与`cudaGetDeviceCount error 803`。
  formal worker按设计在CUDA admission处失败且不落工件。不得使用CPU synthetic trace代替formal
  production artifact。

下一动作：修复GPU driver/library一致性后，从clean committed source运行formal artifact generation，
再完成original replay与重签名state/step/result/policy tamper probes。只有这些通过，V4-2B才可关闭并进入
V4-2C pre-state native initializer；B2继续关闭。

### V4-2B pre-formal hardening

GPU阻塞期间继续对formal runner做攻击面复审，新增两层不依赖正式数值的fail-closed合同：

- fixed production policy不再只检查字段存在/模式，而是精确冻结iteration=`10`、LR=`0.01/0.05`、
  decay=`0.98`、patience=`10`、start-save=`0.5`、pruning/threshold=`true/0.2`、max-time=`60.0`及
  deterministic等所有已知production值；
- 每个step的24项state metadata与lower receipt必须分别等于独立call-tree的`pre_state`和
  `result[0]` metadata。攻击者即使同步重算tensor/step/trace hash，修改mutable state或lower仍会在
  cross-view binding处失败；call-result device只接受严格CUDA语法。

focused=`26 passed`、扩展RVIR-v4=`47 passed`、全量=`1118 passed, 39 skipped`、mypy clean、
Pylint=`10.00/10`。该hardening只加强formal evidence，不改变V4-2B/V4-2/B2未关闭状态。
