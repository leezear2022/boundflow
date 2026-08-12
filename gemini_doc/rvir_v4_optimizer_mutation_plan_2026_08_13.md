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
