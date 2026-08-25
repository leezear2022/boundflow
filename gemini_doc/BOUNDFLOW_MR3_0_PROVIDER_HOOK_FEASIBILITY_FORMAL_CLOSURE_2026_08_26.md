# BoundFlow MR3-0 Provider Hook Feasibility 正式关闭

> 日期：2026-08-26  
> 结论：`VALIDATED-MR3-0-PROVIDER-HOOK-FEASIBILITY`  
> source：`8a63503c6f98357d68066be8ae2ea7a256b65b8f`  
> 性能声明：`performance_claimed=false`

## 1. 结论

真实 αβ-CROWN 的 P-anchor production hook 存在且可精确绑定。MR3-0 以两个 pair、四个 fresh GPU
进程证明：在 `activation_bab_bound / CROWN-optimized` 外层 exact call 中，只读 pass-through hook
可以稳定绑定 start node `/49` 下的 `/input-24` ReLU 与 `/input-20` Conv，且不改变 provider
result、target α 或完整 alpha/beta module state。

本轮只关闭 hook feasibility，不是 candidate bridge correctness。下一步只开放 MR3 fail-closed
candidate bridge implementation；timing、multi-site、S-anchor 与 same-solver performance 继续关闭。

## 2. Formal protocol 与结果

- 顺序=`CP/PC`，2 pair / 4 独立进程；
- 每进程 outer beta optimized exact call=`1`、inner backward evaluation=`10`；
- 两个 probe 合计 P ReLU/Conv original call=`20/20`；
- solver 四侧均为 `verified`、success=`true`、visited domains=`[6]`；
- 每 pair 比较 outer result、10 次 inner result、target α 与完整 module state，各 `9600` 个元素；
- pair-0/pair-1 max abs diff=`2.0265579223632812e-06 / 1.6093254089355469e-06`；
- allclose/sign exact=`true/true`，远低于冻结 `2e-4` 容差；
- P β 在 20 次 evaluation 中均为一个 `[6,0]` provider empty tensor、总 `numel=0`；
- ReLU→Conv handoff 20/20 content exact，pointer 0/20 复用，确认 provider coefficient map 会换 storage；
- CUDA device/stream 前后不漂移；replacement/fallback/eager/native-shadow=`0/0/0/0`。

## 3. Replay 与 tamper

- formal artifact：
  `artifacts/measurement-recovery/mr3-0-provider-hook-feasibility-v1`；
- summary hash=`c19dd2466243b3ddaa43d4313a6b2c3bd1c04f769622b9ac3c247a19f0eee785`；
- manifest hash=`5610cb83a57da3e63555ecd5151d440642edf1c0794e789018ad77765ce7eb88b`；
- semantic replay=`replay-passed`；
- 12 类 fully re-signed attack=`12/12 rejected`；
- tamper report 是 manifest 必需文件，replay 独立验证其 hash、count 与逐项 rejection ledger。

回归与静态验证：MR0–MR3 targeted=`24 passed`；全量=`1703 passed,3 skipped`；Black clean、
mypy clean、pylint=`10.00/10`、`git diff --check`与DocOps lint通过。3个skip均为既有环境/重复编译
边界，不是MR3-0回归。

攻击覆盖 source/order/call count、empty β object/numel、ReLU→Conv content、α shape、CUDA stream、
outer numeric result、target α numeric state、replacement count 与 missing hook。数值攻击同步重签
tensor state hash、worker hash、raw digest 与 manifest hash，仍由 semantic replay 拒绝。

## 4. 所有权结论

MR3 candidate 必须遵守以下真实 provider 边界：

1. exact call 是 beta-split 的 10-evaluation optimized call，不是 5-step alpha 初始化；
2. P β 的合法 absent 表示是 provider-owned `[6,0]` empty tensor，不能按“对象不存在”处理；
3. ReLU→Conv 只保证语义内容连续，不保证 storage pointer 连续；
4. candidate replacement 必须在 Conv handoff 重新绑定输出，不能假设可零拷贝接管 ReLU storage；
5. provider 继续拥有其余 site、optimizer、split/history、termination 与 final commit。

## 5. 唯一下一步

实现 MR3 P-anchor fail-closed bridge：candidate 只替换 `/49` 下该单 site 的 forward/custom VJP，
先补 synthetic negative/atomic rollback，再执行原预注册的 5 pair/10 fresh correctness artifact。
在 MR3 正式通过前不得开始 timing。
