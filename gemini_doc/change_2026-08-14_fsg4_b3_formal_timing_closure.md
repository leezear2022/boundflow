# FSG4/B3 36-Process 正式计时关闭记录

日期：2026-08-14

状态：`VALIDATED-REDUCED-B3`（外部审计待完成）

## 结论

source `36e9069ca4f21183c9b36d74024de0ca8b20f59c`按预注册协议完成六个 B0/B2/B3 全排列、
每配置 6 个 control 和 6 个 profile，共 36 个独立 fresh GPU worker。correctness、environment、
measurement、activation、profile closure、root replay 和十类 outer-resigned tamper 门禁全部通过。

B3 相对 B2 的 core wall 几何平均加速为 `1.071617x`，超过 reduced 门槛 `1.05x`但未达到完整 B3
门槛`1.15x`；B3 query 相对 B2 为`1.006623x`，没有退化。然而 B0/B3 query 几何平均仅
`0.910001x`，即 B3 完整 query 仍约比原始 B0 慢`9.89%`。因此正式分类只能是
`VALIDATED-REDUCED-B3`，不能升级为`VALIDATED-B3`或 BoundFlow 全栈性能主张。

## 正式证据

- artifact：`artifacts/fsg4-b3-same-solver-timing/resnet2b-prop0-v1/`；
- tamper：`artifacts/fsg4-b3-same-solver-timing/resnet2b-prop0-v1-tamper-report.json`；
- source：`36e9069ca4f21183c9b36d74024de0ca8b20f59c`；
- protocol/manifest/summary internal hash：
  `8193010ee200d11ad12ea68c01462c5c1a5854078fa58d895ed745e12c3d642b`/
  `d553a72ea5cae99c367f7c0120966bb36058966bc081241befba24455be2c1c9`/
  `4c19afd43c18e0409932b86506efdaf6bfc3e07baabcc222dbe79c8149f99bac`；
- protocol/manifest/summary file SHA256：
  `cecec7c262040a2953d1e0ec7e1479942d648b39f12f6edaa8787efbca6e79e2`/
  `d88eeecafcd6a7a9394cdf9654962a36497b1c7afd15d7862048b1c3ccd7db4a`/
  `c8666213479328d8406172dd56d70f015e96e6fa9b2f937c90a5ec07fedb2ff0`；
- tamper internal/file hash：
  `b89ada48f40c5766bc0b93c1542d0a5aa7cc741fe250f9cb9efc5c38a2cae799`/
  `bd392e5ca49912e376c4090e7e929077f804b6bbd5d11a250d0790b83847e21b`。

## Correctness、Activation 与环境

- 36/36 worker完成，B0/B2/B3各`6 control + 6 profile`；固定顺序未按结果调整；
- 36/36 environment admitted，runtime identity count=`1`；
- 18 个 control/profile closure row全部通过，最大 closure error/residual share均为
  `0.0025104990`，低于`0.01/0.03`门槛；
- profile/control query扰动全部通过`<=1.05`：B0/B2/B3几何平均分别为
  `1.009256/1.008405/0.994097`，最大值分别为`1.026924/1.043622/1.002677`；
- B0保持 original provider；B2/B3 provider core/compute/update/fallback持续为零；
- 12/12 B3 worker均直接证明 prepared template、PlanInstance、terminal Schedule、assembly、commit和
  post-query audit各一次；headline digest和candidate D2H均为零；
- control不保留详细physical counter；6个B2与6个B3 profile的轻量counter分别保持冻结结构：B2
  snapshots/forward/D2H=`10/5/12`，B3=`0/4/0`，optimizer=`10/9`；
- artifact共157个文件、约2.3 MB，全部路径投影不含本机`/home/`绝对路径。

## 正式性能分类

全部 ratio 采用相同 block 的 control-only pair，并从 raw worker重算：

| 对比（numerator/candidate） | 指标 | Geomean | 结果 |
|---|---|---:|---|
| B2/B3 | core wall | `1.071617x` | 通过 reduced `1.05x`，未过 full `1.15x` |
| B2/B3 | query wall | `1.006623x` | B3不退化 |
| B0/B3 | query wall | `0.910001x` | B3仍慢于B0，未过`1.00x` |
| B0/B3 | core wall | `0.535965x` | B3 core仍显著慢于B0 |
| B0/B3 | peak allocated | `0.998647x` | 无显存收益；B3略高 |
| B0/B3 | peak reserved | `1.000000x` | 无变化 |

六个 B2/B3 core pair全部大于 1，范围`1.063588x—1.090314x`；最差 pair也未触发 5%退化门禁。
六个 B0/B3 query pair则全部小于 1，范围`0.892193x—0.917578x`，所以 B0 parity未成立。

Profile归因显示：B2→B3后 atomic commit wall几何平均从约`73.295 ms`降至`22.476 ms`，但
typed pre-state从约`24.412 ms`增至`58.284 ms`；B3 core share主要为 optimizer `44.73%`、typed
pre-state `23.15%`、KFSB `18.24%`、atomic commit `8.93%`。这解释了结构复用只转化为约`1.07x`
core收益，也给后续 B4 的目标排序提供证据；它不授权在本阶段混入 TIR/JIT/runtime/arena。

## Replay、Tamper 与回归

- 独立 root replay从36个raw run重建配对、activation、closure、全部 ratio和最终分类，输出与
  generator逐字段一致；
- 十类攻击在修改payload并同步重签manifest file digest与manifest hash后仍`10/10 rejected`：control
  latency、worker删除、aggregate order、B3 activation、B3 profile counter、fallback、semantic、formal
  preflight、protocol sequence、summary ratio；
- frozen artifact tests：`6 passed`；
- FSG3/FSG4-B3定向：`114 passed in 7.83s`；
- full：`1314 passed, 3 skipped, 6 warnings in 449.04s`；
- Black clean，Pylint=`10.00/10`；source、tests、docs与JSON/JSONL的`git diff --check`通过。36个
  manifest-bound raw solver stdout/stderr原样保留αβ-CROWN输出中的行尾空格，检查时显式排除这些不可变
  raw logs，未为满足格式检查而改写正式证据。

## Claim 边界与下一步

该结果只覆盖 RTX 4060 Laptop GPU、冻结 ResNet2B property 0、单次固定 solver prefix。它证明 B3
IR/Graph/Plan/Schedule累计机制相对 B2 有可重复的 reduced core收益，但没有回到 B0 parity，也没有
证明 complete-query、TTV、solved verdict、多模型或最终 B7门槛。artifact持续写
`performance_claimed=false`，不得包装成 ASPLOS headline speedup。

下一唯一动作是外部模型按独立 raw 重算方式审计本 closure。外审通过后，B3可作为后续 B4
operator/cross-stage fusion的累计候选；不得把 B3 的`1.071617x`外推为 B4—B7收益，也不得跳过
same-solver B0累计对照。
