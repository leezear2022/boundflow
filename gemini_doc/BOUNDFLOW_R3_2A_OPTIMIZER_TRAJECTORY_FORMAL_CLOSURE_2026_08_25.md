---
status: validated-r3-2a-p-trajectory-r3-2b-open
updated: 2026-08-25T05:45:00+08:00
type: changelog
topic: boundflow
slug: r3-2a-optimizer-trajectory-formal-closure
stage: s01
---

# R3-2A P-anchor 10/9 Optimizer Trajectory 正式关闭

## 1. Verdict

R3-2A 以 `VALIDATED-R3-2A-P-TRAJECTORY` 关闭。formal source=
`e7ae590e8b27882b2a5d8837993f53e652791eb8`；5对/10个fresh subprocess、每个worker 10次
evaluation/9次Adam+LR scheduler mutation、semantic replay和12/12 fully re-signed tamper全部通过。

因此只开放R3-2B：同一P-anchor、同一10/9轨迹的wrapper-inclusive本地物理计时。S-anchor、active β、
multi-site、production default、same-solver仍关闭。

本阶段没有读取latency，`timing_recorded=false`、`performance_claimed=false`。

## 2. 正式artifact与数值

artifact：`artifacts/r3-structured-owner/r3-2a-optimizer-trajectory-v1/`

- pair order=`NC/CN/NC/CN/NC`，pair/worker=`5/10`；
- evaluation/mutation/scheduler mutation=`10/9/9`；
- 最大逐步 lower max abs diff=`8.58306884765625e-06`；
- 最大逐步 compressed dα max abs diff=`8.288770914077759e-08`；
- 最大逐步 α-after max abs diff=`2.384185791015625e-07`；
- 最大 Adam `exp_avg/exp_avg_sq` diff=
  `4.190951585769653e-08 / 1.0459189070388675e-11`；
- lower/dα 50个evaluation均allclose且sign exact；α与optimizer state均在预注册`2e-5`内；
- native absolute peak allocated/reserved=
  `20,692,480 / 25,165,824 B`（5/5一致）；
- candidate absolute peak allocated/reserved=
  `1,214,464 / 4,194,304 B`（5/5一致）；
- worst allocated/reserved ratio=
  `0.05869108004453792 / 0.16666666666666666`；
- candidate逐步custom forward/backward=`10/10`，compiled receipt均saved dense A=`0`、scratch=`2`、
  warm dynamic allocation=`0`、fallback/eager/native-shadow=`0/0/0`；
- immutable α/β、empty P-β、split/history、bounds、parameter/input/objective identity逐步不变；
- dynamic rebind每步只更新P-α content identity，plan/trace/ordinal/lr完整绑定。

memory claim只限定这一冻结P-anchor 10/9 wrapper和PyTorch allocator口径，不外推whole core/query。

## 3. Replay与篡改

manifest=`23c0b685dcc8299be95324946e7e286deee04f21e436b85a8a634adef5f255d4`；
protocol=`92f1abc7757607602bffdceb75bece3fcfc512e6ace54aab8e9bc07f8e93b830`；
summary=`4ebc82b62b25acea9489d178d4fc84ed635e34d7106918df419e6f50fe3aa085`。

replay逐raw重验tensor digest、10/9 lineage、lower/dα/α/moments、rebind、compiled ownership、memory、
protocol语义、code revision和exact文件inventory，再重算summary。

12类全重签攻击全部拒绝：中间lower、dα、α-after、Adam exp_avg、saved dense A、fallback、evaluation
ordinal、terminal update、immutable identity、memory peak、summary admission和protocol tolerance。

## 4. 失败与修正记录

- 两次formal启动在任何worker前fail closed：先后暴露untracked用户PDF不应进入source cleanliness，
  以及DocOps porcelain首列被`.strip()`误读；均独立修复并从新commit启动；
- 第一轮5-pair diagnostic artifact虽执行通过，但审查发现protocol仅hash校验、缺逐字段语义冻结；该
  副本已移动到`/tmp/r3-2a-optimizer-trajectory-pre-protocol-hardening-48a8aca`，没有复用raw；
- 当前formal artifact从hardening source重新运行全部10个worker。

## 5. 验证链

- formal replay=`PASS`；
- fully re-signed tamper=`12/12 rejected`；
- R3 targeted=`50 passed`；
- mypy=`clean`，pylint=`10.00/10`；
- full regression=`1602 passed, 3 skipped, 6 warnings in 662.14s`；
- 3个skip仍为既有TVM重复编译/VNN-COMP checkout环境边界。

## 6. 唯一下一动作

预注册并执行R3-2B wrapper-inclusive local timing：native/candidate必须复用本阶段相同P-anchor、初始state、
10/9 schedule、optimizer、stream policy和correctness关闭路径；compile/preparation在计时外，完整10次
forward+backward、9次Adam/scheduler/clamp必须在计时内。GO=`geomean>=1.20x`、worst pair>=`0.98x`、
correctness保持、memory<=`1.0x`。否则当前single-site variant正式kill，不能靠扩site掩盖。

