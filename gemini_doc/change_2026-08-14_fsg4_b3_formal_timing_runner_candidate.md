# FSG4/B3 36-Process 正式计时 Runner 实现候选

日期：2026-08-14

状态：`IMPLEMENTED-PENDING-CLEAN-SOURCE-FORMAL-RUN`

## 结论

FSG4/B3 已实现冻结的 B0/B2/B3 六全排列、control/profile 共 36 个独立进程的正式计时 runner、
raw-first/resumable artifact generator、独立 replay 和 outer-resigned tamper probe。实现与合同测试已经
通过，但正式 36-process artifact 尚未运行，因此本记录不包含 timing/speedup claim，也不改变 B4—B7
关闭状态。

## 实现范围

- 新增 `boundflow.fsg4-b3-same-solver-timing/v1` typed run、activation receipt、聚合和分类合同；
- 冻结六个 B0/B2/B3 全排列，每个 block 对每个配置各运行一个 control 和一个 profile；
- B0 绑定原始 provider，B2 绑定 whole-call reference replacement，B3 绑定
  `b3_ir_graph_plan_schedule` cumulative replacement；
- B3 每个 worker 必须提供 prepared template、PlanInstance、terminal Schedule、assembly、atomic commit
  和 post-query device audit 的直接 activation receipt；只改标签而没有物理激活会 fail closed；
- control worker 只启用原有计时观察器，不保留详细 physical counter，避免 event journal 污染 headline
  latency；B2/B3 profile worker使用不保留 event 的轻量直接计数器，并受 profile/control 扰动
  `<=1.05`硬门禁；
- generator 先落 root protocol 和每个 raw worker envelope/log，再派生 paired runs、profile spans、
  activation receipts、closure、summary 与 manifest；中断后只复用完整且 source-bound 的 worker；
- 正式 manifest 仅在 correctness、environment、measurement、profile closure 和 activation 全部准入时
  生成；失败尝试只写 `failed_summary.json`，不生成可被误认成正式结果的 manifest；
- protocol 绑定 five-fresh admission manifest、源码 revision、代码 digest、benchmark/input digest、外部
  仓库 revision、解释器/runtime/GPU identity 和 36-run 固定顺序；
- worker command projection、stdout/stderr 与 runtime executable/process 名称在 artifact 中去除本机绝对
  路径，不提交机器相关路径；
- replay 不采信 summary ratio，而是从 36 个 raw worker 重新解析 typed run、重算直接语义、环境、
  activation、profile closure、全部 ratio 和最终分类。

## 篡改门禁

已实现十类 outer-resigned attack inventory；攻击修改 payload 后同步重签 manifest，仍必须被语义重算
拒绝：

1. control latency；
2. raw worker 删除；
3. aggregate order；
4. B3 activation receipt；
5. B3 profile physical counter；
6. B3 provider/fallback；
7. B3 semantic result；
8. formal preflight；
9. protocol sequence；
10. summary ratio/decision。

这些 probe 目前只完成代码与静态 inventory 测试；必须等正式 artifact 生成后实际运行，不能提前写成
“10/10 rejected”。

## 非 Claim 冒烟检查

在独立 GPU 进程中分别执行了 B3 control、B3 profile、B2 profile 与 B0 control：

- B3 control/profile 均由 prepared executor 真实激活，provider core/compute/update/fallback 全零；
- B3 profile 的固定 physical counters 满足 template=`1/1`、module move=`0`、scope=`1`、optimizer=
  `10/9`、snapshots=`0`、forward=`4`、KFSB=`3/3`、candidate D2H=`0`、commit=`12`；
- B2 profile 保持 whole-call reference 结构，snapshots/forward/D2H=`10/5/12`；
- B0 control 保持 original provider 路径；
- 四个结果均未泄漏本机绝对路径。

这些执行只验证 runner 能启动真实 solver 路径及 receipt/counter 结构；它们没有按冻结顺序、重复数、
热状态和配对协议运行，任何单次 wall time 都不得形成性能结论。

## 验证

- 定向合同：`108 passed in 7.63s`；
- 全量回归：`1308 passed, 3 skipped, 6 warnings in 468.95s`；
- Black：8 个 touched 文件 clean；
- mypy：6 个 source files clean；
- Pylint：`10.00/10`；
- `git diff --check`：PASS。

## Claim 边界与下一步

当前只证明正式 runner、schema、raw-first/replay/tamper 机制已实现并通过静态/合同回归。下一唯一动作是
把该实现提交为 clean source，随后从 position 0 运行完整 36-process artifact，再执行 root replay 和
十类 tamper probe。只有正式 artifact 全部门禁通过后，才可按预注册阈值给出
`VALIDATED-B3`、`VALIDATED-REDUCED-B3`、`VALIDATED-NO-GO-B3`或 correctness blocker；B4—B7 继续关闭。
