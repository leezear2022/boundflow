---
status: active
updated: 2026-08-14T12:30:00Z
type: change
topic: boundflow
slug: fsg4-b3c-device-atomic-commit-candidate
stage: s01
---

# FSG4/B3-C Device-Resident Atomic Commit 实现候选

> 本文记录实现候选历史状态，已被
> `gemini_doc/change_2026-08-14_fsg4_b3c_device_atomic_commit_closure.md`正式关闭结论取代。

## 结论

B3-C第一版实现候选已经完成本地CUDA单元与集成验证，状态为
`IMPLEMENTED-PENDING-FRESH-GPU-ARTIFACT`。它尚未获得正式counter artifact，因此不能升级为
`VALIDATED-B3-C-COUNTERS`，也不产生timing或speedup claim。

## 本轮改动

- 新增first-class `DeviceAtomicCommitPlanV1`，在prepared template阶段冻结12条mutable path的role、
  shape、dtype、CUDA device、alias equivalence class和rollback ordinal；
- 新增动态`DeviceAtomicTransactionV1`，绑定`CorePlanInstanceV1`、pre snapshot、12个live target
  version、host packet version和GPU candidate；
- α sparse projection与β location projection直接在CUDA设备上生成private candidate，不创建
  `ProductionStateSnapshotV4` CPU candidate；
- commit前检查完整inventory、shape/dtype/device、alias、finite、tensor version和host version；先生成
  12份device backup，再执行12次同设备`copy_`；任一tensor或host写入失败均恢复12个tensor与host
  pre-image；
- 对五个合法`(6, 0)` SparseBeta目标使用empty-object identity判断alias，避免因这些张量共同
  `data_ptr()==0`而误判；
- 新增hash-free headline provider assembly；headline metadata只记录plan/transaction/placement，不做
  GPU content SHA；
- query CUDA event与wall timer结束并同步后，`finalize_post_query_audit()`才生成candidate/committed
  SHA256，并与plan hash、transaction version和commit hash交叉绑定；
- B2、B3-A、B3-B仍走原stage/commit路径；B3-C只有显式prepared template + terminal schedule +
  device plan组合才启用。

## 预注册物理目标

- candidate materialization=`12`且全部位于`cuda:0`；
- timed candidate D2H=`0`；
- committed mutable paths/device backups/direct commit copies=`12/12/12`；
- full optimizer snapshots=`0`，forward builds=`4`，optimizer=`10/9`，KFSB=`3/3`保持B3-B；
- provider core/compute/update与fallback保持`0/0/0/0`；
- post-query audit耗时必须单列，不能计入query/core headline timing。

## 已完成验证

- B3-C CUDA事务与provider assembly：`10 passed`；
- B2/B3-A/B及B3-C相关定向回归合计：`50 passed`；
- 覆盖正向commit/audit、NaN、stale version、empty-beta alias、mid-copy failure、host failure、
  terminal-lower drift和outer-resigned receipt tamper；
- Black clean；mypy四个touched source clean；Pylint `10.00/10`。

## 尚未成立

- 尚未从提交后的clean source生成fresh真实ResNet2B/prop0 B3-C artifact；
- 尚未独立重放event journal、冻结B2语义锚点和outer-resigned artifact tamper；
- 尚未证明真实worker的candidate D2H确为`0`；
- 尚未完成全量测试；
- 尚未完成正式B3计时前要求的5个fresh B2/B3 correctness pairs；
- 没有延迟、吞吐、显存或端到端speedup claim。

## 下一步

1. 提交本候选源码，保证artifact generator的source-clean gate成立；
2. 从该commit启动一次fresh B3-C真实GPU counter artifact；
3. 若12→0 D2H、冻结语义、replay和tamper全部通过，再做全量回归并关闭B3-C；
4. B3-C关闭后才开始5 fresh B2/B3 correctness pairs；B4—B7继续关闭。
