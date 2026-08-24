---
status: internally-validated-no-go-pending-external-audit
updated: 2026-08-25T00:50:00+08:00
type: change
topic: boundflow
slug: fsg4-b4a-formal-timing-internal-closure
stage: s01
---

# FSG4/B4-A 正式计时内部关闭

> **2026-08-25 历史措辞澄清**：本文“validated mechanism/reduced evidence”不表示performance
> reduced；外审后的最终且唯一性能分类是
> `EXTERNALLY-APPROVED-VALIDATED-NO-GO-B4-A-PERFORMANCE`。机制/正确性可保留，性能候选不可累计。

## 1. 判定

source=`46a8493` 的 v5 正式 artifact 完成全部冻结门禁，内部分类为：

`INTERNALLY-VALIDATED-NO-GO-B4-A-PERFORMANCE-PENDING-EXTERNAL-AUDIT`

B4-A terminal lower/lA handoff 的 correctness 与机制证据保留；因 core wall geomean 未达到预注册
`1.03x`，不得累计为 B4 performance candidate。外审前始终保持 `performance_claimed=false`。

## 2. Artifact 身份

- 路径：`artifacts/fsg4-b4a-formal-timing/resnet2b-prop0-v5`；
- source：`46a8493557c49f327df4e70d7cdd7649227b14b9`；
- manifest hash：`cf0d179c...b6a`；
- summary hash：`46360e41...3d7`；
- 24/24 fresh worker，6 control pair + 12 profile worker；
- `nvidia_powerd=inactive`、`enforced.power.limit=55.0 W`、strict cool-idle preflight逐worker绑定；
- artifact 与 tamper report 均无 `/home/`、`/tmp/` 本机路径泄漏；正式采样后已恢复
  `nvidia-powerd.service=active`。

## 3. Correctness、activation 与环境

- 6/6 pair：semantic failures=`0`，19 tensor/pair，sign全部exact；
- terminal export 全局 max abs diff=`4.410743713378906e-06 <= 2e-4`；
- B3 handoff/rerun=`0/1`，B4-A=`1/0`且lineage=`6`；
- profile结构保持forward=`4`、optimizer bound evaluation=`10`、optimizer
  trace/evaluation/update=`1/10/9`，provider/fallback=`0/0`；
- 24/24 environment admitted，无external compute process、worker overlap或independent thermal slowdown；
- profile closure/residual最大均为`0.0018394918070328713`，低于`1%/3%`门禁。

## 4. 性能与归因

六组 control raw 独立重算：

- core wall geomean=`1.0189949992169265x`，pair范围
  `[1.0001952774637546, 1.0378662482148024]`，**未过**`1.03x`；
- core GPU geomean=`1.0189919319887064x`；
- query wall geomean=`1.0022597825638593x`，worst pair=`0.996947022444439x`，**通过**`0.98x`；
- query GPU geomean=`1.0022597197712242x`；
- peak allocated/reserved全部pair ratio=`1.0`，无显存收益。

profile attribution只作解释、不参与headline：terminal backward/export mean wall 从 B3 `11.811955 ms`
降到 B4-A `1.864271 ms`，局部约`6.335964x`、节省约`9.947684 ms`；但 optimizer、KFSB、commit等
非目标span发生小幅反向波动，最终control core只形成约`1.9%`稳定收益。不得据此调低冻结阈值。

## 5. Replay、tamper 与边界

- root replay逐字重建同一summary与hash；
- 14/14 outer-resigned攻击全部拒绝，包括latency、semantic payload、activation/profile counter、
  runtime identity、environment counter delta、preflight、power policy、protocol、pair与summary；
- manifest内全部文件SHA256独立复核一致；
- kernel/launch差分仍为`DEFERRED-TO-B4-A-KERNEL-DELTA`，不影响已冻结的core/query判定，但不可用于
  kernel-level claim。

## 6. 下一步

下一唯一动作是外部模型从formal raw独立审计本轮NO-GO关闭。在外审批准前B4-B/TIR保持关闭。若外审
批准，B4-A只能作为validated mechanism/reduced evidence保留；是否启动B4-B必须依据B4-0已冻结的
67.72% opportunity与B4总路线单独决定，不能把B4-A计入累计性能基线，也不能改阈值后重跑。

## 7. 验证

- 固定 11 文件 related（含正式 artifact replay）：`73 passed`；
- 全量：`1356 passed, 3 skipped`（6 个既有 warning）；
- Black/Mypy/Pylint：触达代码与测试 clean，Pylint=`10.00/10`；
- `git diff --check`：PASS；
- root replay：exit 0，summary逐字段一致；
- outer-resigned tamper：`14/14 rejected`；
- manifest文件digest：0 mismatch；本机路径扫描：0 hit。
