# BoundFlow ASPLOS’27 S0 explicit transaction 与10×预算修改记录

date: 2026-08-27
status: internally-validated-s0-attribution-closed-s1-implementation-open
external-audit: deferred-by-user
performance-claimed: false

## 1. 本轮完成内容

本轮完成S0第二批，不修改αβ-CROWN、auto_LiRPA、TVM或production BoundFlow执行逻辑：

1. 新增可恢复的solver transaction observer，区分coarse scope与exact transaction；
2. 绑定固定外部commit与11个源码blob上的33个真实call-site target；
3. 新增GPU alternating control/profile runner、semantic replay、hash chain与fully re-signed tamper拒绝；
4. 运行2 workloads × 5 fresh pair formal，关闭97%机制覆盖和5% observer扰动门禁；
5. 以formal raw pooled share生成互斥O1—O5 10×研究预算；
6. 只开放S1 canonical compiler path实现，不开放性能门禁或论文性能claim。

## 2. 代码与artifact

- `boundflow/runtime/solver_transaction_observer.py`
- `scripts/run_asplos27_s0_transaction_markers.py`
- `tests/test_solver_transaction_observer.py`
- `tests/test_asplos27_s0_transaction_artifact.py`
- `boundflow/runtime/asplos27_transaction_budget.py`
- `scripts/run_asplos27_s0_transaction_budget_artifact.py`
- `tests/test_asplos27_transaction_budget.py`
- `tests/test_asplos27_s0_transaction_budget_artifact.py`
- `artifacts/asplos27-s0-transactions/official-b0-five-pair-v1/`
- `artifacts/asplos27-s0-transaction-budget/official-b0-five-pair-v1/`

## 3. Formal结果

| workload | min coverage | max unresolved | median profile/control | max profile/control | conditional projection(h=0) |
|---|---:|---:|---:|---:|---:|
| ResNet2B | 99.632394% | 0.367606% | 0.9959016× | 1.0416400× | 12.562203× |
| MNISTFC | 99.248363% | 0.751637% | 0.9986577× | 1.0653545× | 11.656612× |

10/10 pair均保持result/source/protocol/environment exact；两workload compute signature five-fresh exact；两个
artifact均replay通过。worker semantic summary、target protocol、budget projection、axis target四类全重签篡改
均被拒绝。

## 4. 预算解释

formal pooled baseline share为：

- ResNet：O1 `67.9625%`、O2 `20.7179%`、O3 `7.2578%`、O4 `3.7300%`、unresolved `0.3316%`；
- MNISTFC：O1 `20.2428%`、O3 `78.9729%`、O4 `0.0527%`、unresolved `0.7294%`。

达到10×所需resolved全栈平均speedup分别为`10.3087×/10.7081×`。冻结的O1—O5目标组合只在
integration overhead `h=0`时给出大于10×的条件式数学投影；接入成本尚未计入，任意`h>0`都会降低结果。
目标全部未验证，尤其O1 `16×`只是高于现有local anchor的stretch target。投影不是已实现性能，不能进入
abstract、headline或对外宣传。

## 5. 验证与下一步

已完成：

- 原S0/10×专项：`28 passed`；外审修复后增加显式`h`测试，现为`29 passed`；
- 全量：`1860 passed, 3 skipped, 6 warnings`，skip为既有TVM/VNN-COMP环境边界；
- Black：12个本批Python文件unchanged；
- mypy：6个source文件clean；
- pylint：3个runtime模块和3个runner分别`10.00/10`；合并运行只触发artifact helper的跨文件重复提示，
  无单文件质量错误；
- explicit transaction artifact replay：summary hash=
  `293e31c1db697a701660dbc4e6f8f85671086f0cd556b3c30384477f6a6c1435`；
- transaction budget artifact replay：summary hash=
  `880f89cd2e765a2c519e898d5944851165b9c294a3c904666b7d31bb5317d0a7`；
- `git diff --check`通过；DocOps change=`ev015988`、validation=`ev015989`，`dol lint --soft` PASS。

2026-08-28外审补充：报告`external_audit_asplos27_s0_2026_08_27.md`给出approve-with-minor；两条minor
分别通过显式`h=0`字段/公式和per-workload最大单对扰动披露关闭。派生刷新前后
`worker_runs.jsonl` SHA256均为`90df685e…1bc0`，未改GPU raw。mypy clean仅指本批文件；全量pytest须先
激活`boundflow`环境并加载`env.sh`。

下一唯一实现是S1：已有CIBC winner经canonical
`Primal→Bound→Plan→Relax/TIR→Prepared Runtime`路径执行，correctness、fallback=0后才做direct
CIBC/pipeline/PyTorch三方计时；O1/O3未直测通过前性能门禁保持关闭。
