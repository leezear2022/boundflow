# 2026-08-14 — FSG3 B0/B1/B2 Same-Solver 正式基线关闭

## 目标与判定

FSG3只建立同一official αβ-CROWN solver、同一ResNet2B property 0、同一RTX 4060 Laptop GPU下的
B0/B1/B2全栈分母：

- B0：original `update_bounds_core`；
- B1：RVIR typed transport后仍调用original provider；
- B2：BoundFlow whole-call reference replacement，禁止provider callback与fallback。

本阶段不要求B2更快，也不把B2结果外推为BoundFlow全栈上限。正式状态为
`VALIDATED-FSG3-B0-B1-B2-BASELINE`；B2性能分类为`MEASURED-B2-SLOWER`。FSG4的B3—B7现可按独立
feature gate启动，最终系统门槛仍只施加到累计B7 vs B0。

## 正式工件

- Draft PR：`#60`（base=`main`，head=`feat/rvir-v4-production-state-ownership-v1`）；
- artifact：`artifacts/fsg3-same-solver-timing/resnet2b-prop0-v5/`；
- tamper report：
  `artifacts/fsg3-same-solver-timing/resnet2b-prop0-v5-tamper-report.json`；
- schema：`boundflow.fsg3-same-solver-artifact/v4`；
- source commit：`a4ee2910f4039981338fb6d8688ac4af18508b73`；
- summary hash：`df852590d99be09962c1287e7166b421edb260416403a3c91545dca6e2e1318e`；
- manifest hash：`9089e2019eb5e98cac228151cb061c0f6aceefa0ad6c6b3e298584bcede21e85`。

六个配置全排列block均按control/profile相邻顺序完成，共36个fresh GPU进程；每个配置均为6个
control和6个profile。`correctness_passed=true`、`environment_passed=true`、
`measurement_auditable=true`、`failure_rows=[]`，18个profile span的closure/residual全部过
`<=1%/<=3%`门禁。36/36 worker均使用同一runtime identity并通过正式环境准入。

## 物理执行与正确性

每种配置的12个run均满足冻结counter：

| 配置 | typed validation | provider core / compute / update | fallback | 物理路径 |
|---|---:|---:|---:|---|
| B0 | 0 | `1/14/3` | 0 | original |
| B1 | 1 | `1/14/3` | 0 | RVIR typed passthrough |
| B2 | 1 | `0/0/0` | 0 | BoundFlow whole-call reference |

所有run均保持冻结lower-only sentinel、state、branch、queue与termination语义；B2不是post-hoc IR，也
没有暗中回调original provider。

## 正式计时结果

所有speedup均定义为`B0/candidate`，大于1才表示candidate更快。headline仅使用control；profile只做
归因。

| 指标 | B1 geomean | B1 median | B2 geomean | B2 median |
|---|---:|---:|---:|---:|
| query wall | `0.995657x` | `0.995472x` | `0.908400x` | `0.905937x` |
| core wall | `0.968882x` | `0.979967x` | `0.516767x` | `0.516463x` |
| query GPU | `0.995654x` | `0.995470x` | `0.908395x` | `0.905938x` |
| peak allocated | `1.000000x` | `1.000000x` | `1.000000x` | `1.000000x` |
| peak reserved | `1.000000x` | `1.000000x` | `1.000000x` | `1.000000x` |

因此B1 whole-query约慢`0.44%`；当前未优化B2 whole-query约慢`10.1%`，whole core约为B0的
`1/0.516767 = 1.935x`耗时。B2 compile break-even为`not_reachable`，不是因为compile成本大，而是warm
query本身已经更慢。显存没有改善。

profile/control wall geomean分别为B0=`1.002178`、B1=`1.003107`、B2=`1.001605`，最大值分别为
`1.010248/1.007812/1.012552`，全部低于`1.05`，headline计时可审计。

## B2瓶颈地图

B2 profile core的geometric-mean wall share为：

| 区域 | share |
|---|---:|
| optimizer | `43.9993%` |
| atomic commit | `24.6841%` |
| KFSB | `16.6844%` |
| typed pre-state | `10.7197%` |
| backward | `3.6775%` |

compile只占cold total约`0.1319%`，official post/queue只占query约`0.1082%`。因此FSG4不能再把
selected-CROWN或单个backward kernel当作唯一优化对象；B3首先需要消除optimizer、atomic copy-out、
KFSB和pre-state之间的图/状态/materialization重复，并让Bound/Graph IR与Plan/Schedule真实驱动执行。

## Replay 与篡改门禁

静态replay从raw worker、metadata、profile spans重新检查顺序、环境、物理counter、语义、closure、
扰动和paired statistics。新增8类outer-resigned攻击：攻击者修改payload后同步更新manifest文件digest与
manifest hash，仍必须被语义重算拒绝。覆盖控制时延、删run、配置/模式顺序、B1 provider count、B2
fallback、semantic tensor、温度准入和summary ratio；8/8均被拒绝。

这里的“outer-resigned”只指外层文件digest与manifest hash同步更新，不冒充所有内部语义副本也已同步
重签。

复现命令：

```bash
conda activate boundflow
source env.sh
python scripts/run_fsg3_same_solver_experiment.py replay \
  --artifact-dir artifacts/fsg3-same-solver-timing/resnet2b-prop0-v5
python scripts/probe_fsg3_same_solver_artifact_tamper.py \
  --artifact-dir artifacts/fsg3-same-solver-timing/resnet2b-prop0-v5 \
  --report /tmp/fsg3-tamper-report.json
```

## 边界与下一门禁

- `performance_claimed=false`保留在raw artifact，因为FSG3不形成加速claim；
- `MEASURED-B2-SLOWER`只描述这个固定one-step ResNet workload上的未优化whole-call reference路径，
  不能写成“BoundFlow整体比auto_LiRPA/αβ-CROWN慢”；
- 该工件不包含31-node queue、complete-query、TTV、第二held-out family或natural memory-bound workload；
- FSG5的`1.20x queue / 1.15x complete-query`门槛尚未测试，ASPLOS-ready仍为NO。

下一动作是FSG4/B3预注册：冻结B2语义与B0分母，只允许图/IR/Plan/Schedule复用变量，先量化并消除
optimizer、atomic commit、KFSB、typed pre-state之间的重复编译、验证、materialization与copy-out；
B3关闭前不得把TIR fusion、CUDA Graph、runtime streams或arena reuse混入同一个candidate。

## 本轮验证

- artifact static replay：`replay-passed`，36 runs，summary hash exact；
- FSG3全部测试：`33 passed`；
- 新tamper probe与artifact测试：Black check、mypy clean、Pylint=`10.00/10`；
- 全量回归：`1233 passed, 3 skipped`；三个skip分别是TVM可用时跳过重复allow-no-TVM smoke、两项
  frozen VNN-COMP checkout不可用边界；
- third-party solver原始stdout/stderr按字节进入manifest，`.gitattributes`将对应`logs/*.txt`标记为
  `-diff`，避免格式检查诱导修改冻结raw；其余代码/文档`git diff --check`通过；
- DocOps change/validation/lint在提交前记录。
