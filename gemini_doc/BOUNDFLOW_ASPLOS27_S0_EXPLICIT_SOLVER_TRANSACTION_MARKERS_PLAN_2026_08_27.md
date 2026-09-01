# BoundFlow ASPLOS’27 S0 explicit solver transaction markers 执行计划

date: 2026-08-27
status: internally-validated-closed-budget-recompute-open
parent: `BOUNDFLOW_ASPLOS27_S0_TEN_X_BUDGET_IMPLEMENTATION_PLAN_2026_08_27.md`
external-audit: deferred-by-user
performance-claimed: false

## 1. 目标

解决S0第一批在ResNet2B official B0 fixed-16-iteration prefix中留下的`30.62%—31.51%` inter-call host
mechanism空白。marker只观察，不改变solver、trajectory、tensor值、branching或termination。只有同时满足
profile扰动与机制覆盖门禁，才允许用新bucket重算10×预算。

## 2. 固定外部来源

- αβ-CROWN commit：`e5c7e17bf0488843acb77b7519f59876717a49f4`；
- auto_LiRPA commit：`5a098e8f9fb5786a428a024981d833d303921f2d`；
- VNN-COMP commit：`90419aadcf06cf543ce5c1706cae1059dc9fa6cf`；
- workload、seed、device、dtype、timeout、max iterations、alpha/beta steps与FSG1 B0一致。

目标文件blob：

| 文件 | git blob |
|---|---|
| `complete_verifier/api.py` | `8634ced960d73aa31593c2c3c4df19e9fcca2677` |
| `complete_verifier/abcrown.py` | `5bb2392f84eecc9f442035376d18117398358dc3` |
| `complete_verifier/complete_verifier_func.py` | `18c02c98173bc72fe166f5e0723da47442e0952e` |
| `activation_split/bab_bootstrap.py` | `29f5153bf4b7adc1524ccb795255250b5029fd2b` |
| `activation_split/stage_preprocess.py` | `6ec5c1a794e57e3994a2d616444480737e97e9d1` |
| `activation_split/stage_solve.py` | `75f2d2b7847772d38797066740a57b69190d214f` |
| `activation_split/stage_postprocess.py` | `b5585bb060151c43e48999bdd30d785b299fae18` |
| `activation_split/update_bounds_phases.py` | `60fe57bd78a155cd5bfda68a0161cdfa37519b5a` |
| `complete_verifier/branching_domains.py` | `7ef216f6f26a035316447a69da181de4625751a4` |
| `activation_split/decision_precompute.py` | `52bd81c76b14d36119650f37566eb4f1aeb80a8a` |
| `complete_verifier/heuristics/__init__.py` | `7035fbc811131342233b0685d46edbf219caa720` |

任一commit/blob不符，runner必须在launch前拒绝。

## 3. Marker层级

### 3.1 coarse scope，只提供边界、不计入机制覆盖

- `ABCrownSolver.verify`；
- `ABCROWN.complete_verifier`；
- `ABCROWN.bab`；
- `complete_verifier_func.general_bab`。

这些wrapper即使覆盖全部wall time，也不能把内部空白伪装为“机制已知”。

### 3.2 exact transaction，可计入机制覆盖

- `ABCrownSolver.__init__`：frontend/model/property setup；
- `ABCROWN.incomplete_verifier`：incomplete verification；
- `bab_bootstrap.branch_and_bound_preprocess`：domain pick、branch decision、split/history与bound prepare；
- `bab_bootstrap.branch_and_bound_solve`：BaB solve orchestration；
- `bab_bootstrap.branch_and_bound_postprocess`：transfer、domain commit、sort与queue state；
- `stage_preprocess.update_bounds_pre`；
- `stage_solve.update_bounds_core`；
- `stage_postprocess.update_bounds_post`；
- 现有`BoundedModule.compute_bounds` host/CUDA calls继续作为最深的bound transaction。

实现时根据第一次GPU probe发现的新API真实call site，又绑定了`api.incomplete_verifier_core`、
`api.complete_verifier_core`、`ABCrownSolver._prepare_*`、`IOConstraints`、`_ApiLogger`、
`BatchedDomainList.__init__`和BaB first-decision helpers。第二次probe确认complete-verifier前置空白主要来自
`gc.collect`、`torch.cuda.empty_cache`与实际`ABCrownSolver.bab`边界，因此三者也作为独立exact transaction
记录；它们没有被重命名为算子或solver control。

patch必须施加在实际call-site module global或class attribute，不能只patch原定义却漏掉`from ... import ...`复制。

## 4. 观测合同

每个span记录：稳定target ID、category、resolution、parent、thread、depth、host start/end、returned/raised。
observer采用`perf_counter_ns`，不做CUDA synchronize、tensor读取、DLPack、内存统计或stack inspection。所有patch
在context退出时恢复；异常路径也必须关闭span并恢复原函数。

exclusive summary按最深span归属：compute_bounds高于exact transaction，exact高于coarse。无marker或只落在
coarse scope的时间计为mechanism unresolved；不得用相邻函数推断升级。

## 5. GPU协议与门禁

- 每个workload至少3个fresh control/profile pair，顺序交替；正式目标5 pair；
- control完全不安装transaction/compute observer；profile同时安装两者；
- result status、success、visited domains、compute call sequence必须exact；
- marker target inventory、call count、nesting、scope closure必须exact；
- median `profile/control <=1.05`；
- mechanism coverage `>=0.97`，unresolved `<=0.03`；
- 无失败、fallback或本机路径进入匿名payload。

## 6. 决策

- 扰动和覆盖同时PASS：关闭S0 marker子阶段，重算每个transaction bucket的10×预算；
- 覆盖PASS但扰动FAIL：证据无效，减少marker数量或改用已有timer，不开放S1；
- 扰动PASS但覆盖FAIL：根据最大unresolved span继续补marker，不开放S1；
- `u+h>=0.10`或所有合法机制的physical cap仍无法满足残差`<=0.10`：关闭10× headline；
- 任何情况都不把本阶段observer overhead或derived projection写成性能claim。

## 7. 执行结果

formal artifact：`artifacts/asplos27-s0-transactions/official-b0-five-pair-v1`。

- 2 workloads × 5 pair × control/profile=`20`个fresh进程；
- ResNet minimum coverage=`0.9963239418`、maximum unresolved=`0.0036760582`、median/max perturbation=
  `0.9959015807/1.0416399733`；
- MNISTFC minimum coverage=`0.9924836337`、maximum unresolved=`0.0075163663`、median/max perturbation=
  `0.9986577238/1.0653544672`；门禁只冻结five-pair median `<=1.05`，不要求每对均通过；
- compute signature exact；10/10 pair semantic identity成立；replay通过；全重签summary/protocol篡改拒绝；
- 状态=`s0-explicit-transactions-admitted`，`budget_recompute_open=true`，
  `s1_performance_gate_open=false`，`performance_claimed=false`。

随后派生的事务预算artifact为
`artifacts/asplos27-s0-transaction-budget/official-b0-five-pair-v1`。其研究目标组合在ResNet/MNISTFC上的
在显式假设integration overhead `h=0`时投影分别为`12.562203×/11.656612×`，但所有axis target均
`target_validated=false`，接入成本尚未计入。因此本计划关闭marker
子阶段并开放S1实现，不开放任何性能claim。
