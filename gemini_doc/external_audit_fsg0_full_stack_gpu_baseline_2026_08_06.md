---
status: audited
updated: 2026-08-06T10:05:00Z
type: external-audit
topic: boundflow
slug: external-audit-fsg0-full-stack-gpu-baseline
stage: s01
---

# 外部独立审计报告：FSG0 Full-Stack GPU Baseline Attribution(计划/合同/工具链切片)

- 审计对象：分支 `feat/full-stack-gpu-baseline-attribution-v1`,HEAD=`b641354e86f0310950ac95985a715374665a5ead`
- 审计方式：不信任执行方数字，全部命令亲自复跑；合同语义用独立篡改实验验证
- 审计时间：2026-08-06
- 环境：`/home/lee/miniconda3/envs/boundflow/bin/python` + `source env.sh`

## 总体 Verdict：APPROVE-WITH-MINOR

FSG0 的全部核心声明均可独立复现：git 状态、19 项定向测试、1078 项全量回归、Black/mypy(源文件)/
Pylint 10.00/10、DocOps lint、artifact 语义重放与篡改拒绝。claim 边界无漂移，`performance_claimed=false`
由代码强制。三项 minor findings(文档/schema 枚举命名漂移、测试文件 mypy 未覆盖、replay 不重验
git provenance),均不影响 FSG0 关闭，建议 FSG1 前修正。

## 逐项结论

### 1. git 状态 —— 通过

- `git rev-parse --abbrev-ref HEAD` = `feat/full-stack-gpu-baseline-attribution-v1`;
  HEAD=`b641354e86f0310950ac95985a715374665a5ead`，与声明一致。
- `git fetch origin` 后 `origin/feat/full-stack-gpu-baseline-attribution-v1` = 同一 `b641354...`,
  远端与本地一致，已推送属实。
- `git status --porcelain` 唯一未提交修改为 ` M .docops/ev.jsonl`(PostToolUse 自动事件),与声明一致。
- `git show --stat b641354`:21 个文件、+2506/-78，包含声明的五项产出
  (PLAN 365 行、CHANGELOG 70 行、`gpu_attribution.py` 741 行、runner 200 行、测试 398 行),
  另含对 NRIR49A/执行备忘录/claims map/status 等权威文档的作用域纠正及上一份 G1 关闭审计归档，
  范围与声明相符，无夹带 production 代码改动。

### 2. 计划文档质量 —— 通过(含一项 minor 命名漂移)

- (a) 逻辑衔接：PLAN §0 把 NRIR49A 判定作用域收窄为 `VALIDATED-NO-GO(selected-CROWN-only
  incremental optimization)`,`1/(1-0.070986)=1.0764x` 明确标注为 deletion-only 单区域上限;
  `next_route=gpu-winner-reselection` 作为冻结历史机器输出保留不改。与 NRIR49A G1 plan
  顶部"后续范围纠正"段、执行备忘录 §62 措辞一致；已死的 selected-CROWN 专属 TIR/JIT/融合路线
  未复活(B3—B7 均为独立 feature gate，不含"selected-CROWN-only"实现)。
- (b) 分层模型：PLAN §5 四轴(layer/phase/resource/cache)+ 五种时间口径
  (host_wall/gpu_union/gpu_sum/critical_path/exclusive_critical_path)定义可测量；
  exclusive 归属闭合到 critical path，闭合门禁 `<=1%`、residual `<=3%`、paired wall 中位
  ratio `<=1.05` 均预注册(§5、§8.2),代码 `summarize_run` 逐项落实。多流重叠以
  union/sum 分离，禁止 `gpu_sum/wall` 伪 share,已在 `gpu_overlap_ns` 字段落实。
- (c) B0 control 公平性：§4.1 固定同 solver/repo/config/model/property/seed/branch/iteration/
  termination;combined env 优先，冲突时只允许对称 RPC 且 IPC 成本计入双方；§8.2 预注册
  5 fresh、AB/BA 反平衡、profiler 分类与 unprofiled wall 分离；§4.2 明确列出只能诊断不能
  headline 的对照。公平性设计严谨。
- (d) 门禁预注册：§8.1/8.2/8.3 全部量化(closure 1%、residual 3%、perturbation 1.05、
  5 fresh、queue 1.20x、complete 1.15x、退化上限 5%、memory 路线 25%+1.05),未发现量化漏洞;
  门槛只施加到累计 B7 vs B0，防止单区域外推。
- (e) 无提前性能表述：§1 明确 `UNMEASURED / NOT-YET-AUDITABLE`;全文无 speedup 数字;
  `performance_claimed=false` 在 `FullStackAttributionRun.validate()` 中强制
  (true 直接 raise),artifact manifest 同样硬编码 `False`。
- Minor(M1):PLAN §5 冻结 phase 表写作 `alpha_opt`，而规范代码 `SolverPhase.ALPHA_OPTIMIZE`
  取值 `alpha_optimize`(gpu_attribution.py:35);代码另含 `setup`/`unclassified` 哨兵值，
  CHANGELOG 称"十个 solver phase"与代码 11 个枚举值的口径需要一句话说明。建议以代码枚举为准
  修订 PLAN 表。

### 3. 合同代码审查 —— 通过(含一项 minor replay 缺口)

- schema 覆盖：`StackLayer` 10 值(9 层 + residual)、`SolverPhase` 11 值、`ResourceKind` 5 值、
  `CacheState` 5 值、`ReplacementMode` A0—A4、`FeatureKind` 7 项，与 PLAN §4.3/§5/§5.1 对应。
- 物理激活账本(gpu_attribution.py:212-294):区分对象存在与物理驱动(dispatch 计数、
  stream/event/wait 计数、storage plan enforced、replacement mode);`activated_features()`
  只做投影不做推断，解析器重算投影并拒绝伪造(测试 16 验证)。
- 关键路径闭合:`_validate_critical_path`(561-581)强制互斥(重叠拒绝)、scope 内、
  source span 存在;`summarize_run`(605-658)计算 closure_error/residual_share 并按
  预注册阈值判 `attribution_passed`。span 森林校验含父包含、依赖时序、双图环检测。
- runner:`generate_artifact` 拒绝覆盖非空目录、closure/residual 不过则 fail closed;
  `replay_artifact` 先验 manifest 自哈希与 `code_revision`(对活仓库文件实时重哈希),
  再从 `raw_run.json` 语义重算 summary 并逐字段比对，为真语义重放而非 digest 信任。
- 独立篡改实验(/tmp/fsg0_audit，非测试文件自带用例):
  - A:改 `summary.json` payload 并同步 manifest 文件 digest + 重算 manifest_hash →
    **拒绝**(`semantic replay differs`);
  - B:改 `raw_run.json` span 时间并同步全部 digest → **拒绝**(同上);
  - C:仅伪造 manifest `git_head` 并同步 manifest_hash → **通过**(见 M3)。
- Minor(M3):replay 用活仓库实时重哈希校验 `code_revision`,但不重验 `git_head`/
  `git_dirty_paths`,git provenance 可被伪造后同步 manifest_hash 绕过(实验 C)。建议在
  `replay_artifact` 中对 `git_head` 做 `git rev-parse HEAD` 实时比对，或文档声明其为
  informational 字段。
- 类型标注完整、风格与仓库一致(dataclass frozen、snake_case、简洁 docstring)。

### 4. 测试审查 —— 通过(19/19 复现)

- 覆盖:schema 合法/非法 roundtrip(测试 15)、closure 违反(2)、exclusive 重叠(3)、
  residual 超阈(4)、父/依赖出界(5)、双图环(6)、union 不去重(7)、activation 投影(8/9)、
  deletion ceiling 与 NRIR49A 数字锚定(10,`0.07098631834282758→1.0764104115`)、
  joint Amdahl(11)、interaction 不可线性相加(12)、hash 绑定 feature/timing(13)、
  performance 自声明拒绝(14)、activation 投影伪造拒绝(16)、artifact generate/replay
  roundtrip(17/18)、**digest 同步篡改拒绝**(19)。断言均具体(`pytest.raises(..., match=)`
  或精确数值),无"任意异常"式断言。
- 复跑：`pytest tests/test_full_stack_gpu_attribution.py -v` → **19 passed in 0.99s**,属实。

### 5. 全量回归与工具链 —— 通过(一处 claim 范围需注意)

- 全量:`pytest tests -q -rs` → **1078 passed, 3 skipped in 399.81s**,与声明
  (1078/3,402.60s)一致。skip 原因:`test_artifact_phase5d_smoke`(TVM 可用时跳
  allow-no-tvm smoke)、两个 frozen VNN-COMP checkout 不可用 —— 均为既有良性跳过，与本次无关。
- Black:`black --check` 三个新文件全部 unchanged。
- Pylint:三个新文件 **10.00/10**,声明属实。
- mypy:`--follow-imports=skip` 对 `gpu_attribution.py` + runner **clean**(2 files);
  但对测试文件报 5 个错误(142/143/290/291/334 行，`object` 不可索引)。仓库既有测试文件
  (抽查 `test_ancestral_constraint_refinement_artifact.py`、
  `test_complete_verifier_query_artifact.py`)mypy clean，故测试文件未过 mypy 偏离仓库惯例 ——
  Minor(M2)。建议给 `summary[...]` 取值加 `cast`/类型收窄，或将 CHANGELOG 中
  "targeted mypy clean"的作用域注明为仅源文件。
- DocOps:`dol lint --soft` → `{"ok":true,...,"soft":true}`,PASS。

### 6. claim 边界 —— 无漂移

- `asplos_claims_map.md` 顶部、`current_status_after_pr13.md`(1290-1294 行)、执行备忘录 §62
  三处措辞一致：FSG0 仅 schema/replay 合同验证,FSG1 只建 B0 基线,`performance_claimed=false`,
  ASPLOS-ready 仍 NO。
- 历史数字引用全部与 NRIR49A 冻结记录逐位一致:queue/complete share `7.0986%/7.0523%`
  (=`0.0709863183/0.0705232890`)、`1.0764x`、显存 `0.996%/1.353%`(allocated/reserved 双口径)。
- PLAN §1 显式声明"本文完成不等于 replacement executor、TIR、JIT 或性能结果已经实现",
  未把"合同存在"暗示为"瓶颈已找到";B0 trace 尚不存在，归因份额一处都没有给出。

### 7. observer 纪律 —— 通过(本切片无 instrumentation)

- `gpu_attribution.py` 为纯合同/聚合代码(frozen dataclass + 纯函数),无任何 hook、
  solver 调用或全局状态，不存在改变 solver 语义的可能。
- FSG1 的 observer 要求已预注册(PLAN §13):observer 必须可逆、off 时调用顺序与结果 exact;
  计时口径(host wall 用 unprofiled control、profile 仅分类)在 §5/§8.2 声明清晰。

## Findings 汇总

| severity | path | evidence | advice |
|---|---|---|---|
| minor | gemini_doc/BOUNDFLOW_FULL_STACK_GPU_BASELINE_ATTRIBUTION_V1_PLAN_2026_08_06.md §5 | PLAN 表 `alpha_opt` vs 代码 `SolverPhase.ALPHA_OPTIMIZE="alpha_optimize"`(gpu_attribution.py:35);代码另有 `setup`/`unclassified` | 以代码枚举为规范修订 PLAN 冻结表，注明哨兵值 |
| minor | tests/test_full_stack_gpu_attribution.py:142,143,290,291,334 | `mypy --follow-imports=skip` 报 5 个 `object` 索引错误；既有测试文件 mypy clean | 加 `cast`/类型收窄；或把 mypy 声明作用域注明为仅源文件 |
| minor | scripts/run_full_stack_gpu_baseline_attribution.py:138-174 | 独立实验 C:伪造 `git_head`+重算 manifest_hash 可通过 replay | replay 中对 `git_head` 实时比对 `git rev-parse HEAD`,或声明 informational |

无 blocker、无 major。

## 不可现场复核项

- CHANGELOG 所称"文档作用域只读审计发现 12 个当前指令风险点"的过程本身不可复核(无独立
  审计 artifact);但其产出(各权威文档的作用域纠正)已逐处比对一致，可接受。
- GPU 计时类指标本切片按设计不存在(FSG0 无测量),无此项复核需求。
- `pytest` 全量耗时 399.81s 与执行方 402.60s 的差异为机器噪声，无意义。

## 对 FSG1(B0 control trace 采集)计划的评价

设计合理且门禁完备:official env 内 hook `compute_bounds`/solver phase/CUDA runtime/allocator;
5 fresh paired control/profile;profile 只分类、wall 用 unprofiled control;perturbation ratio
`<=1.05` 预注册。两点风险建议 FSG1 开工时正视:

1. combined env 可行性是最大技术风险(Python 3.11/Torch 2.11 vs 3.12/2.12,TVM 不可 import
   进 competitor env);若落到对称 RPC,§4.1 的对称性条款(IPC 成本计入双方、warmup 对称)
   必须在 FSG1 artifact 中逐项给证据，不能只声明。
2. hook official αβ-CROWN 内部函数的 patch 面要最小化并全部入 manifest(补丁文件 digest),
   否则"同一 solver"前提会被质疑。FSG1 应保持 FSG0 的 raw-first artifact 纪律:raw events
   先行,summary 全部由 replay 重算。

## 复核命令与输出摘录

```text
git rev-parse HEAD -> b641354e86f0310950ac95985a715374665a5ead
git fetch origin && git rev-parse origin/<branch> -> 同一 b641354...
git status --porcelain -> " M .docops/ev.jsonl"(唯一)
pytest tests/test_full_stack_gpu_attribution.py -v -> 19 passed in 0.99s
pytest tests -q -rs -> 1078 passed, 3 skipped in 399.81s
black --check (3 files) -> All done! 3 files would be left unchanged
mypy --follow-imports=skip (2 source files) -> Success: no issues found
mypy (test file) -> 5 errors [index](finding M2)
pylint (3 files) -> 10.00/10
dol lint --soft -> {"ok":true,"soft":true}
独立篡改实验 A/B -> rejected(semantic replay differs);C(git_head 伪造) -> passed(finding M3)
```
