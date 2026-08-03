# RVIR 外部审计报告（r001 audit_response）

- task: rvir-20260803
- round: 1
- from: external-auditor -> to: user / codex
- date: 2026-08-03
- 审计对象：`feat/real-verifier-ir-integration-v1`，实现范围 `1406d4b..5a5a8a4`（外加 `6428665` DocOps、`3c7f739` handoff docs）
- 审计方式：不信任交接数字，全部从代码、artifact、测试与 git 历史独立复核；Python 一律使用 `/home/lee/miniconda3/envs/boundflow/bin/python`，命令前 `source env.sh`
- 说明：本报告覆盖同路径下一份先前生成的同名草稿（untracked，2026-08-03 14:01 +0800）；所有数字均由本轮命令重新产生，未采信该草稿。两处结论与其不同：(a) PR #4 本轮可现场审计（gh 查询成功）；(b) 全量测试需先修正 PATH 才复现 452 passed（见 §6 与 minor M2）。

## 总体判定

**同意以 VALIDATED-REDUCED（CPU correctness/integration）关闭。**

0 blocker，0 major，5 minor（见末节）。四条 acceptance criteria 全部独立复核成立。

## 逐项结论

### 1. 提交、tag、PR、工作树、文件范围 —— 成立

- 分支：`git branch --show-current` = `feat/real-verifier-ir-integration-v1`。
- 提交链：`git log --oneline d457b22..HEAD` 依次为 `1406d4b → 05669ad → 3e7460e → e03b3d2 → 5a5a8a4 → 6428665 → 3c7f739`，与 handoff §3 一致。
- tag：`git cat-file -t rvir-cpu-correctness-validated-reduced` = `tag`（annotated）；`git rev-parse ...^{commit}` = `5a5a8a4446a98f20b0dcd3deb2777a78ddbeacb3`，指向 closure commit `5a5a8a4`，刻意不含后续 DocOps/docs 提交，与声明一致。
- 文件范围：`git show --stat` 逐提交核对：`05669ad`（bound.py/abcrown_adapter.py/bound_ir_interpreter.py/replay 脚本/测试），`3e7460e`（bound.py +110、plan.py、task_v1.py、plan_ir_builder.py、verifier_ir_integration.py +729、测试），`e03b3d2`（恰好冻结 artifact 四文件：394 行 jsonl + manifest + online + resnet，及 runner +370/测试），`1406d4b`/`5a5a8a4`/`6428665` 为文档/DocOps。未发现范围外代码改动。
- PR #4：**成立（本轮可现场审计）**。`gh pr view 4 --json ...` 返回 `baseRefName=feat/compiler-ir-stack-v1`、`headRefName=feat/real-verifier-ir-integration-v1`、`headRefOid=3c7f739…`、`state=OPEN`、`isDraft=true`，与 handoff 一致。
- 工作树：唯一已跟踪脏文件 `M .docops/ev.jsonl`，`git diff` 显示仅追加一行良性 DocOps hook 事件 `ev000037`（`"ty":"cmd","cmd":"Bash:git"`），非审计范围；另有 untracked 的本报告文件。

### 2. RVIR-1 数值与 external intermediate-bound ownership —— 成立（原始 rerun 不可现场审计）

独立解析 `artifacts/rvir/rvir-cpu-correctness-v1-20260803/resnet_semantics.json`（逐字段打印）：

- `intermediate_bound_count=6`、`intermediate_bound_source=external_verifier`、`relu_lower_slope_policy=adaptive`；
- `intermediate_bounds_hash=d51615b04dfb205afd67d2c21680ece4ca92f693157da1e32c7f8202a8e08cf1`；
- `lower_allclose=true`、`lower_max_abs_diff=3.0994415283203125e-06`、`sign_agreement=9/sign_total=9`；
- `device=cpu`、`abcrown_commit=e5c7e17bf0488843acb77b7519f59876717a49f4`、`performance_compliant=false`、`performance_claimed=false`。

代码责任抽查（成立）：

- `boundflow/ir/bound.py:485` `IntermediateBoundSource`、`:492` `ReluLowerSlopePolicy`、`:588-591` 两字段进 `ReluRelaxationAttrs` 并有类型校验（`:605-608`）；`BoundOp.to_dict` 用 `asdict(self.attrs)`（`bound.py:1272`）序列化全部 attrs 字段，`canonical_json`/`stable_hash`（`bound.py:1428-1438`）对其做 SHA256，故两字段确实进入 canonical JSON 与 stable hash。
- `boundflow/runtime/abcrown_adapter.py`：逐 ReLU 捕获 + ordinal/identity/shape/dtype/finite/lower≤upper 逐项 fail closed（`:224-241`），aggregate SHA256 含 ordinal 连续性校验（`:285-299`）。

限制：max diff/sign 的**原始重跑不可现场审计**（需 external αβ-CROWN 运行环境）；本轮验证的是冻结证据 + 生成端门禁（`scripts/run_real_verifier_ir_artifact.py:222-232` 对 count=6/allclose/sign/≤2e-4 逐项 fail closed）。

### 3. RVIR-2 五层 IR 类型、hash、state identity、backend 限制 —— 成立

代码抽查（行号均为当前工作树）：

- Bound：`BoundOpKind.EXTERNAL_VERIFIER_CALL`（`bound.py:482`）；`ExternalVerifierCallAttrs`（`:738-798`）显式拥有 provider/phase/method/requested_bounds/input_region_hash/objective_hash/α/β/split/cuts version，`validate` 强制 `semantics_owner=="external_verifier"`、requested 非空无重复、αβ-CROWN 必须带 α/β/split 版本（`:787-798`）；value contract 要求 perturbation+objective 输入、输出 polarity 与 request 一致（`:950-982`）。
- Plan：`RegionKind.EXTERNAL_VERIFIER`（`boundflow/ir/plan.py:35`）、`BackendKind.EXTERNAL_ABCROWN`（`plan.py:49`）。
- Task：`TaskIRKind.EXTERNAL_VERIFIER_CALL`（`boundflow/ir/task_v1.py:49`）、`TaskExternalDependencyKind.EXTERNAL_VERIFIER_STATE`（`:66`）、region→kind 映射（`:625-626`）、`BackendKind.EXTERNAL_ABCROWN → external_abcrown_exact_call/v1`（`:742`）。
- 编译链：`compile_external_verifier_call`（`boundflow/runtime/verifier_ir_integration.py:301-335`）严格按 ExternalVerifierCallSpec → BFBoundModule → PlanTemplate → PlanInstance → TaskIRModule → ScheduleModule 构造并 `validate()`；`validate`（`:235-263`）强制恰好 1 个 task、正确 TaskIRKind、**恰好一次 launch + 一次 emit**、schedule/backend identity 一致；`hashes()`（`:265-287`）输出五层 stable hash 且每层 hash 链接上游层。
- 执行：`execute_external_verifier_call`（`:338-355`）对 `reference_implementation_id != external_abcrown_exact_call/v1` **拒绝执行**（undeclared backend），随后 `exact_call()` 恰好一次。

fail-closed 独立验证（ad-hoc 探针，非仓库测试，本轮亲自运行）：四项篡改全部被 ValueError 拒绝——(a) backend 改为 `local_fused_kernel/v9` → 拒绝；(b) schedule 去掉 emit → 拒绝；(c) 重复 launch → 拒绝；(d) `semantics_owner` 改为 `boundflow` → 拒绝（"external verifier call must retain external ownership"）。

注意（minor M1）：`tests/` 中没有针对上述 schedule/backend 拒绝路径的**专用负向单测**；正向契约测试存在于 `tests/test_real_verifier_ir_integration.py`（2 个测试，含五层 hash 互异与 launch==emit==1 断言），RVIR-1 侧有 fail-closed 测试（`tests/test_bound_ir_v1_plain_crown.py:296`），Task/Schedule linkage 负向测试在 `tests/test_task_ir_v1.py:169`。

### 4. RVIR-3 query/result/parent accounting 与 observer equivalence —— 成立

独立解析 `online_execution.json`：

- `query_count=377`、`compiled_and_dispatched=377`、`completed=377`；
- activation phase：`336 CROWN + 7 alpha-beta-CROWN = 343`，`activation_effective_method_counts={alpha_beta_crown: 343}`；另有 34 个 initialization phase query（33 CROWN + 1 alpha-CROWN），合计 377；
- `root_query_count=30`、`parent_link_count=347`（30+347=377 ✓）；
- `requested_output_counts={lower: 377}`（全部显式 lower-only）；
- observer 对照：`baseline_visited_domains=[380]`、`profiled_visited_domains=[380]`、`status_match=true`、`visited_domains_match=true`；`result_status=baseline_status=unknown`；`bab_projection` 唯一行 `final_lower=tensor(-0.18902308)`、`visited_domains=380`（生成端 `_bab_projection` 要求 baseline 与 typed 投影逐行相等，`run_real_verifier_ir_artifact.py:155-156`）；
- `semantics_owner=external_verifier`、`performance_claimed=false`、`device=cpu`、abcrown commit `e5c7e17…`。

限制（minor M4）：artifact 内是摘要投影；parent-precedes-child 的顺序约束只在生成端强制（`run_real_verifier_ir_artifact.py:164-168`），artifact 不含可独立重放的在线 queries/records 原文（仅嵌入 source digests）。

### 5. RVIR-4 artifact integrity、self-contained replay 与 394 coverage —— 成立

- replay：`source env.sh && <env python> scripts/run_real_verifier_ir_artifact.py replay --artifact-dir artifacts/rvir/rvir-cpu-correctness-v1-20260803` → 退出码 0，输出 `{"activation_call_count":394,"performance_claimed":false,"status":"replayed"}`。
- 非空转确认（读源码 `run_real_verifier_ir_artifact.py:303-325`）：replay 先校验 manifest schema 与三个文件的 SHA256（`:309-314`），再逐行读取 394 行并对每行调用 `_activation_row` → `ExternalVerifierCallSpec.from_query_dict` + `compile_external_verifier_call(spec)` 完整重编译五层 IR，最后**整行 dict 相等比较**（`row != expected` 即失败，`:318-324`）。输入只来自 artifact 内嵌的 `query` 字段，不依赖本机 ignored 历史目录（self-contained ✓）。
- digest 独立复核：本机 `sha256sum` 三文件结果与 `manifest.json.files` 逐一相同（activation_calls `b8dc6652…`、online `2f89166b…`、resnet `a86056c3…`）。
- 独立解析 `activation_calls.jsonl`（自写脚本逐字段统计）：
  - 394 行、394 个 unique query_id；
  - workload：`official-simple-mlp-cuda-bab=343` + `vnncomp21-resnet2b-prop0=51`；
  - `effective_method={alpha_beta_crown: 394}`（typed admission 394/394：每行均含五个 hash key——bound_module/plan_template/plan_instance/task_module/schedule）；observed method 原始值为 `CROWN=386 + alpha-beta-CROWN=8`，effective 归一为 αβ-CROWN；
  - `backend=external_abcrown_exact_call/v1`、`semantics_owner=external_verifier`、`performance_claimed=false` 均 394/394；
  - 三项 legacy limitation 真实存在且各 394/394：`split_state_values_unresolved`、`legacy_requested_bound_polarity_unresolved_assumed_both`、`legacy_parent_lineage_not_captured`；原始 query 中 `split_tensor_values` 全缺失、`bound_lower_requested` 全缺失（保守 assumed both）、`parent_query_id` 0 行。
- fused coverage `0/394` 历史结论未被覆盖：见 §7。

### 6. 全量测试、mypy、pylint、DocOps —— 成立（DocOps CLI 部分不可现场审计）

- 全量 pytest：`source env.sh; export PATH=<env>/bin:$PATH; python -m pytest -q tests` → **452 passed, 37 skipped**（55.22s），与声明一致。
  - 注意（minor M2）：若不加 PATH（直接用 env 解释器路径调用），`tests/test_phase6h_artifact_runner_smoke.py` 会失败——该测试 `bash scripts/run_phase6h_artifact.sh`，脚本内的 `python` 解析为系统 `/usr/bin/python`（无 torch），报 `ModuleNotFoundError: No module named 'torch'`。这是调用方式问题而非代码回归；修正 PATH 后 452/37 完全复现。
  - skip 原因（`pytest -rs` 汇总，37 = 4+4+4+2+2+2+2+16×1+1）：全部为 `CUDA required/is required/unavailable`（TVM dispatch、PR-12/13 CUDA 套件），仅 1 项为例外但同为环境边界：`test_artifact_phase5d_smoke.py:118`（TVM 可用时主动跳过 allow-no-tvm smoke 以避免重复编译）。无逻辑性 skip。
- mypy：6 文件（verifier_ir_integration/abcrown_adapter/bound/plan/task_v1/plan_ir_builder）→ `Success: no issues found in 6 source files`。
- pylint：`scripts/run_real_verifier_ir_artifact.py tests/test_real_verifier_ir_artifact.py` → `10.00/10`。
- DocOps：`.docops/s.md` = `tp: boundflow, st: s01, pr: [4], next: external-audit-rvir-20260803`，与声明一致；`ev.jsonl` 中 `ev000009`（ch, rvir-cpu-correctness-closure）、`ev000010`（va, pass）均存在。`dol` CLI 在本非交互环境不可用（PATH 与 conda env bin 均无）→ `dol validate` / `dol lint --soft` **不可现场审计**；但底层事件记录已直接抽查且一致。

### 7. handoff §8 八条限制在权威文档中的一致性 —— 成立

逐条核对 `gemini_doc/real_verifier_ir_integration_closure_2026_08_03.md`（§4 不得升级的主张、§5）、`gemini_doc/asplos_claims_map.md`（L5-7、L13、L17、L408、L421-431）、`gemini_doc/current_status_after_pr13.md`（L4-7、L37-38、L50、L85、L154-155）：

1. `0/394` fused coverage 与 `394/394` typed admission 区分明确保留（claims map L408/L423-431；closure §4.1）✓
2. adapter v1 三项 identity 缺失（split values/polarity/lineage）明确保留（closure §4.2；artifact 逐行写入，§5 已验证）✓
3. adapter v2 377-call 仅覆盖 upstream simple MLP CPU（closure §4.3；online_execution.json workload_name）✓
4. ResNet 仅证明 external-semantics initial plain-CROWN（closure §2 RVIR-1、§4；claims map L17 同时保留 local 历史 max diff 796.765）✓
5. 无 fresh CUDA 证据（closure §4.4）✓
6. 无 performance claim（closure §4.5；artifact 处处 `performance_claimed=false`）✓
7. lower-only vs lower+upper 公平性能合同未建立（closure §4.5）✓
8. IR-5 仍 VALIDATED-NO-GO、IR-6 不启动、ASPLOS-ready=NO（closure §5；status doc L6-7/L36/L126；claims map L6-7）✓

未发现性能/CUDA/BoundFlow-αβ-kernel/完整 VNN-COMP E2E 的 claim 漂移。

## Findings

- Blocker：无。
- Major：无。
- Minor：
  - M1：`tests/` 缺 RVIR-2 schedule/backend 拒绝路径的专用负向单测；强制逻辑在库代码中且已被本轮 ad-hoc 探针验证（4/4 拒绝）。建议后续补 `pytest.raises` 用例固化。
  - M2：`test_phase6h_artifact_runner_smoke` 依赖 PATH 中的 `python` 带 torch；未激活 env 时单独失败（452→451+1 failed）。与 RVIR 无关，但 handoff 的 pytest 命令应以 `conda run -n boundflow` 或等价 PATH 为准。
  - M3：审计开始时工作树已存在先前生成的同名 audit_response.md（untracked）；本报告已完全独立复核并覆盖之，且修正了其中两处结论（PR 可现场审计；pytest PATH 前提）。
  - M4：online_execution.json 为摘要投影，parent 顺序与 observer 投影相等性只在生成端强制，artifact 内不含在线原始 queries/records 以供第三方独立重放（仅 source digests）。
  - M5：ResNet max diff 3.09944e-6 / sign 9/9 的原始数值不可现场重跑（需 external αβ-CROWN 环境）；本轮验证范围为冻结证据、digest 与生成端 fail-closed 门禁。

## 结论

四项 acceptance criteria 全部独立复核成立：ResNet max diff `3.0994415283203125e-06`、sign `9/9`（冻结证据+digest 一致）；typed admission `394/394` 且 fused `0/394` 历史结论未被改写；在线 `377/377` 与 `380`-domain observer equivalence；全部 legacy identity 与 CPU/性能限制在权威文档中一致保留。**同意 RVIR 路线以 VALIDATED-REDUCED（CPU correctness/integration）关闭。**
