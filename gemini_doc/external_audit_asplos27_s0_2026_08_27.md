# BoundFlow ASPLOS'27 S0（explicit solver transaction + 10× 预算）独立外部审计报告

- date: 2026-08-28（审计执行日；被审工作标注 2026-08-27）
- auditor: 独立外部审计方（非执行方、非前序外审 GC0-1）
- scope: 分支 `feat/rvir-v4-production-state-ownership-v1` @ `20f4741` 之上**未提交**的 S0 第二批工作
- method: 所有数字从 raw（`worker_runs.jsonl`）用**不依赖 boundflow 包**的 stdlib-only 脚本独立重算；
  artifact replay 用被审代码执行以验证其自洽性；tamper 用自建变体验证
- python: `/home/lee/miniconda3/envs/boundflow/bin/python`（3.12.12），`source env.sh` 后执行

## 0. 总体结论

**Verdict: approve（approve-with-minor）**

执行方声明的所有可复核数字均与独立重算一致；artifact 语义 replay 与全重签 tamper 拒绝经独立抽查成立；
claim 边界（projection ≠ 性能、S1 开放、performance gate 关闭）在代码、artifact、changelog、计划与两份
draft 中一致，未发现越界表述。无 blocker、无 major。两条 minor、三条 info 见第 9 节。

对"S0 attribution 通过 / S1 implementation 开放 / S1 performance gate 关闭"这一状态判定的评价：**恰当且
证据充分**。S0 的门禁（≥97% 机制覆盖、≤1.05 扰动中位、语义一致、compute signature five-fresh exact）从
raw 重算全部通过；预算只推导"数学可达性假设"并显式冻结 `performance_claimed=false`、
`s1_performance_gate_open=false`、`all_axis_targets_validated=false`，没有把投影写成结果。

## 1. 标记实现复核（清单第 1 项）

- `boundflow/runtime/solver_transaction_observer.py`：wrapper 只做 `time.perf_counter_ns` 计时与栈记录，
  不调用 CUDA sync、不读 tensor、不检查调用栈（`observer_protocol` 中三者均声明 false 且在 replay 时强制
  校验，markers 脚本第 815–826 行）；`instrument()` 以 contextmanager 事务化安装并在 `finally` 中逐目标
  恢复原函数（第 219–244 行）；control 模式用 `nullcontext()` **完全不装任何 instrumentation**
  （`run_asplos27_s0_transaction_markers.py` 第 628–633 行），扰动由 profile/control 配对量化。
- 33 个标记独立枚举确认：对 `TARGET_SPECS` 做 AST 解析得 **33 条 = 28 exact_transaction + 5
  coarse_scope**；coarse 为 `verify_scope`、`complete_verification_scope`（abcrown.ABCROWN 与
  api.complete_verifier_core 两处）、`bab_scope`(abcrown.ABCROWN.bab)、`bab_bootstrap_scope`。
  exact/coarse 语义与 `summarize_solver_transactions` 的归属规则一致：compute call 最深者优先，其次最深
  span，exact 才计为 resolved，coarse 只界定 `mechanism_unresolved:*` 范围（observer 第 382–443 行）。
- 目标真实性由 11 个外部源码 blob 的 git blob hash 绑定（`TARGET_BLOBS`），worker 启动时
  `_verify_external_source` 强制核对 αβ-CROWN `e5c7e17` / auto_LiRPA `5a098e8` / VNN-COMP `90419aa`。
- observer 单元测试覆盖恢复语义、嵌套归属、raised 结局、重复 patch 拒绝、schema 漂移拒绝
  （`tests/test_solver_transaction_observer.py`，6 tests）。

结论：**通过**。

## 2. Formal 证据独立重算（清单第 2 项）

独立脚本（stdlib-only，未 import boundflow）从头实现 exclusive-timeline 归属，结果：

- `worker_runs.jsonl` 恰 20 行 = 2 workload × 5 repeat × {control, profile}；`pairs.jsonl` 恰 10 行；
  pair_order 交替（`control-profile`/`profile-control`），每对语义字段
  （configuration/workload/repeat/pair_order/source/solver_protocol/environment/result）逐项相等；
  control 记录 `compute_calls`/`transactions` 为空且 `transaction_summary` 为 null。
- 5 对均按设计由独立子进程产出（runner 第 775–784 行逐 run 起 subprocess，超时隔离）。
- 独立重算的每 run 机制覆盖率与记录值误差 < 1e-15；逐 workload：

| workload | 最低机制覆盖（重算） | 声明 | 扰动中位（重算） | 声明 |
|---|---:|---:|---:|---:|
| cifar10_resnet:000 | 99.6324% | 99.632% | 0.995902 | 0.9959× |
| mnistfc:2 | 99.2484% | 99.248% | 0.998658 | 0.9987× |

- "最低机制覆盖"定义核对：= 5 次 repeat 各自 `mechanism_resolved_ns/scope_ns` 的最小值（summary 第
  997–1008 行 `min(coverages)`），与声明口径一致；max unresolved 分别为 0.3676% / 0.7516%，门禁
  ≥97%/≤3% 通过。
- 每对扰动比 = profile scope_ns / control scope_ns，重算与 `pairs.jsonl` 记录一致（误差 < 1e-18）；
  5 次中位数 0.9959016 / 0.9986577，门禁 ≤1.05 通过。**注意**：门禁作用于中位数，MNIST-FC r0 单对为
  1.065354（>5%）——按冻结协议不构成失败，但见 findings info-2。
- 两 workload 的 compute signature five-fresh exact（`compute_signature_exact=true`）；全部
  `performance_claimed=false`。
- 环境：20 个 run 全部 RTX 4060 Laptop GPU、torch 2.11.0+cu130、python 3.11.15（外部 venv），timeout 60s、
  max_iterations 16，result 均 verified 且 pair 内一致。

结论：**通过，数字全部独立复现**。

## 3. Replay 与 tamper（清单第 3 项）

- transactions artifact replay（本机执行）：
  `{"status":"replay-passed","evidence_status":"s0-explicit-transactions-admitted","pair_count":10,
  "summary_hash":"3edaab81df10…606ca0"}`，与 changelog 记录 hash 一致。
- transaction budget artifact replay：
  `{"status":"replay-passed","evidence_status":"s0-transaction-budget-research-route-open",
  "s1_implementation_open":true,"s1_performance_gate_open":false,
  "summary_hash":"25ced926a58c…c168c"}`，与 changelog 一致。
- 另复跑第一批 tenx 诊断 artifact replay：`replay-passed`，hash `386d2aeb…056eb9`，与第一批 changelog 一致。
- replay 是语义重算而非纯 hash 比对：`replay_artifact` 从 raw `worker_runs.jsonl` 重新执行
  `_derive_payloads`（内部对每个 profile 重新 `summarize_solver_transactions` 并拒绝 summary 漂移，
  第 891–898 行），再逐字节比对重新生成的 `pairs.jsonl`/`summary.json`/`protocol.json` 等。
- tamper：被审测试含 4 个全重签变体（worker summary、protocol target、projection、axis target，均重签
  worker_hash/protocol_hash/report_hash/manifest_hash 后仍被拒）。本审计另自建 2 个变体：
  - A：raw span 收缩（exact span 截半）+ 重签 worker_hash + 重签 manifest → 拒于
    "worker semantic summary differs"；
  - B：仅改 `pairs.jsonl` 的扰动比 + 重签 manifest → 拒于 "semantic replay differs: pairs.jsonl"。

结论：**通过**。

## 4. 10× 预算推导链（清单第 4 项，重点）

- 推导链独立重算：对每 workload 汇总 5 个 profile 的 `sum(category_ns)/sum(scope_ns)`，按 O1–O5
  互斥轴分类（`mechanism_unresolved:*` 一律进不可优化桶 U、按 1× 冻结）：
  `projection = 1 / (share_U + Σ share_i/target_i)`，target = (O1 16, O2 8, O3 12, O4 20, O5 4)。
- 重算结果（stdlib 独立算术）：ResNet2B **12.562203×**（声明 12.5622×）、MNIST-FC **11.656612×**
  （声明 11.6566×）；轴份额 ResNet O1 67.962%/O2 20.718%/O3 7.258%/O4 3.730%/U 0.332%，MNIST-FC
  O1 20.243%/O3 78.973%/O4 0.053%/U 0.729%，与 changelog 第 45–48 行完全一致；达 10× 所需 resolved
  全栈平均 10.3087×/10.7081× 亦一致。
- 投影/目标定性披露充分：轴 policy 携带 `target_validated=false`、`evidence_note`（O1 自标 "stretch
  target"，锚点 CIBC 12.795× 与 B4-B2 4.898× 为不同 scope）；report 级 `status=
  s0-transaction-budget-research-route-open`、`all_axis_targets_validated=false`、
  `s1_performance_gate_open=false`、`performance_claimed=false`；changelog §4 明文"投影不是已实现性能，
  不能进入 abstract、headline 或对外宣传"。
- 隐性乐观假设检查：
  - unresolved 按 1× 冻结——**保守**，已披露（protocol `unresolved_policy=immutable_at_1x`）；
  - 轴覆盖率假设为全份额——已披露（"research target; not measured"）；
  - **integration overhead h=0 未在预算公式中出现**（`derive_transaction_budgets` 无 h 项），主计划 A.3
    的通用公式含 h 且明确 `u+h>=0.10` 即不可达，但 12.56×/11.66× 预算表旁未显式标注 h=0——见
    findings minor-1。

结论：**数字通过；定性边界通过；h=0 披露记 minor**。

## 5. 测试与静态检查（清单第 5 项）

- 专项 28：6 个被审测试文件 `--collect-only` 恰 28 项（observer 6 + transaction artifact 3 + budget
  artifact 3 + tenx artifact 3 + tenx budget 9 + transaction budget 4），运行 **28 passed**（2.30s）。
- 全量：`pytest tests -q -rs`（source env.sh 后）= **1860 passed, 3 skipped, 6 warnings in 706.09s**，
  与声明逐项一致；3 个 skip 为既有环境边界（1× TVM 重复编译 smoke 主动跳过、2× frozen VNN-COMP
  checkout 不可用）。
  （注：不 source env.sh 时 3 个 TVM 相关文件 collection error，属环境前置条件，非本批缺陷。）
- black：`--check` 12 个本批 Python 文件全部 unchanged（本机 black 报 "12 files would be left
  unchanged"）。
- mypy：3 个 runtime 模块 `--follow-imports=silent` **clean**；不加该 flag 时报 68 个错误，全部位于
  `boundflow/domains/interval.py` 等 14 个**既有**文件，无一归因于本批文件——changelog 的"6 个 source
  文件 clean"按文件口径成立，但仓库级 mypy 并不干净（既有问题，见 info-3）。
- pylint：3 个 runtime 模块 10.00/10（本机复核）；changelog 称 3 个 runner 亦 10.00/10。
- `git diff --check`：通过。
- `dol lint --soft`：`{"ok":true,...,"soft":true}` 通过；`.docops/ev.jsonl` 含 ev015988/ev015989 记录；
  `.docops/s.md` 状态与本批一致（next=implement-s1-canonical-cibc-vertical-path）。

结论：**通过**（含全量 1860 passed / 3 skipped，见第 11 节）。

## 6. Claim 边界与文档（清单第 6 项）

- 状态表述一致性：changelog、主计划 A.5.2、gemini_doc/README 索引、两个 artifact summary 全部一致——
  S0 attribution admitted、S1 implementation open、performance gate closed、`performance_claimed=false`。
- `BOUNDFLOW_ASPLOS27_RAPID_REVIEW_TWO_PAGE_STORY_DRAFT_2026_08_27.md`：自述
  `evidence-aware-draft-not-submission-ready`；投影写为"可证伪预算……O1—O5 target 尚未 direct 验证，S1
  performance claim 仍关闭"；历史反例数字（CIBC 12.795×/IBP 2.456×/B4-B2 4.898×/MR5 0.834×/MR6
  0.903×/B3 0.910×）与本批无关但与既有 artifact 口径一致；**未发现**把投影写成加速、把 IBP 结果写成
  BaB/query 结果或 ASPLOS-ready 暗示。
- `BOUNDFLOW_CIBC_TO_ASPLOS27_CHANGES_NOTE_DRAFT_2026_08_27.md`：明文"changes note 不得暗示新增系统
  已经实现、达到预算投影或优于 CIBC"；boundary 清晰。
- 主计划 v6（`BOUNDFLOW_README_PIPELINE_..._PLAN_2026_08_26.md` M 改动）：A.5.2 节数字与本批 raw
  重算逐项一致，O1 16× 自标"高于现有 CIBC local 12.795×……只是待证伪目标"，并承诺任何轴不达标须用
  direct cumulative 结果重算、不得保留 12.56×/11.66× 投影；A.2 明确只有 same-solver complete-query direct
  raw 才能升级 headline。
- claim-boundary 漂移评估：**未发现漂移**。

结论：**通过**。

## 7. 流程：未提交工作的审计风险（清单第 7 项）

- 风险确认：本批全部文件为 untracked/modified，source identity 不能由 commit 锚定；若工作树在生成
  artifact 后被改动，纯 commit 链无法发现。
- 缓解已就位且本机核对通过：两个 artifact 的 `manifest.json` 均含 `code_revision`（本批代码文件的
  sha256）与 `source_git_head=20f4741236fed26e7dfb6061a06d446e5b141186`；本审计逐一重算当前工作树
  5 个被绑定文件的 sha256，**全部 MATCH**；replay 本身也强制 code_revision 与当前文件一致，因此
  "artifact 绑定代码 ≡ 当前工作树"在现场成立。
- 残余风险（info）：该绑定只覆盖 `CODE_FILES` 列出的 5 个文件；测试/文档改动不会破坏 replay。建议尽快
  commit 以 commit hash 锚定本批。
- 工作树中他人遗留（GC0-1 外审产物、`docs/CIBC_for_DAC.pdf`）未触碰；本审计除本报告外未修改仓库，
  `/tmp/audit_recompute.py` 为审计脚本，不留仓内。

结论：**通过（附尽快提交的 info 建议）**。

## 8. 不可现场复核项

- 20 个 GPU worker run 的原始执行过程本身不可重放（成本与外部 venv 依赖）；本审计以 raw 记录内部一致性、
  语义 replay、逐对语义相等、compute signature 稳定与环境记录一致作为替代证据。
- αβ-CROWN/auto_LiRPA/VNN-COMP 外部 pin 的当前磁盘状态未逐一重核（worker 生成时已强制核对；blob hash
  清单在 protocol 中固定）。
- "5 对 fresh 独立进程"由 runner 结构（逐 run subprocess）推得，worker 记录不含 PID，无法事后取证。

## 9. Findings

| severity | path | evidence | advice |
|---|---|---|---|
| minor | `boundflow/runtime/asplos27_transaction_budget.py:246-261` | 投影公式无 integration-overhead 项（隐含 h=0）；主计划 A.3 定义了 h 且明确 `u+h>=0.10` 即不可达，但预算表/changelog 未在 12.5622×/11.6566× 旁标注 h=0 | 在 budget protocol `decision_rules` 与 changelog 预算表加注"投影假设 integration overhead h=0，接入成本未计入" |
| minor | `scripts/run_asplos27_s0_transaction_markers.py:995-997` | 扰动门禁作用于中位数；MNIST-FC r0 单对 1.065354 已超 5%（中位 0.998658 通过）。协议本身如此冻结，非违规 | 文档/summary 中补报 per-pair max 扰动，避免读者误以为每对都 ≤1.05 |
| info | `scripts/run_asplos27_s0_transaction_markers.py:217-221` 等 | manifest `code_revision` 仅绑定 5 个代码文件；本批未提交，锚定为文件 hash 而非 commit | 尽快 commit；或把 tests/docs 纳入绑定清单 |
| info | 仓库级 mypy | 全量 mypy 在 14 个既有文件有 68 错误（如 `boundflow/domains/interval.py:176`）；本批文件 clean | 与本次无关；changelog 的"clean"声明建议写明"按文件口径" |
| info | 环境前置 | 不 `source env.sh` 直接跑全量 pytest 会 collection error（tvm import） | 在 changelog 验证节注明环境前置（AGENTS.md 已有，仅提示复核者） |

## 10. 关键命令与输出摘录

```bash
# replay 1（transactions artifact）
python scripts/run_asplos27_s0_transaction_markers.py replay \
  --artifact-dir artifacts/asplos27-s0-transactions/official-b0-five-pair-v1
# → {"status":"replay-passed","evidence_status":"s0-explicit-transactions-admitted","pair_count":10,
#    "summary_hash":"3edaab81df10882f1421950afc5a47df80ddda1408dbbe0ee691718da6606ca0",...}

# replay 2（transaction budget artifact）
python scripts/run_asplos27_s0_transaction_budget_artifact.py replay \
  --artifact-dir artifacts/asplos27-s0-transaction-budget/official-b0-five-pair-v1
# → {"status":"replay-passed","evidence_status":"s0-transaction-budget-research-route-open",
#    "s1_implementation_open":true,"s1_performance_gate_open":false,
#    "summary_hash":"25ced926a58c354fae055264b3967c4931bd9e855818786a3fe5bfc9752c168c",...}

# 独立重算（stdlib-only，/tmp/audit_recompute.py）关键输出：
# cifar10_resnet:000: min_cov=99.6324% median_pert=0.995902
# mnistfc:2:          min_cov=99.2484% median_pert=0.998658
# cifar10_resnet:000: projected=12.562203x required_uniform=10.3087
# mnistfc:2:          projected=11.656612x required_uniform=10.7081
# manifest code_revision × 5 文件全部 MATCH

# 自建 tamper 变体
# VARIANT A（raw span 截半+全重签）→ rejected: "S0 transaction worker semantic summary differs"
# VARIANT B（改 pairs.jsonl+重签 manifest）→ rejected: "S0 transaction semantic replay differs: pairs.jsonl"

# 专项
pytest tests/test_solver_transaction_observer.py tests/test_asplos27_s0_transaction_artifact.py \
  tests/test_asplos27_s0_transaction_budget_artifact.py tests/test_asplos27_s0_tenx_artifact.py \
  tests/test_asplos27_tenx_budget.py tests/test_asplos27_transaction_budget.py -q
# → 28 passed in 2.30s（--collect-only 恰 28 项）

# 静态
black --check <12 文件>            → 12 files would be left unchanged
mypy --follow-imports=silent <3 模块> → Success: no issues found in 3 source files
pylint <3 模块>                     → 10.00/10
git diff --check                    → clean
dol lint --soft                     → {"ok":true,...,"soft":true}

# 全量
pytest tests -q -rs（source env.sh）→ 1860 passed, 3 skipped, 6 warnings in 706.09s（详见第 11 节）
```

## 11. 全量测试补记

`pytest tests -q -rs`（`source env.sh`，conda python 3.12.12）实测输出：

```text
SKIPPED [1] tests/test_artifact_phase5d_smoke.py:118: TVM is available; skip allow-no-tvm smoke to avoid duplicate compilation cost.
SKIPPED [1] tests/test_cross_axis_verification_batch_artifacts.py:70: frozen VNN-COMP checkout is unavailable
SKIPPED [1] tests/test_root_projection_floor_artifacts.py:66: frozen VNN-COMP checkout is unavailable
1860 passed, 3 skipped, 6 warnings in 706.09s (0:11:46)
```

与 changelog 声明"1860 passed, 3 skipped, 6 warnings，skip 为既有 TVM/VNN-COMP 环境边界"逐项一致，
不降级 verdict。
