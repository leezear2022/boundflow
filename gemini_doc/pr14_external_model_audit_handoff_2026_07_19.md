# BoundFlow 项目起点、ASPLOS 路线与 PR-14 现状：外部模型审计交接

> 文档日期：2026-07-19
> 文档用途：交给未参与开发的其他大模型或人工审计者，独立判断路线、实现、数字和结论是否成立。
> 当前分支：`feat/pr14-real-verification`
> 当前 HEAD：`9bc7c4b`
> 冻结基线：annotated tag `pr13-validated-reduced`，解引用 commit 为 `57a854b`
> 当前判定：PR-14B **VALIDATED-NO-GO**；PR-14C 不启动；C3 降级；下一步冻结 C1+C2 论文主线。
> 重要说明：本文是导航和审计任务书，不替代 raw artifact、manifest、代码、测试或阶段报告。

## 1. 给审计者的最短结论

BoundFlow 不是要发明新的神经网络验证算法，而是研究：当 CROWN/αβ-CROWN/BaB 产生大量相关
bound query 时，能否用 verification-aware IR、显存/方法感知 Planner、多后端执行和 query
runtime 改善物化、调度、批处理与显存行为，同时保持参考 bound computation。

项目从基础 IR、reference semantics、Task/Planner/TVM 和 artifact pipeline 逐步推进到 PR-10～14。
PR-10、PR-11、PR-12、PR-13 都产生了有效但边界受限的结果。PR-14 将这些机制接到真实
αβ-CROWN/VNN-COMP workload 后得到负结论：

- 共观察 540 个真实 `compute_bounds` 调用；
- initial phase 为 143/146 region-level eligible；
- complete-verification 的 activation-BaB 核心阶段为 **0/394 eligible**；
- initial plain-CROWN fixed replay 在 simple MLP 上保持 lower bound，但输出请求不公平；
- 在 VNN-COMP ResNet-2B 上，BoundFlow whole-query lower 与 external lower 最大差
  `796.765`，符号只对齐 3/9；
- 因此不能接入 same-solver、不能产生公平性能 claim，也不能启动 PR-14C。

这不是“项目实现全失败”。成立的结论是 C1 structured representation 与 C2 multi-backend
Planner 的 compiler infrastructure；不成立的是把 C3 扩大为“真实 complete verifier 加速”。

## 2. 这个问题是怎么开始的

### 2.1 项目的原始问题

传统 eager tensor verification execution 会把 CROWN coefficient 提前或反复展开成大 dense
tensor，也难以统一协调：

- structured coefficient 在何处保留、何处物化；
- spec/domain batch 如何组织；
- 显存预算下选择 eager、chunked、structured 或 compiled backend；
- compile/cache 成本能否摊销；
- BaB 父子 query 的状态哪些可以复用、哪些只能 warm-start、哪些必须失效。

BoundFlow 的原始系统命题是：保留 linear-bound operator 的结构与显式 materialization semantics，
让 Planner/runtime 看见 eager execution 隐藏的形状、生命周期、显存、状态和跨查询关系。

### 2.2 工程阶段与 ASPLOS PR 编号是两套视角

早期 Phase 0～6 是工程能力演进：

| 工程阶段 | 解决的问题 |
|---|---|
| Phase 0～2 | 环境、包结构、Primal IR、Torch frontend |
| Phase 3 | Python IBP reference 与 auto_LiRPA correctness baseline |
| Phase 4 | Task/Planner/Executor、Spec、TVM、ONNX 闭环 |
| Phase 5 | JSONL schema、benchmark、postprocess、artifact/AE 产线 |
| Phase 6 | CROWN、α-CROWN、αβ-CROWN、BaB 与 E2E 归因基础 |

2026-07-12 起，ASPLOS 执行计划把论文风险收敛为 Gate 0 和 PR-10～14。它不是另起一个项目，
而是利用 Phase 0～6 的代码基础，按论文 claim 和硬门禁重新组织工作。

### 2.3 本轮直接起因：新环境恢复时使用了旧状态

本轮对话从迁移到新环境、恢复仓库与依赖开始。初始判断曾沿用只到 PR-10 的旧审计，错误地把
下一步描述成 `PR-10B.2 → PR-11`，甚至考虑历史分支
`bench/pr10b2-real-bab-fixed-domain-replay`。

用户随后提供了较新的 closure 信息。重新检查 research branch、annotated tags、closure 文档与
Git 历史后确认真实状态已经是：

```text
PR-10  Representation / Materialization       DONE
PR-11  Planner                                VALIDATED-REDUCED
PR-12  Multi-backend compiler backend         VALIDATED-REDUCED
PR-13  Query runtime prototype                VALIDATED-REDUCED
PR-14  Real verification workload execution   NEXT / CURRENT
```

所以问题不是 PR-10 的历史技术判断完全错误，而是恢复入口停留在错误时间点。仓库当前文档明确
规定不能只看旧 `main`，必须检查 research branch、annotated tag 和 closure 文档。

## 3. ASPLOS 计划、目标和贡献假设

### 3.1 北极星命题

总体计划的一句话主张是：

> Preserving linear-bound operators across repeated verification queries enables a
> compiler/runtime to jointly optimize materialization, memory, and batching decisions that
> eager tensor execution cannot coordinate.

正确性术语被严格限制为：

> preserving the reference bound computation under the same floating-point semantics

当前没有 outward rounding、error envelope 或独立 proof checker，因此不能把 GPU FP32 allclose
扩大成严格实数 numerical soundness。

### 3.2 三项原始贡献假设

| 贡献 | 原始目标 | 当前状态 |
|---|---|---|
| C1 Structured Bound-Operator IR | 保留 coefficient operator 结构；显式 barrier/reason/bytes/lifetime | validated foundation |
| C2 Method-/Autograd-/Memory-Aware Planner | 联合选择物化、batch、backend、cache/recompute、storage | validated-reduced |
| C3 Verification-Aware Query Runtime | query classification、state validity、batching、capability routing、same-solver execution | reduced foundation；真实加速 claim 已降级 |

C3 的目标从未是“普通 batch engine 更快”本身。正式 baseline 必须是成熟 verifier 的公平 batched
executor，逐节点 baseline 只能用于机制诊断。

### 3.3 唯一执行顺序

```text
Gate 0 → PR-10 → PR-11 → PR-12 → PR-13 → PR-14
```

该顺序用于防止同时扩张 Planner、kernel 和 BaB runtime，导致任何结果都无法归因。

## 4. PR-10～PR-13 做了什么，为什么还需要 PR-14

| 阶段 | 目标 | 关键结果 | 不能扩大的结论 |
|---|---|---|---|
| PR-10 | structured coefficient/materialization | structured plain-CROWN 代表点 peak 约降 29.8%，但慢约 9.17×；α/αβ 出现 6 OOM | structured 不能作为统一默认表示 |
| PR-11 | selective/global materialization Planner | 1,416 次 execution；final held-out 23/23 feasible；形成 topology/liveness 与 bounded retry | 不是完整论文级最优 Planner；只到 validated-reduced |
| PR-12 | plain-CROWN fused/multi-backend execution | eager/chunked/structured/TVM fused、多预算选择；72/72 feasible opportunities | fused 不是普遍最快；compile 只在部分 repeated regime 可摊销 |
| PR-13 | state-versioned query runtime 与 same-solver adapter | query/state contract、dynamic batching、OOM bisection、same-solver reduced GPU E2E | 96.52×/9.93× 主要对逐节点；相对公平 batched original hard E2E 为 0.980× |

PR-13 证明了 query abstraction 和 runtime plumbing 可工作，但没有证明它能改善真实
complete-verification workload。尤其 α/β/split query 对 PR-12 compiled capability 不兼容，且
PR-13 non-toy/VNN-COMP 证据为空。因此 PR-14 是必须的 coverage/real-workload 止损阶段。

## 5. PR-14 的正式研究问题和冻结边界

### 5.1 三个研究问题

1. **RQ1：真实 verifier 产生什么 query？** 统计 solver phase、method、grad、α/β、split、
   spec/domain、layer pattern 和 arrival/query identity。
2. **RQ2：现有 Planner/backend 覆盖多少？** 每个 query 做 capability 判定，unsupported 必须
   fail closed，不能为了提高 coverage 静默改写模型/property。
3. **RQ3：相对公平 batched verifier 是否有系统价值？** 固定 solver/property/branch/split/
   seed/timeout/numeric policy，只替换 bound-call execution。

### 5.2 不允许改变的内容

- branch heuristic、priority queue、node order；
- α/β optimization、split/cuts、termination、timeout；
- property 语义和数值策略；
- PR-11/12/13 已冻结 artifact 与 held-out split。

### 5.3 允许替换的内容

- bound evaluation adapter；
- query packing/scheduling；
- capability-safe Planner/backend dispatch；
- 遵守 `EXACT_REUSE`、`CONDITIONAL_REUSE`、`WARM_START_ONLY`、`INVALIDATE` 的 cache/reuse。

## 6. PR-14 原计划与实际执行

### 6.1 原计划

```text
PR-14A  adapter + real query coverage
    ↓ 只有 coverage/语义门禁通过
PR-14B  real fixed-query replay + fair backend comparison
    ↓ 只有 0 correctness failure 且公平 baseline 成立
PR-14C  same-solver complete-verification E2E
```

PR-14C 不是必做项，而是 PR-14B Go 后才允许启动的条件阶段。

### 6.2 实际提交链

PR-14 分支从 `57a854b` 建立，实际有 5 个提交，不应简写成只有 4 个：

| Commit | 时间（Asia/Shanghai） | 内容 |
|---|---|---|
| `c740f30` | 2026-07-19 00:17 | 冻结 PR-14 workload execution model |
| `ba7260d` | 2026-07-19 00:58 | verification workload adapter/profile schema |
| `83dbf1b` | 2026-07-19 01:11 | 生成真实 verifier query traces |
| `71f2ff2` | 2026-07-19 01:25 | 修复新环境 tvm-ffi shared library 搜索路径 |
| `9bc7c4b` | 2026-07-19 02:27 | real initial-CROWN capture/replay、exact box、closure |

docs、adapter、traces、replay 四个主切片与原建议相符；`71f2ff2` 是在新环境中运行 TVM
replay 必需的独立环境修复。

## 7. PR-14A：真实 Query Coverage

### 7.1 实现

核心代码：

- `boundflow/runtime/verification_profile.py`
- `boundflow/runtime/abcrown_adapter.py`
- `scripts/profile_pr14_abcrown_workload.py`
- `tests/test_phase7a_pr14a_verification_profile.py`

adapter 可撤销地包装外部 `BoundedModule.compute_bounds`，不接管 solver 的 branch/split/α/β。
每个 external call 先映射到已有 PR-13 `BoundQuery`，再派生
`VerificationQueryProfile`。profile 包含：

```text
query_id / parent_query_id / sequence_number
solver_phase / bound_method / requires_grad
alpha_enabled / beta_enabled / split_state
spec_size / domain_size / layer_pattern
backend_eligible / reason_if_not / eligible_capability_ids
```

### 7.2 Workload 和版本

- αβ-CROWN commit：`e5c7e17bf0488843acb77b7519f59876717a49f4`
- auto_LiRPA commit：`5a098e8f9fb5786a428a024981d833d303921f2d`
- VNN-COMP 2021 commit：`90419aadcf06cf543ce5c1706cae1059dc9fa6cf`

| Workload | Query | Initial | Activation | Eligible | Frontend |
|---|---:|---:|---:|---:|---|
| official simple MLP | 377 | 34 | 343 | 33/377 | supported |
| official simple CNN | 1 | 1 | 0 | 0/1 | `AveragePool` fail closed |
| VNN-COMP ResNet-2B prop0 | 162 | 111 | 51 | 110/162 | supported |
| 合计 | **540** | **146** | **394** | **143/540** | mixed |

分 phase 的决策数字是：

- `alpha_crown_initialization`：143/146 region-level eligible；
- `activation_bab_bound`：**0/394 eligible**。

`143/146` 只表示 query 内存在 capability-legal affine→ReLU region，不表示整个 external
CROWN call 已能被 BoundFlow 等价替换。

### 7.3 Observer 透明性

profile-off / profile-on：

- MLP status 与 visited domains：unknown/unknown，508/508；
- CNN status：verified/verified；
- ResNet status 与 visited domains：unknown/unknown，192/192；
- query/profile 一一对应，无 duplicate/loss。

ResNet 两次独立 FP32 solver run 的 final lower 差约 `1.2e-7`；由于没有冻结逐 split lineage，
不得据此声称 branch sequence 逐项完全相同。

### 7.4 PR-14A 判定

- activation-BaB：NO-GO，不新增 α/β/split kernel；
- initial plain-CROWN：NARROW GO，只允许 MLP/ResNet fixed replay；
- CNN unsupported 必须保留，不计入 backend coverage。

## 8. PR-14B：Initial Plain-CROWN Fixed Replay

### 8.1 实现

新增或修改的关键路径：

- `boundflow/runtime/abcrown_adapter.py`：冻结真实 `x_L/x_U/C`、external bounds、method、phase、
  requested outputs，并保留同进程 external replay closure；
- `boundflow/runtime/perturbation.py`：exact per-element `BoxPerturbation`；
- `boundflow/runtime/task_executor.py`：`InputSpec.box`；
- `boundflow/runtime/bab_query.py`：box content identity 进入 query identity；
- `scripts/replay_pr14_abcrown_initial_crown.py`：external/BF/ONNX 分层门禁与 backend replay；
- `tests/test_phase7a_pr14b_abcrown_capture.py`；
- `tests/test_phase7a_pr14b_box_perturbation.py`。

runner 强制：

```text
complete_verifier=skip
bound_prop_method=crown
init_bound_prop_method=same
pgd_order=skip
```

它先检查 nominal ONNX/BF forward、同进程 external replay、BoundFlow/external bounds 和 requested
outputs，只有全部通过才允许记录 BoundFlow latency/peak memory。

### 8.2 为什么必须是 exact box

VNN-COMP ResNet property 经 clipping/normalization 后每个输入元素的区间宽度不同。正式 payload
的 width 最小 `0.05996275`、最大 `0.06442014`，共有 28 个 FP32 unique width，不能用统一
L∞ ε 代替。query identity 必须包含 lower/upper content，避免同中心不同 box 被错误复用。

### 8.3 正式 v4 结果

| Workload | Nominal BF vs ONNX | External replay | BF lower vs external | 输出契约 | 判定 |
|---|---:|---:|---:|---|---|
| simple MLP | 0 | 0 | eager/chunked/TVM 均 0 | external lower-only；BF lower+upper | 数值通过，性能 N/A |
| ResNet-2B prop0 | `1.67e-6` | `1.07e-6` | eager/chunked 最大约 `796.765` | external lower-only；BF lower+upper | bound-equivalence failure |

ResNet 9 个 robustness spec 中：external lower 6 个非负，BoundFlow lower 0 个非负，符号只一致
3/9。nominal forward 已对齐，BoundFlow eager/chunked 彼此也对齐，因此失败定位在
whole-query bound semantics 未保持，而不是简单 ONNX forward 导入错误或两个 BoundFlow backend
互相分叉。

这里不能推出 BoundFlow bounds 在数学上 unsound；能推出的是它没有保持 external reference
bound computation，会改变 verifier 的 verified/prune decision，因此不能作为透明编译替换。

### 8.4 为什么没有 BoundFlow 性能结论

- ResNet 在 bound-equivalence gate 失败；
- MLP 数值通过，但 external 请求 lower-only，BoundFlow 固定计算 lower+upper；
- v4 manifest 因此将 BoundFlow timing/peak 写为 N/A；
- earlier debug timings 不能进入 claim。

### 8.5 PR-14B/PR-14C 判定

- Activation route：NO-GO；
- Initial whole-query replacement：NO-GO；
- Fair performance claim：NO-GO；
- PR-14C：BLOCKED BY PREDECLARED GATE，不运行 full E2E 掩盖 mismatch；
- C3：DOWNGRADED 为支撑 C1/C2 的 query/state/capability infrastructure。

## 9. 环境恢复过程中解决的问题

新环境已有 TVM build，但新版 `tvm-ffi` 通过 dynamic loader 查找 shared library。原 `env.sh`
只设置 `TVM_LIBRARY_PATH`，没有把以下目录加入 `LD_LIBRARY_PATH`：

```text
boundflow/3rdparty/tvm/build-boundflow/lib
boundflow/3rdparty/tvm/build-boundflow
```

提交 `71f2ff2` 更新了 `env.sh`、staged installer、Conda hooks 和 `tests/test_env.py`。激活
`boundflow` 后 `import tvm, tvm_ffi` 通过；deactivate 恢复原环境变量。

完整测试还曾依赖 `.gitignore` 下的历史 PR-12 split artifact。`9bc7c4b` 中的测试 fixture 改为
从代码冻结的 split builder 确定性重建，因此干净 clone 不再需要旧 raw artifact。

所有测试命令必须先激活 Conda。未激活时 `scripts/run_phase6h_artifact.sh` 的裸 `python` 可能解析
到 `/usr/bin/python` 并因没有 Torch 失败；这不构成代码回归，也不能用未激活环境的失败覆盖
正式结果。

## 10. 当前 Git、测试和 artifact 状态

### 10.1 Git

撰写本文前：

```text
branch: feat/pr14-real-verification
HEAD:   9bc7c4b
base:   pr13-validated-reduced^{} = 57a854b
```

注意 annotated tag 有 tag-object SHA。`git rev-parse pr13-validated-reduced` 当前返回 tag object
`2a88e35`，应使用 `git rev-parse pr13-validated-reduced^{commit}` 得到 commit `57a854b`。

### 10.2 测试

- PR-14A/B 专属 contracts：16 passed；
- 当前完整 `pytest -q tests`：372 passed、1 skipped；
- 独立审计首次出现的 1 fail 已定位为 shell 未激活 Conda；把环境正确激活后该测试通过。

### 10.3 Raw artifact

当前机器存在：

```text
artifacts/phase7a-pr14/pr14a-real-query-trace-20260719-v2/
artifacts/phase7a-pr14/pr14b-initial-replay-20260719-v4/
```

这些目录由 `.gitignore` 的 `artifacts/` 规则排除。Git 提交的是 runner、schema、tests、manifest
解读和结论文档，不提交模型 tensor payload。clean clone 必须从 manifest 中的 upstream commit、
command、model/property hash 重生成。

当前机器的 upstream checkout 仍在：

```text
/tmp/boundflow-pr14-audit.5KB1NO/alpha-beta-CROWN
/tmp/boundflow-vnncomp2021.HjbKHs/repo
```

它们属于临时路径，不是仓库长期接口；审计时应以 commit/hash 为身份，而不是依赖路径永久存在。

## 11. 哪些结论已经成立，哪些没有

| 命题 | 判定 | 证据强度 |
|---|---|---|
| C1 能显式表示 structured coefficient/materialization | 成立的 foundation | 代码、correctness/gradient/trace contracts、PR-10 artifacts |
| C2 能在多 backend/预算下做 capability-safe 选择 | validated-reduced | frozen held-out、correctness、feasibility、regret/artifacts |
| C3 query/state/runtime abstraction 可工作 | reduced foundation | PR-13 contracts、dynamic batching、same-solver reduced experiments |
| PR-12 compiled backend 覆盖真实 activation-BaB | 不成立 | 0/394 eligible |
| BoundFlow 可透明替换 ResNet initial whole-query CROWN | 不成立 | max diff `796.765`、符号 3/9 |
| BoundFlow 相对 mature batched verifier 更快 | 未证明 | MLP output contract 不公平；ResNet equivalence fail；无合法 timing |
| PR-14C 可启动 | 不成立 | PR-14B 预设 correctness/fairness gate 失败 |
| 当前实现严格实数 numerical sound | 未声称/未证明 | 无 outward rounding 或 proof checker |

## 12. 审计边界和已知小出入

1. PR-14 实际是 5 个提交；把它称为“四个提交”会遗漏 `71f2ff2` 环境修复。
2. 任何内部 agent/goal 用时都不是 Git 可审计证据，本文不记录或引用它。
3. 最近一次独立审计已经从 raw PR-14A JSONL 重算 540、143/146、0/394，并重跑 16 个
   contract 与完整测试。
4. 最近一次独立审计核对了 PR-14B v4 manifest/payload 存在和文档数字，但没有再次执行
   replay runner。因此 PR-14B 数值当前属于“有 raw artifact/manifest 和原始运行记录支持，尚未
   被该独立审计二次重跑”。
5. PR-14A observer-on/off 没有冻结逐 split lineage，不能声称逐分支序列完全一致。
6. `backend_eligible` 是 region-level capability 结果，不等于 whole-query replacement readiness。

## 13. 建议外部模型执行的独立审计

### 13.1 Git 与路线

```bash
git status --short
git branch --show-current
git rev-parse --short 'pr13-validated-reduced^{commit}'
git log --oneline --decorate 57a854b..HEAD
git show --stat c740f30 ba7260d 83dbf1b 71f2ff2 9bc7c4b
```

预期：当前分支正确，基线为 `57a854b`，五个提交内容与表格一致；除本审计文档提交外无无关
修改。

### 13.2 PR-14A 行数与 phase/eligibility 独立统计

```bash
wc -l artifacts/phase7a-pr14/pr14a-real-query-trace-20260719-v2/*/queries.jsonl
wc -l artifacts/phase7a-pr14/pr14a-real-query-trace-20260719-v2/*/profiles.jsonl
```

预期 query/profile 分别为 1、377、162，总计各 540。

建议审计者自己解析全部 `profiles.jsonl`，按 `solver_phase` 和 `backend_eligible` 聚合，不直接抄
文档数字。预期：initial 143/146，activation 0/394。

### 13.3 测试

```bash
source /home/lee/miniconda3/etc/profile.d/conda.sh
conda activate boundflow
pytest -q \
  tests/test_phase7a_pr14a_verification_profile.py \
  tests/test_phase7a_pr14b_abcrown_capture.py \
  tests/test_phase7a_pr14b_box_perturbation.py
pytest -q tests
```

预期分别为 16 passed，以及 372 passed、1 skipped。若未激活 Conda，测试结果不具备正式环境
可比性。

### 13.4 PR-14B manifest/payload 与可选重跑

先审计：

```text
artifacts/phase7a-pr14/pr14b-initial-replay-20260719-v4/simple-mlp/manifest.json
artifacts/phase7a-pr14/pr14b-initial-replay-20260719-v4/simple-mlp/payload.pt
artifacts/phase7a-pr14/pr14b-initial-replay-20260719-v4/resnet2b-prop0/manifest.json
artifacts/phase7a-pr14/pr14b-initial-replay-20260719-v4/resnet2b-prop0/payload.pt
```

不要只确认文件存在；应检查 manifest 的 `capture`、`external.captured_replay`、`boundflow`、
`benchmark_contract`、timing/peak N/A 原因和 command/upstream commit。需要最高置信度时，在确认
临时 upstream checkout 版本和 GPU 环境后，按两个 manifest 的 `command` 字段重跑到新的
artifact 目录，不能覆盖 v4。

### 13.5 文档一致性

重点交叉检查：

1. `gemini_doc/current_status_after_pr13.md`
2. `gemini_doc/pr14_execution_plan.md`
3. `gemini_doc/pr14a_real_query_coverage_2026_07_19.md`
4. `gemini_doc/pr14b_initial_crown_fixed_replay_2026_07_19.md`
5. `gemini_doc/asplos_claims_map.md`
6. `gemini_doc/asplos_execution_memo_v1_0.md`

审计重点不是寻找所有数字是否重复，而是检查：negative evidence 是否被保留、region-level
eligibility 是否被扩大、性能 N/A 是否被偷换成 speedup、PR-14C gate 是否被绕过。

## 14. 当前下一步

PR-14 implementation 应停止。下一分支建议为：

```text
docs/asplos-c1-c2-story-freeze
```

下一阶段只做：

1. 将摘要、前两页和 claims 收敛为 C1 structured representation + C2 multi-backend Planner；
2. 把 C3 reduced positive evidence、PR-14 coverage 和 No-Go limitation 明写；
3. 用现有证据重新做 ASPLOS 2027 paper-level Go/No-Go；
4. 不回 PR-10B.2，不继续孤立 TIR 调优，不用 PR-14C E2E 绕过 correctness gate。

未来若要重启真实 verifier execution，必须提出新的研究假设，例如复用 external intermediate-bound
semantics、只替换 capability-legal region，而不是 current whole-query replacement。那应当是新
branch、新 split、新门禁，不再属于当前 PR-14。

## 15. 审计者最终应回答的问题

请独立回答以下问题，而不是只复述本文：

1. Git/tag/branch 是否证明项目确实从 PR-13 进入 PR-14，而非停在 PR-10？
2. PR-14A observer 是否透明、可撤销、无 query loss，并覆盖指定三类 workload？
3. 540、143/146、0/394 能否从 raw JSONL 独立重算？
4. `backend_eligible` 的口径是否严格保持为 region-level？
5. PR-14B 是否使用同一真实 `x_L/x_U/C`、exact box、method 和 requested-output contract？
6. ResNet mismatch 是否排除了 nominal ONNX frontend 错误，但又没有被误写成数学 unsound？
7. MLP 为什么不能产生公平 performance claim？
8. PR-14C 停止是否来自预先声明的门禁，而不是事后挑结果？
9. C1/C2/C3 当前 claim 强度是否与证据相符？
10. 下一步冻结 C1+C2 是否比回 PR-10、继续 kernel 或强行做 PR-14C 更合理？

只有以上问题均由代码、raw artifact、manifest、测试和 Git 历史支持，才能接受本文结论。
