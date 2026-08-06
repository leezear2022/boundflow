# BoundFlow 修改记录（Change Log）

## 2026-08-06：FSG1 official B0 full-stack baseline 正式关闭

- 两workload各5 fresh control/profile pairs，10/10 semantic/closure/replay通过；
- ResNet/MNIST observer perturbation median=`1.026200/1.001089<=1.05`；ResNet每profile
  234 calls、6064 visited domains，MNIST每profile 1 call并自然verified；
- summary/manifest hash=`1e5f2946…7d92`/`c9496d27…d1e`，全程
  `performance_claimed=false`；下一步FSG2 RVIR-v3 replacement correctness；
- 详细关闭记录见`gemini_doc/fsg1_fixed_iteration_control_protocol_2026_08_06.md`。

## 2026-08-06：FSG1 改用 official fixed-iteration 归因协议

- 首轮 ResNet 60 秒 control/profile 出现 `150022/150018` visited-domain timeout 漂移，正式执行被
  主动中止，未产出性能结论；
- runner 现使用 αβ-CROWN 原生 `bab/max_iterations=16` 固定求解前缀，60 秒仅作保险丝；
- 首次 fixed-16 smoke 又发现 auto batch 会随 observer 的显存状态改变，故关闭 auto enlargement；
  batch64 的单 pair observer ratio=`1.060620>1.05`，故在正式候选前冻结 batch=256，
  并固定/重置 seed；该 smoke 的 `18944/18954` 结果不采信；
- batch256 diagnostic仍有`1.075754`扰动；observer phase识别由每call `inspect.stack()`改为轻量
  `f_back`遍历，采集合同与solver执行不变；
- 轻量observer diagnostic ratio=`1.032419<=1.05`，result exact、visited domains=`[6064]`、
  profile calls=234；只准入正式采集，不形成性能claim；
- iteration budget 进入 raw protocol、pair exact gate 和 semantic replay；详细记录见
  `gemini_doc/fsg1_fixed_iteration_control_protocol_2026_08_06.md`。
- 定向`10 passed`、全量`1089 passed, 3 skipped`、Black、mypy、Pylint 10.00/10及
  `git diff --check`均通过；正式artifact仍须从提交后的clean revision生成。

## 2026-08-06：FSG1 official B0 control runner 准备完成

- 新增official control/profile worker，使用αβ-CROWN独立Python 3.11/Torch 2.11 CUDA环境；
- 观测`BoundedModule.compute_bounds`的嵌套host/CUDA event、solver phase、stream和allocator peak，
  observer退出后恢复原方法；
- 新增raw→exclusive critical-path重建、control/profile semantic exact、扰动`<=1.05`门禁及B0证据派生；
- 每个fresh worker使用独立VNNLIB临时副本，消除`.compiled`缓存造成的pair-order偏差；
- 真实`mnistfc:2` smoke语义exact、profile/control ratio约`1.0148`、1个initial-CROWN call；只作
  instrumentation准入，不形成正式性能claim；
- 定向`10 passed`，全量`1089 passed, 3 skipped`，三个新文件mypy clean、Pylint 10.00/10；
  正式五轮GPU artifact须在本代码提交后生成。

## 2026-08-06：FSG0 外部审计三项 minor 全部关闭

- 计划枚举已与代码规范统一：`alpha_optimize`、`setup/unclassified`、
  `unclassified_residual/not_applicable`不再漂移；
- FSG0测试用`cast`完成类型收窄，mypy现覆盖合同、runner与测试三个文件；
- replay新增当前`git_head`校验，同步篡改manifest hash仍会fail closed；
- 定向`20 passed`，全量`1079 passed, 3 skipped`，Black/mypy/Pylint 10.00/10与DocOps均通过；
- 外部审计结论保持`APPROVE-WITH-MINOR`原文，三项finding由executor后续修复关闭；无性能claim。

## 2026-08-06：GPU 编译器路线升级为 v1.2 full-stack overlay

- 保留 NRIR49A/G1 的正式数据和冻结 artifact，但将结论收窄为仅关闭
  selected-CROWN 单点增量优化，不再外推为 BoundFlow 全栈 GPU 上限；
- 旧 G2—G4 保留历史/gated 语义；冻结 artifact 中的 `gpu-winner-reselection` 不改写，
  但不再是当前指令；
- FSG0 schema/critical-path/replay合同已以20项定向测试和`1079 passed, 3 skipped`回归关闭；当前
  下一步是FSG1 official αβ-CROWN B0 full-stack baseline；目标是建立全栈分母，不是再找一个单点
  winner；
- 本次仅修正文档路线与 claim 边界，没有新的性能结果，不宣称 BoundFlow
  已比 auto_LiRPA/αβ-CROWN 更快；
- 详细记录：
  `gemini_doc/BOUNDFLOW_GPU_COMPILER_ACCELERATION_RESEARCH_V1_CHANGELOG_2026_08_05.md`；
  当前唯一入口：
  `gemini_doc/BOUNDFLOW_FULL_STACK_GPU_BASELINE_ATTRIBUTION_V1_PLAN_2026_08_06.md`。

## 2026-08-06：根据评审收紧 GPU 编译器计划的可证伪门禁

- G1新增精确Amdahl反解；任一scope不可达或required region speedup `>10x`时，不启动latency G3；
- benchmark矩阵硬性要求至少一个双方可solve的公开held-out workload，并在G1预判
  `B80_alloc/B80_reserved/B_OOM` physical-memory可达性；
- G8主比较冻结为RVIR exact-call合同内同一alpha-beta-CROWN host solver A/B，Planner claim缩为
  GPU-context selector；
- 增加GPU恢复/备用资源、frontend逐op覆盖和G2 qualification timebox；G1 chunk sweep保持只读；
- 修正外部审计指出的PR-12J compile phase归属，不启动TIR实现。

## 2026-08-05：新增 GPU 编译器加速诊断与执行计划

- 基于当前 IR/runtime、PR-12/13 与 NRIR43/46/47/48 证据，确认 GPU 路线值得以新的
  selected-CROWN production hypothesis 重开，但不覆盖历史 NO-GO；
- 将 selected-objective/BoundConv TIR、流程级融合、physical arena、ragged batching、multi-stream
  和条件 JIT/CUDA Graph 收敛为 G0—G8 依赖链；
- 明确本会话无法访问 NVIDIA driver/GPU，用户报告的 BoundConv `40x` 尚未由当前分支独立复现；
  40x源码缺失时禁止传播该claim但不阻塞独立GPU profiling，下一步只做环境/证据恢复和公平baseline；
- 冻结 kernel→region→child→queue→complete-query 的评估层级、benchmark matrix、artifact/replay
  合同、预注册 kill gate 和外部模型审计模板；
- 详细记录：`gemini_doc/BOUNDFLOW_GPU_COMPILER_ACCELERATION_RESEARCH_V1_PLAN_2026_08_05.md`。

## 2026-08-05：预注册 NRIR46 Template/Instance compiler IR

- Phase-B raw shards 将约 31.3 秒 trace 拆为 floor median 约 10.82 秒、两条 packed slice 各约
  9.93 秒，plan compile/rank 仅约 0.146/0.025 秒；
- diagnostic repeat0 定位 60 child prepared compile/execute 约 5.30/5.66 秒，per-child total 约
  10.98 秒；以上只用于路线选择，不是 formal claim；
- 新分支冻结 first-class `PlanTemplate/ScheduleTemplate + PlanInstance/InstanceSchedule`，不共享动态
  target ledger，不改 policy/budget/batching；
- PR #56 已在用户豁免外部 review 后合入 main；NRIR46 从最新 main 重定基后进入 Phase 0。

## 2026-08-05：用户豁免 NRIR45 外部 review

- 用户明确要求后续不再调用其他模型 review，由当前执行方持续自检推进；
- 保留 `nrir45-20260805` exchange 与审计材料，但不伪造 auditor verdict；
- PR #56 的合并门禁改为 artifact replay/tamper、targeted/full regression、静态检查与 DocOps
  确定性验证；
- claim boundary 不变：fixed ResNet2B property 0 CPU8 internal `VALIDATED-REDUCED`，final unknown。

## 2026-08-05：NRIR45 PR #56 外部审计交接

- 发布 draft PR #56，并冻结 base `b6eb697`、feature closure `8b8766e` 与 publication head
  `af1031e`；
- 新增 `gemini_doc/BOUNDFLOW_NRIR45_EXTERNAL_AUDIT_HANDOFF_2026_08_05.md`，定义 AC1—AC6、
  独立 artifact replay/tamper、全量回归与 claim boundary；
- 该交接最初用于外部审计；随后用户明确豁免执行，材料仍保留且未伪造批准。

## 2026-07-19：新增 PR-14 外部模型审计交接

- 串联项目起点、Phase 0～6、ASPLOS PR-10～14、真实 verifier coverage/replay 与最终 No-Go；
- 修正 PR-14 实际为 5 个提交，并排除不可由仓库复核的内部执行时长；
- 区分已独立重算的 PR-14A 与尚未二次重跑的 PR-14B 数值边界；
- 提供 Git、JSONL、tests、manifest/payload 和可选 replay 的逐项审计清单；
- 详细记录：`gemini_doc/pr14_external_model_audit_handoff_2026_07_19.md`。

## 2026-07-19：完整测试不再依赖 ignored PR-12 split

- 从代码冻结的 v1/v2 builder 确定性重建 test fixture；
- runner smoke 在 `tmp_path` 写 process-shareable split；
- 干净 clone 无需历史 raw artifacts 即可运行完整测试。

## 2026-07-19：PR-14B Initial Plain-CROWN Fixed Replay

- 新增 exact Box perturbation、external capture 与 real-query replay runner；
- MLP lower 等价但 requested outputs 不公平，性能 N/A；
- ResNet nominal forward 正确，但 whole-query lower max diff `796.765`、符号 3/9；
- PR-14B `VALIDATED-NO-GO`，PR-14C blocked，C3 降级为 C1/C2 基础设施；
- 详细记录：`gemini_doc/pr14b_initial_crown_fixed_replay_2026_07_19.md`。

## 2026-07-19：修复 tvm-ffi 动态库搜索路径

- 激活环境与 staged installer 同时暴露 `build-boundflow/lib` 和 `build-boundflow`；
- Conda deactivate 会恢复用户原有 `LD_LIBRARY_PATH`；
- 新增环境回归门禁，已构建的新环境无需重编 TVM。

## 2026-07-19：PR-14A 真实 Query Trace 与 Coverage 判定

- 生成官方 MLP/CNN 与 VNN-COMP ResNet-2B 的 540 个真实 `compute_bounds` profiles；
- initial phase 143/146 eligible，activation-BaB 0/394 eligible；
- observer-on/off 三组 status 与 visited-domain count 一致；CNN AveragePool frontend fail closed；
- PR-14B 仅对 initial plain-CROWN NARROW GO，activation backend replay NO-GO；
- 详细记录：`gemini_doc/pr14a_real_query_coverage_2026_07_19.md`。

## 2026-07-19：PR-14A αβ-CROWN Query Profile Adapter

- 新增由现有 `BoundQuery` 派生的 coverage profile 与聚合报告，不复制 query/state schema；
- 新增可撤销的外部 `BoundedModule.compute_bounds` observer 和官方 ONNX+VNNLIB runner；
- 真实 verifier 方法扩展为 IBP/forward/CROWN/α/αβ，并以 capability reason fail closed；
- contract 4 passed，PR-13 focused + PR-14A 19 passed，Mypy success，Pylint 10.00/10；
- 详细记录：`gemini_doc/change_2026-07-19_pr14a_abcrown_query_profile_adapter.md`。

## 2026-07-12：PR-10 以 guarded structured path 完成

- 360-row clean GPU 对照：354 ok、6 structured OOM；全量 179 passed。
- structured 消除 persistent dense，但 α/β memory guardrail 失败，因此 dense 保持默认。
- 详细记录：`gemini_doc/change_2026-07-12_complete_pr10_guarded.md`。

## 2026-07-12：增加 dense/structured ReLU profile 对照

- runner 新增双模式矩阵；严格比较 trace-off latency/peak 与 trace-on lifetime mechanism。
- 补齐 αβ fixed-split 和真实 solve_bab 搜索的 dense/structured oracle。
- 详细记录：`gemini_doc/change_2026-07-12_add_dense_structured_profile_comparison.md`。

## 2026-07-12：ReLU backward 保持 structured coefficient

- 默认返回 SignSplit operator；bias reduction 只做有 reason/site 的 ephemeral materialization。
- 保留 dense reference fallback，完成 local/full/gradient 与全量 177 passed 回归。
- 详细记录：`gemini_doc/change_2026-07-12_preserve_structured_relu_coefficients.md`。

## 2026-07-12：增加精确 SignSplitLinearOperator

- 实现 `A⁺⊙s⁺ + A⁻⊙s⁻`，禁止未经证明将 sign-split 下推穿过 matmul/conv。
- 覆盖 flat/NCHW、gradient、composition、reduction 与 ephemeral trace。
- 详细记录：`gemini_doc/change_2026-07-12_add_exact_sign_split_linear_operator.md`。

## 2026-07-12：冻结 dense ReLU backward reference oracle

- 抽出显式 dense reference，返回 `A_u/A_l/b_u/b_l`，现有路径只调用该 oracle。
- 增加 flat/NCHW、stable/unstable、α、β coefficient 与 α gradient 回归。
- 详细记录：`gemini_doc/change_2026-07-12_freeze_dense_relu_reference_oracle.md`。

## 2026-07-12：修复 αβ 多首层 halfspace 的 autograd graph 重用

- certificate optimizer 将固定 halfspace 系数与模型参数计算图分离，避免重复 backward。
- 新增双首层卷积分支、双 split halfspace 的 detector 与完整 oracle 回归。
- 详细记录：`gemini_doc/change_2026-07-12_fix_alpha_beta_halfspace_autograd_reuse.md`。

## 2026-07-12：增加 PR-10 materialization profile runner

- 新增机制与 mini-ResNet workload、CROWN/α/αβ、spec/domain fixed-batch replay 扫描。
- 严格分离 trace-on characterization 与 trace-off latency/CUDA peak。
- 详细记录：`gemini_doc/change_2026-07-12_add_pr10_materialization_profile_runner.md`。

## 2026-07-12：冻结 Materialization Trace Schema v1

- 补齐 query/event、operator tree、batch axes、logical lifetime 和 autograd/α/β 字段。
- 分离 logical bytes、allocator delta 与 CUDA allocated/reserved peak，预留其它 verifier state。
- 详细记录：`gemini_doc/change_2026-07-12_freeze_materialization_trace_schema_v1.md`。

## 2026-07-12：启动 PR-10 materialization instrumentation

- 新增 opt-in materialization trace，并标记 ReLU backward upper/lower dense barrier。
- 记录 reason/site/operator/shape/bytes/lifetime；未改变 ReLU 数学与 Planner。
- 详细记录：`gemini_doc/change_2026-07-12_pr10_materialization_instrumentation.md`。

## 2026-07-12：增加 Gate 0 reduced 环境基线

- artifact runner 新增 `reduced` 档（small matrix、warmup 3、iters 10）。
- 环境 baseline 阶段改用 reduced MLP/CNN 基线，并明确它不替代论文独立重复。
- 修正 manifest 的完整命令、dirty 状态和最终输出路径。
- 详细记录：`gemini_doc/change_2026-07-12_add_reduced_environment_baseline.md`。

## 2026-07-12：新增日常构建与运行工作流

- 固定 BoundFlow/TVM/tvm-ffi/LLVM 修改后的编译、运行、验证与禁止操作。
- 详见 `gemini_doc/boundflow_build_and_run_workflow.md`。

## 2026-07-12：定稿 ASPLOS 执行计划 v1.0

- 收敛 C1/C2/C3、状态有效性、correctness 术语、baseline、workload 与提前后的投稿门禁。
- 同步仓库指令、文档索引、协作 workflow 和 claims map。
- 详见 `gemini_doc/change_2026-07-12_finalize_asplos_execution_plan_v1_0.md`。

## 2026-07-12：新增 ASPLOS 执行备忘录 v1.0

- 锁定结构化 Bound Operator IR、query/memory Planner、BaB-oriented runtime 与 Gate 0～PR-13
  的唯一执行顺序。
- 详见 `gemini_doc/asplos_execution_memo_v1_0.md`。

## 2026-07-12：新增 ASPLOS 总体计划候选稿

- 以 Phase 7A PR-9 为实际基线，统一论文定位、三项系统贡献、PR-10～PR-15 路线、实验、
  rapid-review、Go/No-Go 与 artifact 约束。
- 文档当前供多模型/人工评审，尚未标记为最终执行版。
- 详见 `gemini_doc/boundflow_asplos_master_plan_2026_07_12.md`。

## 2026-07-13：PR-11 冻结、regret 归因与 PR-12 入口

- 新增高 regret attribution：9 个 `regret >= 1.5` case 均首先归因为 bounded candidate set
  未包含 measured oracle；backend gap 仅作为待 PR-12 验证的诊断假设。
- 新增 content-addressed evidence freeze 工具，固定 schema、split、workload/source hashes、硬件、
  commit/tag、seeds、oracle 与 regret 定义，不覆盖 PR-11 原始工件。
- 将 PR-12 收敛为无梯度 plain CROWN 的 ReLU+Linear/Conv fused TIR lowering。
- 详见 `gemini_doc/pr11_regret_attribution_2026_07_13.md` 与
  `gemini_doc/pr12_fused_crown_task_plan_2026_07_13.md`。

## 2026-07-13：PR-12 起点与 fused ReLU+Linear TIR

- 从只读 `pr11-validated-reduced` tag 创建 PR-12 分支，冻结 baseline、Planner reference 和
  `pr12-final-heldout-v1`，PR-11 的 7 个 backend-gap case 只作为 development set。
- 新增 placement/backend 二维 candidate schema 与显式 capability rejection；当前只开放
  static FP32 CUDA plain-CROWN Linear，不提前宣称 Conv 支持。
- 新增 ReLU+Linear fused TIR：直接产生 upper/lower `A_prev` 与 bias delta，不分配完整
  `A_scaled`；提供 thin Relax `call_tir` wrapper。
- 详见 `gemini_doc/change_2026-07-13_pr12_start_and_fused_linear.md`。

## 2026-07-13：PR-12 fused ReLU+Conv2d TIR foundation

- 冻结 DSCOHW/OIHW/DSCIHW layout 与显式 input-shape/output-padding contract；
- 实现 output-centric gather，覆盖 1×1/3×3、stride 1/2、padding 0/1、bias 有/无；
- CUDA matrix 对齐 upper/lower coefficient 与 bias；pre/post schedule 无 scaled-A/im2col；
- 三个 codegen 代表点 ptxas 0 stack/spill，最大 40/40/48 registers/thread；
- calibration sanity 中 stride-2 medium 仍为 1.717× slowdown，保留为 limitation；final held-out
  未使用，端到端 CROWN 尚未接入。
- 详见 `gemini_doc/change_2026-07-13_pr12_fused_conv2d.md`。

## 2026-07-12：Conda activate/deactivate 自动钩子

- 激活 `boundflow` 时自动加载 `env.sh`，反激活时完整恢复之前的路径与变量状态。
- 修复 zsh 下间接参数展开触发的 `bad substitution`，同时保留 bash 兼容性。
- 详见 `gemini_doc/change_2026-07-12_conda_activate_deactivate_hooks.md`。

## 2026-07-12：CachyOS / PyTorch 2.12.1 / CUDA 13.2 环境链路

- 分阶段重写开发安装入口，锁定 Python 3.12、CUDA 13.2、LLVM/Clang 20.1.8。
- PyTorch 使用官方 cu132 wheel；auto_LiRPA 强制 `--no-deps` 并通过 IBP/CROWN 门禁。
- TVM/Python 统一使用 TVM 内嵌 tvm-ffi；静态链接并隐藏 LLVM，解决 TVM→Triton abort。
- 新增 doctor、CUDA/TVM smoke；全量测试 162 passed、1 intentional skip。
- 修复未激活 shell 执行 `verify` 以及 hook 输出污染工具链路径探测的问题。
- 详见 `gemini_doc/change_2026-07-12_cachyos_cuda132_environment_bootstrap.md`。

约定：
- 记录按“自然批次”追加（一次明确目标的修改算一条），每条包含目的、改动点、影响面、验证方式。
- 默认在 conda 环境 `boundflow` 下验证：`conda activate boundflow`。

---

## 2025-12-17：Phase 0/1 首次止血与 IR 加固

**动机**
- 清理重复包结构，支持 `pip install -e .` 的标准安装路径。
- 将 Primal IR 升级为 Node/Value 双层结构并加入一致性校验，避免后续 planner/runtime 返工。

**主要改动**
- 工程化：新增 `pyproject.toml`，支持 `python -m pip install -e .`。
- 清理结构：删除重复/空壳目录 `boundflow/boundflow/`（避免 `boundflow.boundflow.*` 迷惑路径）。
- IR：重写 `boundflow/ir/primal.py` 为 `Node`/`Value`/`TensorType`，并提供 `BFPrimalGraph.validate()`。
- 前端壳子对齐：更新 `boundflow/frontends/pytorch/frontend.py`、`boundflow/frontends/onnx/frontend.py` 以适配新的 `BFPrimalGraph()` 构造方式；`boundflow/frontends/normalize.py` 增加 `graph.validate()`。
- 测试：新增 `tests/test_ir_primal_validate.py`；`tests/test_env.py` 增加 BoundFlow import，并打印 `CONDA_DEFAULT_ENV`（非 `boundflow` 环境提示如何运行）。
- 仓库忽略：新增 `.gitignore`（忽略 `__pycache__/`、`*.egg-info/`、`.pytest_cache/` 等）。
- 文档：新增/更新 `docs/strategy_a_refactor_plan.md`，并吸收 `docs/strategy_response_and_suggestions.md` 的观点（Node/Value、Executor、Relax、Multi-Task）。

**验证**
- 在 `boundflow` 环境：`conda run -n boundflow python tests/test_env.py`。
- IR 单测：`conda run -n boundflow python -m pytest -q tests/test_ir_primal_validate.py`。

---

## 2025-12-17：Phase 2 TorchFrontend 最小可用（torch.export → Primal IR）

**动机**
- 让仓库从“只有 IR 草图”变成“能实际导入一个 Torch 模型并得到可校验的 Primal IR”，为后续 Bound IR/Interpreter/Planner 打基础。

**主要改动**
- Torch 前端：实现 `boundflow/frontends/pytorch/frontend.py`：
  - `export_mode="export"`：使用 `torch.export.export()` 获取 FX Graph，并转换为 Primal IR（Node/Value、shape/dtype、inputs/outputs、参数占位符映射）。
  - 将常见 `aten.*` 映射到 v0.1 primitive 名称（`linear/relu/add/...`），未知 op 保留原名便于 debug。
  - 将参数 placeholder 名映射到 `ExportedProgram.state_dict`，填充 `BFPrimalProgram.params`。
- Normalizer：`boundflow/frontends/normalize.py` 增加最小规范化（`call_method::*` → `reshape/transpose` 等），并在入口处 `validate()`。
- 测试：新增 `tests/test_torch_frontend_import.py`，验证小 MLP 的 torch.export 导入、primitive 映射、输入/参数 kind、以及图校验。

**验证**
- Torch 前端单测：`conda run -n boundflow python -m pytest -q tests/test_torch_frontend_import.py`
- 回归：`conda run -n boundflow python -m pytest -q tests/test_ir_primal_validate.py`
- 环境 smoke：`conda run -n boundflow python tests/test_env.py`

---

## 2025-12-17：Phase 3 Interval IBP + PythonInterpreter（对齐 auto_LiRPA）

**动机**
- 提供一个“正确性优先、可调试”的 reference executor（fallback backend），并且用 auto_LiRPA 的 IBP 作为 ground truth 对齐，确保后续扩算子/接 TVM 时不漂。

**主要改动**
- Interval 域：新增 `boundflow/domains/interval.py`：
  - `IntervalState(lower, upper)`（torch.Tensor）
  - `IntervalDomain` 支持 `linear/relu/add/mul` 的 IBP 规则（v0.1 先覆盖 MLP 子集）
- Runtime：新增 `boundflow/runtime/executor.py`：
  - `LinfInputSpec(value_name, center, eps)`（L∞ 输入扰动）
  - `PythonInterpreter.run_ibp(program, input_spec)`：对 Primal IR 顺序执行，输出最终 output value 的 interval
  - `boundflow/runtime/__init__.py` 导出 `PythonInterpreter/LinfInputSpec`
- 测试：新增 `tests/test_phase3_ibp_against_auto_lirpa.py`：
  - 用一个小 MLP（Linear→ReLU→Linear）在同一输入与 eps 下对齐 `auto_LiRPA` 的 `compute_bounds(method='IBP')`

**验证**
- 对齐测试：`conda run -n boundflow python -m pytest -q tests/test_phase3_ibp_against_auto_lirpa.py`
- 回归：`conda run -n boundflow python -m pytest -q tests/test_ir_primal_validate.py tests/test_torch_frontend_import.py`

---

## 2025-12-17：Phase 3 扩展 Conv2d/Flatten（对齐 auto_LiRPA MNIST CNN 的 IBP）

**动机**
- 对齐 `auto_LiRPA/examples/vision/simple_verification.py` 的最简单 CNN 路径，使 v0.1 的 IBP reference 能覆盖 Conv2d，并能作为后续 TVM lowering/Planner 的 ground truth。

**主要改动**
- Torch 前端：`boundflow/frontends/pytorch/frontend.py`
  - 增加 op 映射：`aten.conv2d.default`→`conv2d`，`aten.flatten.using_ints`→`flatten`
  - 为 `conv2d/flatten` 提取常量 attrs（stride/padding/dilation/groups、start_dim/end_dim）
- Interval 域：`boundflow/domains/interval.py`
  - `affine_transformer` 增加 Conv2d IBP：权重正负分解后用 `torch.nn.functional.conv2d` 计算上下界
- Reference executor：`boundflow/runtime/executor.py`
  - `PythonInterpreter` 增加 `conv2d/flatten` 支持
  - `reshape` 由“占位”改为按 output meta shape 执行真实 `reshape`（用于 Flatten 后的线性层输入形状对齐）
- 测试：新增 `tests/test_phase3_ibp_cnn_against_auto_lirpa.py`
  - MNIST 风格 CNN（Conv→ReLU→Conv→ReLU→Flatten→Linear→ReLU→Linear）下，对齐 auto_LiRPA 的 `IBP` bounds

**验证**
- CNN 对齐：`conda run -n boundflow python -m pytest -q tests/test_phase3_ibp_cnn_against_auto_lirpa.py`
- MLP+CNN 对齐：`conda run -n boundflow python -m pytest -q tests/test_phase3_ibp_against_auto_lirpa.py tests/test_phase3_ibp_cnn_against_auto_lirpa.py`

---

## 2025-12-17：Phase 4 Task/Planner v0（把 IBP 解释执行抽象成任务）

**动机**
- 让执行路径从“逐节点解释器”过渡到“任务（Task）”形态：为后续 fusion/batching/reuse、以及 TVM Relax/TIR lowering 建立稳定接口。
- 同时避免 Domain 通过 tensor rank 猜测算子类型：由任务/算子显式携带 op 信息（例如 linear vs conv2d）。

**主要改动**
- Task IR：更新 `boundflow/ir/task.py`
  - 新增 `TaskOp`（可执行算子表示）与 `TaskKind.INTERVAL_IBP`
  - `BFTaskModule` 引入 `entry_task_id` 与 `validate()`，支持 Multi-Task 容器（v0 仍是单任务）
- Planner v0：新增 `boundflow/planner/interval_v0.py`
  - `plan_interval_ibp_v0(program)`：把整张 Primal Graph 打包成一个 `ibp_task0`
  - 对 `linear/conv2d` 的 TaskOp 写入 `attrs["op"]`，对 `reshape` 写入 `attrs["shape"]`（来自 output meta）
- Task Runtime：新增 `boundflow/runtime/task_executor.py`
  - `PythonTaskExecutor.run_ibp(BFTaskModule, LinfInputSpec)`：执行 TaskOp 序列（reference backend）
- 兼容层：重写 `boundflow/runtime/executor.py`
  - `PythonInterpreter` 保持 Phase 3 API（输入 `BFPrimalProgram`），内部改为 `plan_interval_ibp_v0` + `PythonTaskExecutor`
- Domain：`boundflow/domains/interval.py` 的 `affine_transformer` 支持 `attrs["op"]` 显式分派（并保留旧的 rank fallback）
- 测试：新增 `tests/test_phase4_task_pipeline_against_auto_lirpa.py`
  - 走 Phase 4 的 planner+task executor 路径，对齐 auto_LiRPA 的 `IBP` bounds

**验证**
- Task pipeline 对齐：`conda run -n boundflow python -m pytest -q tests/test_phase4_task_pipeline_against_auto_lirpa.py`

---

## 2025-12-17：Phase 4A Task pipeline 覆盖 CNN + permute/transpose 语义补齐

**动机**
- Phase 4 的 Task pipeline 需要覆盖从 MLP 扩展到 CNN，才能作为后续优化与 lowering 的稳定 ground truth。
- `transpose` 之前缺少维度信息，属于占位实现；真实模型中常见的 `permute` 需要被正确执行。

**主要改动**
- 新增 Phase 4 CNN 对齐测试：`tests/test_phase4_task_pipeline_cnn_against_auto_lirpa.py`
  - MNIST 风格 CNN 走 `plan_interval_ibp_v0 + PythonTaskExecutor`，对齐 auto_LiRPA `IBP` 输出。
- Torch 前端提取 permute dims：`boundflow/frontends/pytorch/frontend.py`
  - `aten.permute.default`（映射为 `transpose`）现在会写入 `attrs["dims"]`。
- Task executor 执行真实 transpose：`boundflow/runtime/task_executor.py`
  - `op_type == "transpose"` 时对 interval 的 lower/upper 执行 `permute(*dims)`，dims 缺失则显式报错。

**验证**
- Phase 4（MLP+CNN）：`conda run -n boundflow python -m pytest -q tests/test_phase4_task_pipeline_against_auto_lirpa.py tests/test_phase4_task_pipeline_cnn_against_auto_lirpa.py`

---

## 2025-12-17：Phase 4B.0 引入 StoragePlan（Memory/Storage 抽象接口钉住）

**动机**
- `docs/stage_4_critical_review.md` 指出：没有 Memory/Storage 抽象，Global Planner 很难名副其实；后续 aliasing/复用/显存峰值控制也会返工。
- v0.1 先把 schema 与默认填充方式钉住，保持现有对齐测试不受影响。

**主要改动**
- Task IR：`boundflow/ir/task.py`
  - 新增 `BufferSpec` 与 `StoragePlan`，并在 `BFTaskModule` 增加 `storage_plan` 字段与校验。
- Planner：`boundflow/planner/interval_v0.py`
  - `plan_interval_ibp_v0` 现在会填充默认 `StoragePlan`（一值一 buffer，`buf_<value_name>`）。
- 测试：新增 `tests/test_phase4b_storage_plan.py`
  - 验证 planner 生成的 module 含非空 `storage_plan`，且映射关系自洽。

**验证**
- StoragePlan 单测：`conda run -n boundflow python -m pytest -q tests/test_phase4b_storage_plan.py`
- Phase 4 对齐回归：`conda run -n boundflow python -m pytest -q tests/test_phase4_task_pipeline_against_auto_lirpa.py tests/test_phase4_task_pipeline_cnn_against_auto_lirpa.py`

---

## 2025-12-17：Phase 4B.2 Spec/Property（C 矩阵）对齐 auto_LiRPA

**动机**
- Phase 4 的计划要求补齐 Spec/Property，尤其是对齐 auto_LiRPA 的 `compute_bounds(C=..., method=...)` 语义。
- 注意：对 IBP 来说，`C` 不是简单“把 logits interval 再乘一次 C”就能对齐 auto_LiRPA；auto_LiRPA 会在最后线性层将 `C` 融合进权重/偏置，从而避免对 logits 各维独立化造成的额外松弛。

**主要改动**
- 新增 SpecIR：`boundflow/ir/spec.py`
  - `LinearSpec(C)`：C shape `[B,S,O]`，输出 shape `[B,S]`
- 新增 Planner v1：`boundflow/planner/interval_v1.py`
  - `plan_interval_ibp_with_linear_spec(program, spec)`：
    - 优先将 `C` 融合进最后 `linear`（`W' = C@W`, `b' = C@b`）以对齐 auto_LiRPA 的 IBP + C 行为
    - fallback：无法融合时追加 `spec_linear` op（语义正确但可能更松）
- Task executor：`boundflow/runtime/task_executor.py`
  - 支持 batched linear 权重（`w` rank-3 `[B,O,I]`）以执行融合后的 property
  - 保留 `spec_linear`（直接对 logits 做 C 线性组合）的执行支持
- 导出：`boundflow/planner/__init__.py`

**测试**
- 新增对齐测试：`tests/test_phase4b2_margin_c_against_auto_lirpa.py`
  - 同一模型/输入/eps 下，BoundFlow(task+spec) 输出 == auto_LiRPA `compute_bounds(C=C, method='IBP')`

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase4b2_margin_c_against_auto_lirpa.py`

---

## 2025-12-17：Phase 4C v0 TVMExecutor（Python driver + TVM kernel demo）

**动机**
- 参考 `docs/phase4_plan.md`：在不引入复杂 Relax orchestration 的前提下，先打通 TVM lowering/执行通路，证明“同一个 Task：Python reference vs TVM backend 输出一致”。

**主要改动**
- TVM kernel（interval linear）：`boundflow/backends/tvm/interval_linear.py`
  - 基于 TE → `te.create_prim_func` → `tvm.build` 生成 `interval_linear_ibp`（输入 `x_l/x_u/w/b`，输出 `y_l/y_u`）
  - 注意：本仓库的 TVM runtime 张量类型是 `tvm.runtime.Tensor`（不是 `tvm.nd.NDArray`），因此 executor 使用 `tvm.runtime._tensor.tensor/empty` 分配与拷贝
- TVM executor：`boundflow/runtime/tvm_executor.py`
  - `TVMTaskExecutor`：Python driver 顺序执行 TaskOp，v0 仅加速 `linear`（2D weight），其它 op fallback 到 torch
- 测试：`tests/test_phase4c_tvmexecutor_matches_python.py`
  - 验证 MLP 下 `TVMTaskExecutor` 输出与 `PythonTaskExecutor` 完全一致（允许浮点误差）

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase4c_tvmexecutor_matches_python.py`

---

## 2025-12-17：Phase 4B/4C 小修：permute 命名与 StoragePlan 字段占位

**动机**
- 吸收 `docs/stage_4_critical_review.md` 的建议：`permute` 不应误叫 `transpose`，否则后续 layout 分析/优化容易混淆；同时 StoragePlan 需要尽早预留后端关键字段避免返工。

**主要改动**
- 前端/规范化：`aten.permute.default` 现在映射为 `permute`（并保留旧 `transpose` 的 backward-compat）
  - `boundflow/frontends/pytorch/frontend.py`
  - `boundflow/frontends/normalize.py`
- Runtime：task executors 对 `permute/transpose` 统一执行真实 `permute(*dims)`
  - `boundflow/runtime/task_executor.py`
  - `boundflow/runtime/tvm_executor.py`
- StoragePlan schema：`BufferSpec` 增加占位字段 `device/layout/strides/alignment/alias_group`
  - `boundflow/ir/task.py`
  - 默认 planner 填充 `scope`（param/const/global）与 `layout`
  - `boundflow/planner/interval_v0.py`

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase4_task_pipeline_against_auto_lirpa.py tests/test_phase4_task_pipeline_cnn_against_auto_lirpa.py tests/test_phase4b_storage_plan.py tests/test_phase4b2_margin_c_against_auto_lirpa.py tests/test_phase4c_tvmexecutor_matches_python.py`

---

## 2025-12-17：Phase 4C 增补：auto_LiRPA vs PythonTaskExecutor vs TVMTaskExecutor 三方对齐测试

**动机**
- `tests/test_phase4c_tvmexecutor_matches_python.py` 只验证了 TVMExecutor 对齐 Python reference，但没有把 auto_LiRPA 拉进同一条链路里做端到端 sanity check。

**主要改动**
- 新增测试：`tests/test_phase4c_tvmexecutor_against_auto_lirpa.py`
  - 断言 `PythonTaskExecutor` 的 IBP 输出与 auto_LiRPA `compute_bounds(method="IBP")` 一致
  - 断言 `TVMTaskExecutor` 的输出与 `PythonTaskExecutor` 一致（从而形成三方闭环）

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase4c_tvmexecutor_against_auto_lirpa.py`

---

## 2025-12-17：测试体验修复：避免收集 3rdparty 测试 & 修正 test_env.py 的 pytest 行为

**动机**
- `pytest` 默认会递归收集 `boundflow/3rdparty/*` 下的 upstream 测试，导致大量 collection error（这些不属于 BoundFlow 的回归范围）。
- `tests/test_env.py` 原先是脚本式写法（import 时 `sys.exit`），会导致 `pytest tests` 直接在 collection 阶段失败。

**主要改动**
- 新增 `pytest.ini`
  - 将默认 `testpaths` 限制在 `tests/`
  - `norecursedirs` 排除 `boundflow/3rdparty`
- 重写 `tests/test_env.py`
  - 提供 `test_env_smoke_imports()` 作为 pytest 测试（不再在 import 时退出）
  - 保留 `python tests/test_env.py` 的脚本用法（通过 `main()` + `__main__`）

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_env.py`
- `conda run -n boundflow python -m pytest -q`

---

## 2025-12-17：Phase 4C 增补：TVM interval conv2d kernel + CNN 对齐测试 + 运行统计

**动机**
- 之前 `TVMTaskExecutor` 仅加速 `linear(w:2D)`，无法覆盖 CNN 的主要算子（`conv2d`），也不易判断到底哪些 op 走了 TVM。

**主要改动**
- TVM kernel：`boundflow/backends/tvm/interval_conv2d.py`
  - 新增 `interval_conv2d_ibp`（NCHW）用于 IBP：输入 `x_l/x_u/w/b` 输出 `y_l/y_u`
  - v0 限制：仅支持 `groups==1`（其余走 fallback）
- TVM executor：`boundflow/runtime/tvm_executor.py`
  - `conv2d` 优先走 TVM kernel（不满足条件则 fallback 到 `IntervalDomain`）
  - 新增 `last_stats`（记录本次 run 中走 TVM 的 op、fallback 的 op、以及 kernel 编译缓存命中信息）
- 导出：`boundflow/backends/tvm/__init__.py`

**测试 / 基准**
- 新增测试：`tests/test_phase4c_tvmexecutor_matches_python_cnn.py`
  - MNIST CNN 下 `TVMTaskExecutor` 输出与 `PythonTaskExecutor` 对齐，并断言至少一次 `conv2d` 走 TVM
- 新增基准脚本：`scripts/bench_phase4c_tvmexecutor.py`
  - 对比 `PythonTaskExecutor` vs `TVMTaskExecutor` 的运行耗时（以 IBP 为目标）

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase4c_tvmexecutor_matches_python_cnn.py`

---

## 2025-12-17：Phase 4B.3：layout-only `permute` 简化 pass（合并/消除）

**动机**
- `permute` 属于 layout-only op，Phase 5 做 transpose sinking/elimination 之前，需要先把“能确定消去的情况”在 planner 层钉住，避免无意义重排在后端固化成 kernel。

**主要改动**
- 新增 planner pass：`boundflow/planner/passes/layout_only.py`
  - 连续 `permute` 做组合：`permute(p1) -> permute(p2)` 合成一个 `permute(compose(p1,p2))`
  - identity `permute` 直接消除，并通过 value alias 重写后续输入/输出
  - 统一把 `transpose` 视为 `permute`（向后兼容）
- `plan_interval_ibp_v0` 默认启用该 pass：`boundflow/planner/interval_v0.py`

**测试**
- 新增：`tests/test_phase4b3_layout_permutes.py`

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase4b3_layout_permutes.py`

---

## 2025-12-17：Phase 4D：ONNX 前端最小闭环（shape_infer + Primal IR 映射）

**动机**
- Phase 5 之前需要避免“前端分叉”：Torch-export 与 ONNX-import 必须统一到同一套 Primal IR + planner/executor，才能稳定做后续优化与对齐。

**主要改动**
- ONNX frontend：`boundflow/frontends/onnx/frontend.py`
  - 支持 `onnx.shape_inference.infer_shapes`
  - 将 ONNX Graph 映射到 Primal IR（覆盖闭环子集）：`Gemm/MatMul/Conv/Relu/Add/Mul/Flatten/Reshape/Transpose/Identity/Constant`
  - `Reshape` 的 shape 必须是常量（initializer/Constant），并被固化到 `attrs["shape"]`（避免引入 shape 计算子图）
  - initializers/Constant 进入 `program.params`，并建立 `Value` meta（shape/dtype）
- 新增对齐测试：`tests/test_phase4d_onnx_frontend_matches_torch.py`
  - MLP 与 MNIST-style CNN：`Torch import` 与 `ONNX import` 在 IBP 输出上对齐（allclose）

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase4d_onnx_frontend_matches_torch.py`

---

## 2025-12-17：TVM 后端更新：默认改为 Relax 算子实现（不再手写 TE/TIR）

**动机**
- TE 已逐步不再作为 TVM 推荐的“上层入口”；希望用 Relax op 表达 kernel 逻辑，由 TVM 自行 legalize/lower（内部仍会生成 TIR，但不需要我们手写）。
- 当前仓库的 TVM runtime 没有 `tvm.nd`（使用 `tvm.runtime.Tensor`），但 Relax VM 在该 fork 下是可用的，适合做“先不用手写 TIR”的阶段性实现。

**主要改动**
- 新增 Relax kernel builder：
  - `boundflow/backends/tvm/relax_interval_linear.py`
  - `boundflow/backends/tvm/relax_interval_conv2d.py`
- `TVMTaskExecutor` 默认使用 Relax VM（可通过 `TVMExecutorOptions(kernel_style=\"te\")` 退回旧 TE demo）：
  - `boundflow/runtime/tvm_executor.py`

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase4c_tvmexecutor_matches_python.py tests/test_phase4c_tvmexecutor_matches_python_cnn.py tests/test_phase4c_tvmexecutor_against_auto_lirpa.py`

---

## 2025-12-18：Phase 5A PR#1：TaskGraph + PlanBundle/PlannerPass 骨架 + 串行 scheduler

**动机**
- 进入 Phase 5 需要把 “整图单 task” 的执行模型升级为可调度的 Task DAG（为后续 cache/reuse/batching/部分 TVM 做地基）。
- 同时需要一个可扩展的 planner 输出容器（PlanBundle）与 pass pipeline 骨架，便于系统化消融与科研迭代。

**主要改动**
- TaskGraph IR：`boundflow/ir/task_graph.py`
- Planner skeleton：`boundflow/planner/core.py`（`PlannerConfig` / `PlanBundle` / `PlannerPass`）
- BFTaskModule 扩展：`boundflow/ir/task.py`
  - 增加 `task_graph` 字段与 `get_task()`
- Scheduler：`boundflow/runtime/scheduler.py`
  - 支持按 TaskGraph topo 顺序串行执行
- PythonTaskExecutor：`boundflow/runtime/task_executor.py`
  - 增加 `run_ibp_task()`（task 级执行单元，为 scheduler 提供基础能力）

**关键调整（避免 Phase5B/5E 返工）**
- TaskGraph edge 升级为 **buffer 级依赖**（携带 `src/dst value + buffer_id`，并对齐 `StoragePlan.value_to_buffer`）
- Scheduler/env 升级为 **buffer_id -> IntervalState**（TaskIO contract 明确化）

**测试**
- `tests/test_phase5a_pr1_taskgraph_and_scheduler.py`

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5a_pr1_taskgraph_and_scheduler.py`

---

## 2025-12-18：Phase 5A PR#2：interval_v2 最小 partition（多 task DAG）+ 等价回归

**动机**
- 在不引入 cost model 的前提下，先让 planner 能输出“多 task + TaskGraph”，并验证它与 Phase 4 的单 task 行为完全等价。

**主要改动**
- 新增 v2 planner：`boundflow/planner/interval_v2.py`
  - 复用 v0 lowering（稳定的 TaskOp + StoragePlan）
  - baseline partition：layout-only（permute）单独成段，其余算子作为 compute 段；若仍不足 `min_tasks` 则按 op 数量二分
  - 生成多 `BoundTask` + `TaskGraph`（buffer 级依赖）
  - 每个 task 显式填充 TaskIO：`input_buffers` / `output_buffers`（对齐 StoragePlan）
- planner 导出：`boundflow/planner/__init__.py`
- scheduler 默认输出推断增强：`boundflow/runtime/scheduler.py`
  - 当 `output_value` 为空时，尝试根据 task_graph 推断唯一 sink task 的唯一输出；否则要求显式指定 `output_value`

**测试**
- 新增：`tests/test_phase5a_pr2_partition_multitask_equivalence.py`
  - MLP/CNN：`plan_interval_ibp_v2 + run_ibp_scheduled` 输出 == `plan_interval_ibp_v0 + PythonTaskExecutor.run_ibp`
  - 手工构造 branch+merge primal graph：确保 cross-segment use/def 在 buffer 级正确连边

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5a_pr2_partition_multitask_equivalence.py`

---

## 2025-12-18：Phase 5B PR#3：task 粒度 liveness + physical buffer reuse（v0）

**动机**
- Phase 5A 已有 multi-task DAG（TaskGraph）与 buffer 级 TaskIO contract；Phase 5B 需要把“生命周期/复用”升级为 planner 产物，为后续 cache/reuse/Relax lowering 签名做地基。

**主要改动**
- StoragePlan 支持 logical vs physical：
  - `boundflow/ir/task.py`（新增 `physical_buffers` / `logical_to_physical` / `to_physical()`）
- Liveness IR + 计算：
  - `boundflow/ir/liveness.py`（task 粒度、保守）
- Planner passes（骨架 + 可复用 helper）：
  - `boundflow/planner/passes/liveness_pass.py`
  - `boundflow/planner/passes/buffer_reuse_pass.py`
- Runtime env 改为 physical buffer id：
  - `boundflow/runtime/scheduler.py`
  - `boundflow/runtime/task_executor.py`
- interval_v2 可选开启复用（默认关闭）：
  - `boundflow/planner/interval_v2.py`（`enable_storage_reuse`）

**测试**
- `tests/test_phase5b_pr3_buffer_reuse.py`

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5b_pr3_buffer_reuse.py`

---

## 2025-12-18：Phase 5B PR#3.1：Correctness hardening（edge-driven last_use + physical env 断言）

**动机**
- 强化“env 只认 physical buffer id”与“跨 task last_use 以 TaskGraph 为准”的不变量，避免后续引入更激进复用/cache/Relax lowering 时出现隐式别名错误。

**主要改动**
- `boundflow/ir/liveness.py`：跨 task last_use 由 `TaskGraph.edges` 驱动更新
- `boundflow/runtime/scheduler.py`、`boundflow/runtime/task_executor.py`：当存在 `physical_buffers` 时强校验 env key 必须是 physical id
- `boundflow/ir/task.py`：`TaskOp.memory_effect` 占位字段（未来 alias/memory-effect 模型用）
- `boundflow/planner/passes/buffer_reuse_pass.py`：预留 `ReusePolicyFn` hook（默认 LIFO 不变）

**验证**
- `conda run -n boundflow python -m pytest -q`

---

## 2025-12-18：Phase 5B PR#4A：PlannerConfig 复用配置 + ReuseStats 可观测性

**动机**
- 在进入 5B.2（放宽 key / policy 消融 / bench）前，先把“复用参数与统计”挂到 PlannerConfig/PlanBundle，保证实验可复现并能解释 miss 原因。

**主要改动**
- `boundflow/planner/storage_reuse.py`：`StorageReuseOptions`、`ReuseKeyMode/ReusePolicy`、`BufferReuseStats`、`estimate_bytes_saved()`
- `boundflow/planner/core.py`：`PlannerConfig.storage_reuse`
- `boundflow/planner/passes/buffer_reuse_pass.py`：输出 `reuse_stats` 到 `PlanBundle.meta`
- `boundflow/planner/interval_v2.py`：透传 `reuse_key_mode/reuse_policy`（默认 STRICT/LIFO，不改变默认行为）
- `scripts/bench_storage_reuse.py`：bench/统计脚本（不放进 pytest 阈值）

**验证**
- `conda run -n boundflow python -m pytest -q`

---

## 2025-12-18：Phase 5B PR#4B：memory_effect Enum + bench 输出 + 更细 miss reasons

**动机**
- 让 `memory_effect` 类型更稳（避免字符串拼写导致的隐式分支爆炸），并把复用统计输出落地到 bench（CSV/JSON），方便后续 5B.2/5F 做消融与画图。

**主要改动**
- `boundflow/ir/task.py`：新增 `MemoryEffect` enum，`TaskOp.memory_effect` 改为 `Optional[MemoryEffect]`
- `boundflow/planner/storage_reuse.py`：新增 `StorageReuseOptions.respect_memory_effect`（占位）与 `ReuseMissReason.KEY_MISMATCH`
- `boundflow/planner/passes/buffer_reuse_pass.py`：miss reason 更细分（NO_FREE/KEY_MISMATCH/LIFETIME_OVERLAP），并记录 overlap 阻塞者 task topK 与 pool 碎片度统计
- `scripts/bench_storage_reuse.py`：支持 `--format text|json|csv` 与 `--out`，并输出 `git_commit`、版本/env vars 白名单、DAG stats 与 why-not-reused/key 分布 topK

**验证**
- `conda run -n boundflow python -m pytest -q`

---

## 2025-12-18：工程环境：默认禁用 TVM-FFI 可选 torch-c-dlpack JIT（避免 tvm import/pytest 卡住）

**动机**
- `tvm` import 会触发 `tvm-ffi` 的可选 torch-c-dlpack 扩展 JIT 编译；在部分环境下会显著拖慢甚至卡住 import，导致 pytest 超时。该扩展对当前阶段不是必需，因此默认禁用并把 cache/tmp 放入 repo。

**主要改动**
- `env.sh`
  - 默认 `TVM_FFI_DISABLE_TORCH_C_DLPACK=1`（可通过设为 0 覆盖）
  - 默认 `TVM_FFI_CACHE_DIR=$BOUNDFLOW_ROOT/.cache/tvm-ffi`
  - 默认 `TMPDIR=$BOUNDFLOW_ROOT/.tmp`

**验证**
- `conda activate boundflow && source env.sh && python -c "import tvm; print('tvm_ok')"`
- `pytest -q`

---

## 2025-12-18：Phase 5C PR#5：Planner Pipeline 统一入口 + config_dump 可复现消融

**动机**
- Phase 5 的 planner 消融需要“统一入口 + 可序列化配置快照”，否则实验不可比、不可复现。

**主要改动**
- `boundflow/planner/options.py`：结构化选项（partition/lifetime/layout/debug，占位但稳定）
- `boundflow/planner/pipeline.py`：`plan()` 统一入口，输出 `PlanBundle.meta["config_dump"]` 与 `planner_steps`
- `scripts/bench_planner_pipeline.py`：最小 pipeline bench（输出 config_dump）
- `tests/test_phase5c_pr5_pipeline_config_dump.py`：不同 config 下 task 数变化但输出等价

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5c_pr5_pipeline_config_dump.py`

---

## 2025-12-18：Phase 5C PR#6：Invariant Verifiers + Pipeline Instrument（pass contract）

**动机**
- PR#5 已统一 planner 入口与 config_dump；PR#6 进一步钉住“每一步产物是否仍合法”的 pass contract，避免后续 Relax/cache/CROWN 扩展时出现 silent wrong。

**主要改动**
- `boundflow/planner/verify.py`：TaskGraph/StoragePlan/Liveness+Reuse 三类核心不变式 verifier
- `boundflow/planner/instrument.py`：timing + verify instrument（before/after step hooks）
- `boundflow/planner/pipeline.py`：`validate_after_each_pass=True` 时每步后运行 verifier 并写入 `PlanBundle.meta["verify"]`
- `tests/test_phase5c_pr6_validators.py`：负例覆盖 broken edge/mapping/overlap + pipeline verify 记录

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5c_pr6_validators.py`

---

## 2025-12-18：Phase 5C PR#7：Determinism + DumpPlanInstrument + 结构化 VerifyError

**动机**
- 进一步钉住 planner 的可复现性（determinism）与可观测性：避免 topo 顺序/统计在不同运行漂移，并提供 step 级 JSON snapshot 便于定位 verifier 报错与 silent wrong。

**主要改动**
- `boundflow/ir/task_graph.py`：`topo_sort()` 改为 heapq 驱动的确定性顺序（按 task_id 字典序）
- `boundflow/planner/passes/buffer_reuse_pass.py`：overlap blocker 选择增加稳定 tie-break（避免 set 迭代顺序影响）
- `boundflow/planner/verify.py`：`VerifyError` 增加 `where`，关键错误填充定位信息
- `boundflow/planner/instrument.py`：`PlannerInstrument.should_run()` 预留 + `DumpPlanInstrument`（step 后 dump JSON）+ verify 输出包含 `where`
- `boundflow/planner/options.py`：`PlannerDebugOptions` 增加 `dump_plan/dump_plan_dir/dump_plan_run_id`
- `boundflow/planner/pipeline.py`：debug 开启后自动启用 DumpPlanInstrument；hook 调用统一走 should_run
- `tests/test_phase5c_pr7_determinism_and_dump.py`：新增 determinism 与 dump 回归

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5c_pr7_determinism_and_dump.py`
- `conda run -n boundflow python -m pytest -q`

---

## 2025-12-18：Phase 5D PR#8：Task → Relax IRModule lowering skeleton（interval linear）

**动机**
- 在 5C 的 pipeline contract/determinism/dump 就位后，开始补齐编译后端链路：把 interval-IBP task lower 成 Relax IRModule，为 PR#9（TVMExecutor compile+execute）做准备。

**主要改动**
- `boundflow/backends/tvm/relax_task_lowering.py`：新增 task-level lowering（`RELAX_OPS` 与 `CALL_TIR` 两种模式），v0 仅支持 single-op `linear`
- `boundflow/backends/tvm/interval_linear.py`：新增 `build_interval_linear_primfunc()`（给 `relax.call_tir` 使用）
- `tests/test_phase5d_pr8_relax_lowering_skeleton.py`：IRModule 可构建 + `relax.build(..., target="llvm")` 可编译回归

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr8_relax_lowering_skeleton.py`
- `conda run -n boundflow python -m pytest -q`

---

## 2025-12-18：Phase 5D PR#9：TVMTaskExecutor：compile cache + run_ibp_task（scheduler 对齐）

**动机**
- PR#8 已证明 Relax IRModule 可构建/可编译；PR#9 将其接到 runtime，通过 scheduler 的 physical env contract 跑通执行闭环，并与 Python reference allclose 对齐。

**主要改动**
- `boundflow/backends/tvm/relax_task_lowering.py`：新增 key 驱动的 `build_interval_linear_relax_ir_module()`（并修复 CALL_TIR 下 global_symbol 重复问题）
- `boundflow/runtime/tvm_executor.py`：实现 `run_ibp_task()`（physical env），并为 interval linear 引入编译缓存（支持 `kernel_style=relax|call_tir`）
- `tests/test_phase5d_pr9_tvm_executor_linear_equiv.py`：scheduler 下 Python vs TVM allclose 回归

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr9_tvm_executor_linear_equiv.py`
- `conda run -n boundflow python -m pytest -q`

---

## 2025-12-18：Phase 5D PR#10：TVM 编译侧可观测性（PassTimingInstrument + DumpIR）

**动机**
- planner 侧已经有 `timings_ms/config_dump/verify`；为了系统消融需要把 TVM compile 侧的 per-pass timing 与 IR dump 也纳入可观测数据，拆清楚 compile vs run 开销。

**主要改动**
- `boundflow/runtime/tvm_executor.py`：新增 compile-side 选项（pass timing / dump ir / cache tag），并在 `relax.build` 外包 `tvm.transform.PassContext(instruments=[...])`；将 compile stats 暴露为 json-able 数据
- `tests/test_phase5d_pr10_tvm_compile_instruments.py`：回归测试（timing 与 dump 落盘）

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr10_tvm_compile_instruments.py`
- `conda run -n boundflow python -m pytest -q`

---

## 2025-12-18：Phase 5D PR#11A：compute task 全 TVM（RELAX_OPS：linear+relu(+add/mul)）

**动机**
- 减少 task 内 per-op 的 Python↔TVM 往返与解释器开销；为后续 PR#11B（fusion 使 call_tir 数量下降）提供 reference 路径。

**主要改动**
- `boundflow/backends/tvm/relax_interval_task_ops.py`：新增 task-level RELAX_OPS lowering（lane 拆分契约，输出扁平 tuple）
- `boundflow/runtime/tvm_executor.py`：新增 `enable_task_relax_ops` 开关（默认 False）；开启后尝试整 task 编译/执行，失败则回退到 per-op 执行
- `tests/test_phase5d_pr11a_task_relax_ops_equiv.py`：scheduler 下 TVMTaskExecutor(RELAX_OPS) vs Python allclose

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr11a_task_relax_ops_equiv.py`
- `conda run -n boundflow python -m pytest -q`

---

## 2025-12-18：Phase 5D PR#11B：可控 fusion pipeline + call_tir 数量统计

**动机**
- 将 “RELAX_OPS → legalize/fuse” 变成可控编译开关，并把 `call_tir` 数量等 IR 结构统计落到 compile_stats，支撑后续论文级消融（调用次数下降 vs runtime）。

**主要改动**
- `boundflow/backends/tvm/relax_analysis.py`：新增 `call_tir` 计数与 IR stats
- `boundflow/runtime/tvm_executor.py`：task-level compile 支持 fusion pipeline（LegalizeOps/Annotate/FuseOps/FuseTIR），并在 `compile_stats["ir_stats"]` 记录各阶段统计
- `tests/test_phase5d_pr11a_task_relax_ops_equiv.py`：开启 fusion pipeline 回归并检查 `call_tir` 单调性（best-effort）

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr11a_task_relax_ops_equiv.py`
- `conda run -n boundflow python -m pytest -q`

---

## 2025-12-18：Phase 5D PR#11C：降低 Relax VM 调用开销（VM/PackedFunc 缓存 + VM-level passes 插槽）

**动机**
- 在 task-level RELAX_OPS + fusion pipeline 之后，进一步减少 VM/dispatch 开销，并预留可插拔的 VM-level pass 插槽，方便后续做 tuple 展开/删无用参数/inline 等消融而不改架构。

**主要改动**
- `boundflow/runtime/tvm_executor.py`：新增 VM/PackedFunc 缓存（按 `(cache_key_hash, dev.type, dev.index)`），并加入 `task_vm_opt_passes` 插槽
- `tests/test_phase5d_pr11c_vm_cache_and_opt_passes.py`：开启缓存与 pass 插槽后仍与 Python reference allclose

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr11c_vm_cache_and_opt_passes.py`
- `conda run -n boundflow python -m pytest -q`

---

## 2025-12-20：Phase 5D PR#11C.1：save_function bench + task-level pipeline 修复

**动机**
- 增加 save_function closure micro-bench（对比 VM 调用方式），并修复 task-level 自定义 relax_pipeline：必须组合 TVM 官方 `default_build_pipeline()`，否则可能触发 VM codegen 对 `alloc_tensor` 的不支持错误。

**主要改动**
- `boundflow/runtime/tvm_executor.py`：task-level compile pipeline 改为 `pre-pass + default_build_pipeline()`；fusion 统计链补上 `ConvertToDataflow`
- `boundflow/backends/tvm/relax_interval_task_ops.py`：从 StoragePlan.scope 推断 param/const，避免把 param 当作 interval 输入
- `scripts/bench_relax_vm_overhead.py`：save_function micro-bench（JSON 输出）
- `tests/test_phase5d_pr11c1_save_function_closure.py`：save_function 输出一致性回归

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr11c1_save_function_closure.py`
- `conda run -n boundflow python -m pytest -q`

---

## 2025-12-20：补充大模型协作工作流摘要

**动机**
- 将“输入计划 → 修正测试 → 总结 → 下一步计划”的交流流程固定成简明摘要，方便复用与对齐。

**主要改动**
- 更新 `gemini_doc/llm_collaboration_workflow.md`，新增“快速版：对话工作流摘要”与 6 步流程。
- 更新 `AGENTS.md` 的关键文档索引，标注工作流摘要入口。

**验证**
- 无（文档更新）

---

## 2025-12-20：Phase 5D PR#12：StaticPlanBlockMemory baseline × BoundFlow reuse（开关 + memory stats + 四象限 bench）

**动机**
- 将 TVM Relax 的 `StaticPlanBlockMemory` 作为 “intra-function 静态内存规划” baseline，与 BoundFlow 的 “inter-task logical→physical reuse” 放到同一张四象限表里，支撑后续系统消融与论文叙事。

**主要改动**
- `boundflow/runtime/tvm_executor.py`：新增 `MemoryPlanMode` 与 `TVMExecutorOptions.memory_plan_mode`，并在 task-level pipeline 中可选择跳过 `StaticPlanBlockMemory`；`compile_stats` 增加 `memory_plan_mode/memory_stats`。
- `boundflow/backends/tvm/relax_analysis.py`：新增 `collect_relax_memory_stats`，统计 `relax.vm.alloc_storage/alloc_tensor` 以及 `alloc_storage_total_bytes/max_bytes`（IR 侧估算）。
- `tests/test_phase5d_pr12_static_plan_modes.py`：DEFAULT vs DISABLE_STATIC_PLAN 两种模式下仍与 Python reference allclose。
- `scripts/bench_static_plan_baseline.py`：四象限 bench（reuse ON/OFF × static plan ON/OFF），输出 JSON 字段用于表格/画图。

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr12_static_plan_modes.py`
- `conda run -n boundflow python -m pytest -q`

---

## 2025-12-20：Phase 5D PR#12.1：memory estimator 对照 + DEFAULT pipeline 边界 + tir_var_upper_bound 占位

**动机**
- 为 PR#12 的 memory planning baseline 增加 TVM 官方 `estimate_memory_usage` 口径佐证；同时让 `MemoryPlanMode.DEFAULT` 更贴近 TVM 官方默认 pipeline，并预留 dynamic shape 上界变量（避免后续引入导致历史数据不可比）。

**主要改动**
- `boundflow/runtime/tvm_executor.py`
  - `MemoryPlanMode.DEFAULT/FORCE_STATIC_PLAN` 使用 `tvm.relax.pipeline.default_build_pipeline()`；`DISABLE_STATIC_PLAN` 仍用“等价默认但移除 StaticPlanBlockMemory”的自定义 pass 列表
  - `compile_stats["memory_stats"]` 结构调整为 `{by_scan, by_tvm_estimator}`
  - 预留 `TVMExecutorOptions.tir_var_upper_bound` 并纳入 cache key 与 compile_stats
- `tests/test_phase5d_pr12_static_plan_modes.py`：适配 `memory_stats` 新结构
- `scripts/bench_static_plan_baseline.py`：输出字段增加 `tir_var_upper_bound`，汇总字段使用 `compile_ms_total`

**验证**
- `conda run -n boundflow python -m pytest -q`

---

## 2025-12-20：Phase 5D PR#12.2：estimator_stage 固定 + 写入 tir_var_upper_bound attrs + dynamic 回归用例

**动机**
- 固定 `estimate_memory_usage` 的调用阶段并记录，避免未来 pipeline 调整导致数据漂移；同时让 `tir_var_upper_bound` 从“仅记录 options”升级为“真实写入 Relax function attrs 并可观测到效果”。
**主要改动**
- `boundflow/runtime/tvm_executor.py`
  - `compile_stats["memory_stats"]` 增加 `by_tvm_estimator_stage`（固定为 `pre_static_plan`）
  - 当 `TVMExecutorOptions.tir_var_upper_bound` 非空时，把它写入 task-level `main` Relax function 的 `tir_var_upper_bound` attrs（best-effort）
- `tests/test_phase5d_pr12_2_tir_var_upper_bound_effect.py`
  - 构造带动态维度 `n` 的 Relax module，使用 TVM `default_build_pipeline()` lowering 后对比 `collect_relax_memory_stats`：有 upper bound 时可折算出常量 bytes 且 nonconst bytes 下降

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr12_2_tir_var_upper_bound_effect.py`
- `conda run -n boundflow python -m pytest -q`

---

## 2025-12-20：Phase 5D PR#13A：Ablation matrix bench（统一 JSONL schema + 最小矩阵）

**动机**
- 在 Phase 5D 进入论文/系统化消融阶段前，先钉死统一的实验矩阵与输出 schema（JSONL/一行一条 run），避免后续每次加变量都返工 bench 字段。

**主要改动**
- `scripts/bench_ablation_matrix.py`
  - 提供 `partition/reuse/static_plan/fusion` 的 2×2×2×2 默认矩阵输出（JSONL）
  - 可选 `--matrix small` 跑单点，用于 CI/快速排查
  - 支持可选 auto_LiRPA baseline timing（默认开启，可用 `--no-auto-lirpa` 关闭）
- `tests/test_phase5d_pr13_ablation_matrix_smoke.py`
  - smoke 测试：跑 `--matrix small` 并断言输出 JSONL schema

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr13_ablation_matrix_smoke.py`
- `conda run -n boundflow python -m pytest -q`

---

## 2025-12-20：Phase 5D PR#13B：bench 计时公平性/可解释性增强 + env.sh stdout 清洁

**动机**
- 系统化消融阶段要求：stdout 可机器解析（JSONL/CSV 不被环境提示污染）、compile vs run 明确拆分、baseline 的 setup/compute 拆分、并记录差异幅度便于 debug。

**主要改动**
- `env.sh`
  - 提示信息默认写入 stderr（不污染 stdout），并支持 `BOUNDFLOW_QUIET=1` 静默。
- `boundflow/runtime/tvm_executor.py`
  - 增加 task-level compile cache 统计：`get_task_compile_cache_stats()` 返回 hit/miss/fail（用于 bench 公平性解释）。
- `scripts/bench_ablation_matrix.py`
  - 增加 `compile_first_run_ms`（首次运行/含编译触发）并保留 steady-state `run_ms_*`
  - 输出 compile cache stats；auto_LiRPA baseline 增加 `setup_ms`；correctness 增加 max abs diff 指标
- `tests/test_env_sh_quiet_stdout.py`
  - 回归：`env.sh` 默认不写 stdout，且可用 `BOUNDFLOW_QUIET=1` 静默

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_env_sh_quiet_stdout.py`
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr13_ablation_matrix_smoke.py`
- `conda run --no-capture-output -n boundflow python scripts/bench_ablation_matrix.py --matrix small --warmup 1 --iters 1 --no-auto-lirpa --no-check --output /tmp/boundflow_ablation.jsonl`

---

## 2025-12-20：Phase 5D PR#13C：JSONL schema 文档 + schema_version/time_utc + cache delta + rel diff

**动机**
- 系统化消融进入“多人协作 + 论文画图”阶段后，需要固定 bench 的 JSONL schema，并补齐去歧义字段（schema 版本、UTC 时间、cache 增量、相对误差）。

**主要改动**
- `scripts/bench_ablation_matrix.py`
  - 顶层增加 `schema_version`；`meta` 增加 `time_utc`
  - 增加 `compile_cache_stats_delta_compile_first_run`（首次运行/编译触发阶段的 cache 增量）
  - correctness 增加 `*_max_rel_diff_*`（相对误差）
- `docs/bench_jsonl_schema.md`
  - 新增 JSONL 字段/口径说明，固定输出协议（stdout 纯 payload）
- `AGENTS.md`
  - 在关键文档索引中注册 `docs/bench_jsonl_schema.md`

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr13_ablation_matrix_smoke.py`
- `conda run --no-capture-output -n boundflow python scripts/bench_ablation_matrix.py --matrix small --warmup 1 --iters 1 --no-auto-lirpa --no-check --output /tmp/boundflow_ablation.jsonl`

---

## 2025-12-20：Phase 5D PR#13D：JSONL schema contract test

**动机**
- 将 JSONL schema 从“约定”升级为 CI 级契约：逐行可解析、关键字段存在且类型/范围合理，防止后续字段漂移导致画图/后处理阶段才发现输出断裂。

**主要改动**
- `tests/test_phase5d_pr13d_bench_jsonl_schema_contract.py`
  - 运行 `bench_ablation_matrix --matrix small` 并逐行解析 JSONL，校验 schema_version/time_utc/runtime/correctness/compile cache delta 等关键字段与类型。

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr13d_bench_jsonl_schema_contract.py`

---

## 2025-12-20：Phase 5D PR#13E：postprocess 产线（JSONL → CSV/表格/图）

**动机**
- 将 bench 产物从 JSONL 进一步变成“可直接画图/做表”的数据与产线脚本，面向论文/AE 的复现与后处理。

**主要改动**
- `scripts/postprocess_ablation_jsonl.py`
  - JSONL 扁平化导出 `out/phase5d/ablation.csv`
  - 最小汇总表 `out/phase5d/tables/ablation_summary.csv`
  -（可选）示例图 `out/phase5d/figures/cache_miss_vs_compile_first_run.png`（若 matplotlib 可用）
- `tests/test_phase5d_pr13e_postprocess_jsonl.py`
  - 合成最小 JSONL 样例，验证 postprocess 输出 CSV/表格落盘
- `docs/bench_jsonl_schema.md`
  - 增加后处理脚本说明

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr13e_postprocess_jsonl.py`

---

## 2025-12-20：Phase 5D PR#13E.1：postprocess hardening（缺失值/分组/流式读取/enum 修复）

**动机**
- 修复后处理脚本的 4 类静默口径错误：missing correctness 不应当 0、group key 需包含 eps/input_shape/domain/spec、防大 JSONL 内存峰值、修正 enum repr 解析。

**主要改动**
- `scripts/postprocess_ablation_jsonl.py`
  - JSONL 流式读取（按行迭代）
  - 修正 enum 解析正则（`:\s*'value'`）
  - group key 纳入 `input_shape/eps/domain/spec` 防止混组
  - summary 不再把缺失 correctness 当 0，并输出 `python_vs_tvm_rel_diff_missing`
- `tests/test_phase5d_pr13e_postprocess_jsonl.py`
  - 新增缺失 correctness 与 group key 分组回归
- `tests/test_postprocess_enum_normalization.py`
  - 覆盖 enum repr value 解析
- `docs/bench_jsonl_schema.md`
  - 补充缺失值约定说明

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr13e_postprocess_jsonl.py`
- `conda run -n boundflow python -m pytest -q tests/test_postprocess_enum_normalization.py`

---

## 2025-12-20：Phase 5D PR#13F：bench 支持 eps/batch 覆盖（用于分组验证）

**动机**
- 为了用真实 bench 输出验证 postprocess 的 group key 不混组（eps/input_shape 变化），并为后续消融扩展预留入口，给 `bench_ablation_matrix.py` 增加最小旋钮 `--eps/--batch`。

**主要改动**
- `scripts/bench_ablation_matrix.py`
  - 新增 `--eps`（覆盖 Linf eps）与 `--batch`（覆盖输入 batch size；当前仅支持 `workload=mlp`）
- `docs/bench_jsonl_schema.md`
  - 增加“Workload 参数化（用于分组验证）”说明

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr13d_bench_jsonl_schema_contract.py`

---

## 2025-12-20：Phase 5D PR#13E.2：MANIFEST 换行修复 + --no-check 结构稳定

**动机**
- 修复 MANIFEST.txt 的可读性（避免字面量 `\\n`），并让 `--no-check` 时 JSONL 的 correctness diff 字段结构稳定（输出为 null），减少下游处理分支与口径歧义。

**主要改动**
- `scripts/postprocess_ablation_jsonl.py`
  - MANIFEST 使用真实换行符写入
- `scripts/bench_ablation_matrix.py`
  - `--no-check` 下 diff keys 仍存在（值为 null）
- `tests/test_phase5d_pr13e_postprocess_jsonl.py`
  - 回归：MANIFEST 不含字面量 `\\n`
- `tests/test_phase5d_pr13d_bench_jsonl_schema_contract.py`
  - 回归：`--no-check` 仍输出 diff keys（为 null）
- `docs/bench_jsonl_schema.md`
  - 补充 `--no-check` 的字段口径说明

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr13d_bench_jsonl_schema_contract.py`
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr13e_postprocess_jsonl.py`

---

## 2025-12-22：忽略 artifacts/out 运行产物目录

**动机**
- `artifacts/` 与 `out/` 是运行产物目录（JSONL/CSV/图/manifest），体积与内容随实验增长/变化，不适合纳入 git 版本控制；复现应由 runner/bench 重新生成。

**主要改动**
- `.gitignore`
  - 新增忽略：`artifacts/`、`out/`

**验证**
- `git status --porcelain`

---

## 2025-12-22：PR#15A/15B：baseline 外提预计算 + schema_version 冻结为 1.0

**动机**
- auto_LiRPA baseline 不依赖矩阵旋钮；将 baseline 计算外提到矩阵循环外，减少重复开销并避免点内触发带来的边界条件。
- Phase 5D 的 JSONL 字段与计时/分组口径已稳定，冻结为 `schema_version=1.0`，降低后续 Phase 6 扩展时“口径被冲掉”的风险。

**主要改动**
- `scripts/bench_ablation_matrix.py`
  - baseline 在进入矩阵循环前预计算，并在每行 JSONL 直接附加（点内不再触发 compute_bounds）。
  - `schema_version` 从 `0.1` 升级为 `1.0`。
- `docs/bench_jsonl_schema.md`
  - 更新当前版本为 `1.0`，并说明 1.0 为 Phase 5D 冻结口径。
- `tests/test_phase5d_pr13d_bench_jsonl_schema_contract.py`
  - 适配 `schema_version=1.0`。

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr13d_bench_jsonl_schema_contract.py`
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr13e_postprocess_jsonl.py tests/test_phase5d_pr14d_postprocess_baseline_dedup.py`

---

## 2025-12-22：PR#15C：TVM task-level compile cache 落盘（跨进程复用）

**动机**
- AE/大矩阵多次运行时，进程内编译缓存无法跨进程复用，可能导致重复编译耗时与算力浪费。
- 增加可选 `--tvm-cache-dir`：将 task-level RELAX_OPS 的编译产物落盘，并在下次运行直接加载，缩短 cold-start。

**主要改动**
- `boundflow/runtime/tvm_executor.py`
  - `TVMExecutorOptions` 新增：`compile_cache_dir`、`compile_cache_refresh`
  - task-level 编译：支持从磁盘 cache（`task_<hash>.so` + `task_<hash>.spec.json`）加载（best-effort）。
- `scripts/bench_ablation_matrix.py`
  - 新增 CLI：`--tvm-cache-dir`、`--tvm-cache-refresh`
  - TVM executor options 透传上述参数，并默认将 `compile_cache_tag` 设为当前 git commit（降低跨版本误命中风险）。
- `docs/bench_jsonl_schema.md`
  - 补充 `compile_cache_dir` 字段说明
- `tests/test_phase5d_pr15c_tvm_disk_cache.py`
  - 回归：同一 cache_dir 下二次运行应避免 compile miss（delta miss=0）

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase5d_pr15c_tvm_disk_cache.py`

---

## 2025-12-22：Phase 5 完成声明 + 全流程文档更新

**动机**
- Phase 5 已完成工程收口（schema/产线/基线/可观测性），需要一份面向论文/AE 的“完成声明”与固定复现入口。
- 同步修订全流程总览文档，避免与最新实现（artifact runner、workload 支持、schema_version=1.0、可选 tvm 落盘 cache）不一致。

**主要改动**
- 新增：`docs/phase5_done.md`
  - Phase 5 完成声明（覆盖范围、复现入口、DoD、已知限制、Phase 6 边界）。
- 更新：`gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md`
  - 修正 Non-goals/TODO，反映 runner/workload/口径冻结的最新状态。

**验证**
- 文档变更（无额外运行时验证）。

---

## 2025-12-22：新增 Quick Restart（IBP 边界快速复跑）

**动机**
- 目前 BoundFlow 已支持 interval IBP 的端到端路径（reference + 可选 TVM + 可选 auto_LiRPA 对照），需要一个“重新上手/快速复跑”的最短指南，降低新同学/AE 复现成本。

**主要改动**
- 新增：`gemini_doc/quick_restart_ibp.md`
  - 环境启动、自检、bench 一条命令出 JSONL、artifact runner、最小 Python API 示例。
- 更新：`gemini_doc/README.md`
  - 增加 Quick Restart 文档索引入口。

**验证**
- 文档变更（以脚本 `--help` 口径校验参数存在）。

---

## 2025-12-22：重写 Phase 5「实验产线与系统化消融」总结

**动机**
- `gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md` 中 Phase 5 小节原先偏“组件罗列”，对论文/系统结构视角的“为什么这样组织实验产线、如何支持系统化消融”表达不足。

**主要改动**
- 更新：`gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md`
  - 将 Phase 5 从“5D/5E 列表”重写为“产线叙事”：Phase 4 knob → 实验矩阵；bench→JSONL（schema 冻结）→postprocess→artifact runner；并强调 compile/run 口径分解、baseline 证据链与 contract tests。

**验证**
- 文档变更（无额外运行时验证）。

---

## 2025-12-22：重写 Phase 0~6 路线图与验收标准（风格统一）

**动机**
- `gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md` 的 Phase 0~6 小节内部风格不统一（尤其 Phase 5 的叙事风格与其它 Phase 不一致），且“系统结构解读”小节缺少可引用的序号。

**主要改动**
- 更新：`gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md`
  - 为“系统结构解读（学术视角：分层、可验证、可复现实验）”补充 `3.1` 序号。
  - 重写 Phase 0~6：统一为“目标/实现状态/关键实现/验收标准/主要产物”的模板，并对缺失证据部分以 TODO 明示。

**验证**
- 文档变更（无额外运行时验证）。

---

## 2025-12-22：gemini_doc 总导引 + AGENTS.md 索引注册

**动机**
- `gemini_doc/` 内文档数量较多，需要一个目录级导引，便于交接与快速定位关键文档入口。
- 在 `AGENTS.md` 注册导引，方便后续大模型/新贡献者按索引阅读与遵循维护规则。

**主要改动**
- 新增：`gemini_doc/README.md`
  - gemini_doc 目录导引（阅读路径、关键文档、维护规则）
- 更新：`AGENTS.md`
  - 关键文档索引新增 `gemini_doc/README.md`

**验证**
- 文档变更（无额外运行时验证）。

---

## 2025-12-22：继续优化 full pipeline 文档元信息（v2）

**动机**
- `gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md` 顶部元信息需要更便于引用与追溯（schema 版本、代码版本），且系统架构图已包含 Normalize 节点但“对应实现入口”列表缺少该索引。

**主要改动**
- 更新：`gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md`
  - 元信息改为列表，并补充 `docs/bench_jsonl_schema.md` 链接与 git short SHA。
  - “对应实现入口”补充 `normalize_primal_graph` 的实现入口（`boundflow/frontends/normalize.py`）。

**验证**
- 文档变更（无额外运行时验证）。

---

## 2025-12-22：补齐论文辩护（为何不端到端用 auto_LiRPA / 为何不直接用 TVM）

**动机**
- 论文里需要把“为什么不用 auto_LiRPA”和“为什么不直接只用 TVM”讲成 reviewer-proof 的系统分层论证，并能指向仓库证据。

**主要改动**
- 更新：`gemini_doc/why_boundflow_not_auto_lirpa_or_tvm.md`
  - 将 TVM 的定位改为“张量编译/codegen”，强调 BoundFlow 的缺失中间层贡献（verification-aware IR/Planner/Scheduler + 复现/消融/对齐契约）。
  - 新增“仓库证据索引”，便于论文/AE 引用实现与测试。
- 更新：`gemini_doc/README.md`
  - 将辩护文档加入关键索引。
- 新增：`gemini_doc/change_2025-12-22_paper_defense_why_boundflow_not_auto_lirpa_or_tvm.md`
  - 记录本次论文辩护表述的调整与证据索引补齐。

**验证**
- 文档变更（无额外运行时验证）。

---

## 2025-12-22：新增多范数输入扰动（L∞/L2/L1/L0）设计文档

**动机**
- 需要把不同扰动集合下的线性算子上/下界公式讲清楚，并给出 BoundFlow 的可扩展落地设计（不破坏现有 interval IBP 管线）。

**主要改动**
- 新增：`gemini_doc/perturbation_support_design.md`
  - `PerturbationSet` + support function 的统一设计与公式汇总（Lp 对偶范数、L0 top-k）。
  - 给出与现有 pipeline 的最小侵入式对接点与迭代路线（conv2d 可先降级再 tighten）。
- 新增：`gemini_doc/change_2025-12-22_perturbation_support_design.md`
  - 记录本次设计文档新增的动机与内容摘要。

**验证**
- 文档变更（无额外运行时验证）。

---

## 2025-12-22：新增 IBP/CROWN/αβ-CROWN/BaB 的统一设计文档

**动机**
- Phase 6 将引入更强的 LiRPA 方法族与可选 BaB；需要一个三轴解耦的系统设计，避免接口与实现路径爆炸，并明确“新增扰动/新增方法”的工作量边界。

**主要改动**
- 新增：`gemini_doc/bound_methods_and_solvers_design.md`
  - 三轴解耦：`PerturbationSet × BoundMethod/DomainState × Solver(BaB)`。
  - 方法族以可组合 stages 表达（forward/relax/backward/optimize），并给出 cache/batching 统一策略与工程原则（控制流留在 Python，TVM 专注张量核）。
- 新增：`gemini_doc/change_2025-12-22_bound_methods_and_solvers_design.md`
  - 记录本次设计文档新增的动机与内容摘要。

**验证**
- 文档变更（无额外运行时验证）。

---

## 2025-12-22：补充 bound_methods 设计中的 `concretize` 实现模式

**动机**
- `concretize` 的职责边界容易混淆；需要明确 Interval/Linear 两类状态在输入处数值化的统一模式，避免把 concretize 散落进各个 Domain 子类导致重复与组合爆炸。

**主要改动**
- 更新：`gemini_doc/bound_methods_and_solvers_design.md`
  - 在 §6 增补 `concretize` 实现模式（Interval 直接返回；Linear 在输入处调用 `PerturbationSet.concretize(A, x0)`）。
- 新增：`gemini_doc/change_2025-12-22_bound_methods_concretize_section.md`
  - 记录本次补充的动机与摘要。

**验证**
- 文档变更（无额外运行时验证）。

---

## 2025-12-22：补强 bound_methods 设计的“落地避坑清单”

**动机**
- 把“方向正确但落地会撞墙”的关键风险点固化成接口约束/DoD：`A` 不强制显式化、task contract、αβ warm-start 与 cache key、subproblem 约束结构、CachePlan 粒度、逐阶段对齐测试等。

**主要改动**
- 更新：`gemini_doc/bound_methods_and_solvers_design.md`
  - 新增 `§7 落地避坑清单（建议作为 Phase 6 的接口约束/DoD）`。
- 新增：`gemini_doc/change_2025-12-22_bound_methods_landing_pitfalls_checklist.md`
  - 记录本次补强的动机与摘要。

**验证**
- 文档变更（无额外运行时验证）。

---

## 2025-12-22：Phase 6A 起步——引入 InputSpec + LpBallPerturbation（L∞/L2）与线性 concretize

**动机**
- 将“输入扰动集合”从 Phase 5 的 `LinfInputSpec` 固化形式中解耦出来，为 Phase 6 的 CROWN/αβ-CROWN/BaB 打基础，并先实现最小可行的 `L2` 线性层 concretize。

**主要改动**
- 新增：`boundflow/runtime/perturbation.py`
  - `PerturbationSet`、`LpBallPerturbation`、`InputPerturbationState`。
- 更新：`boundflow/runtime/task_executor.py`
  - 新增 `InputSpec`，`PythonTaskExecutor.run_ibp` 支持 `L2` 输入在线性层用对偶范数公式 concretize。
- 更新：`boundflow/runtime/scheduler.py`、`boundflow/runtime/tvm_executor.py`、`boundflow/runtime/executor.py`
  - 接口允许透传 `InputSpec`，但 scheduler/TVM 当前仍仅支持 `L∞`（非 L∞ 明确报错）。
- 新增：`tests/test_phase6a_inputspec_lpball_linear.py`

**验证**
- `python -m pytest -q tests/test_phase6a_inputspec_lpball_linear.py`

---

## 2025-12-23：Phase 6B 起步——最小 CROWN-IBP（MLP: Linear+ReLU）

**动机**
- 在 Phase 6A 的扰动解耦基础上，跑通 “IBP forward + CROWN backward” 的最小闭环（先覆盖 MLP 的 `Linear+ReLU`），为后续 multi-spec batching 与 α/β/BaB 打基础。

**主要改动**
- 更新：`boundflow/runtime/perturbation.py`
  - 新增 `concretize_affine(center, A, b)`（显式张量 `A` 的最小实现）。
- 新增：`boundflow/runtime/crown_ibp.py`
  - `run_crown_ibp_mlp(...)` 最小 CROWN-IBP 执行器（single-task，`linear/relu` 子集）。
- 新增：`tests/test_phase6b_crown_ibp_mlp.py`
  - `L∞`/`L2` 采样 soundness，以及 `L∞` 下 upper bound 不劣于 IBP。
- 新增：`gemini_doc/change_2025-12-23_phase6b_crown_ibp_mlp_minimal.md`
  - 记录本次 Phase 6B 起步的动机与摘要。

**验证**
- `python -m pytest -q tests/test_phase6b_crown_ibp_mlp.py`

---

## 2025-12-23：Phase 6B 补强——L1 测试 + brute-force + multi-spec 入口

**动机**
- 用测试把 CROWN-IBP 的关键语义（对偶范数、ReLU 符号选择）钉牢，并为下一步 multi-spec batching 提供最小入口（`linear_spec_C`）。

**主要改动**
- 更新：`boundflow/runtime/task_executor.py`
  - 新增 `InputSpec.l1(...)`。
- 更新：`boundflow/runtime/crown_ibp.py`
  - `run_crown_ibp_mlp(..., linear_spec_C=...)` 多目标入口（`C: [B,S,O]`）。
- 更新：`tests/test_phase6b_crown_ibp_mlp.py`
  - 新增 `L1` 采样 soundness 与 `L∞` brute-force 网格测试。
- 新增：`gemini_doc/change_2025-12-23_phase6b_crown_ibp_mlp_hardening.md`
  - 记录本次补强的动机与摘要。

**验证**
- `python -m pytest -q tests/test_phase6b_crown_ibp_mlp.py`

---

## 2025-12-23：Phase 6C（CROWN-IBP MLP）multi-spec 真 batch——吞吐 microbench + forward 复用回归

**动机**
- Phase 6B 已将 CROWN-IBP（MLP: Linear+ReLU）的 correctness 风险钉死；Phase 6C 需要开始验证 “multi-spec 真 batch” 的系统收益，并用回归测试保证 forward IBP 不随 spec 维度重复计算。

**主要改动**
- 新增：`scripts/bench_phase6c_crown_ibp_multispec_throughput.py`
  - 比较 `run_crown_ibp_mlp(..., C:[B,S,O])`（batch）与循环 `C[:,s:s+1,:]`（serial）的 p50 耗时，并输出 JSON payload。
- 新增：`tests/test_phase6c_crown_ibp_multispec_batch.py`
  - forward 复用回归：统计 `IntervalDomain` 的 forward transformer 调用次数，断言 `S=1` 与 `S=32` 时次数相同。
- 新增：`gemini_doc/change_2025-12-23_phase6c_multispec_true_batch_microbench_and_reuse_test.md`
  - 记录本次 Phase 6C 的动机、改动与验证口径。

**验证**
- `python -m pytest -q tests/test_phase6c_crown_ibp_multispec_batch.py`
- `python scripts/bench_phase6c_crown_ibp_multispec_throughput.py --device cpu --specs-list 1,4,16,64`

---

## 2025-12-23：Phase 6C microbench 稳态增强——元信息/计时后端/串行口径说明

**动机**
- 为避免吞吐 microbench 的复现实验常见质疑点，进一步把计时与输出口径工程化钉死（warmup/iters、可选计时后端、串行 baseline 口径透明、CUDA 同步等）。

**主要改动**
- 更新：`scripts/bench_phase6c_crown_ibp_multispec_throughput.py`
  - 增加 `meta` 输出（torch 版本、计时参数、串行口径说明等），并支持 `--timer torch_benchmark`。
- 更新：`tests/test_phase6c_crown_ibp_multispec_batch.py`
  - 统一用 `torch.inference_mode()` 包裹，减少上下文差异引入的波动。
- 新增：`gemini_doc/change_2025-12-23_phase6c_microbench_stability_metadata_and_timer.md`
  - 记录本次 microbench 稳态增强的动机与摘要。

**验证**
- `python -m pytest -q tests/test_phase6c_crown_ibp_multispec_batch.py`
- `python scripts/bench_phase6c_crown_ibp_multispec_throughput.py --device cpu --timer torch_benchmark --specs-list 1,4,16,64`

---

## 2025-12-23：Phase 6D（α-CROWN MLP）起步——ReLU 下界 α 参数 + K-step 优化 + warm-start

**动机**
- 在 CROWN-IBP 的 correctness 与 multi-spec 系统收益稳定后，引入可优化的 ReLU lower relaxation（α），并用 autograd 形成最小优化闭环，为后续 BaB 节点 warm-start 做接口准备。

**主要改动**
- 新增：`boundflow/runtime/alpha_crown.py`
  - `run_alpha_crown_mlp(...)`：最小 α 优化循环（best-of，支持 warm-start）。
- 更新：`boundflow/runtime/crown_ibp.py`
  - `run_crown_ibp_mlp(..., relu_alpha=...)`：不稳定 ReLU 的 lower slope 支持按节点输入 α 参数化。
- 新增：`tests/test_phase6d_alpha_crown_mlp.py`
  - 优化不回退、warm-start 不劣、采样 soundness、梯度链路钉子。
- 新增：`scripts/bench_phase6d_alpha_opt_convergence.py`
  - 输出 α 优化轨迹（stdout JSON）。
- 新增：`gemini_doc/change_2025-12-23_phase6d_alpha_crown_mlp_alpha_opt_and_warm_start.md`

**验证**
- `python -m pytest -q tests/test_phase6d_alpha_crown_mlp.py`

---

## 2025-12-23：Phase 6E（BaB MLP）起步——split state + α-CROWN bound oracle + priority-queue driver

**动机**
- 把 split state 作为运行时一等公民接入 BaB 控制流，跑通 “节点评估 → 分支 → 剪枝/终止” 的最小闭环（MLP 链式子集）。

**主要改动**
- 新增：`boundflow/runtime/bab.py`
  - `ReluSplitState` + `solve_bab_mlp(...)`：最小 BaB driver（控制流在 Python）。
- 更新：`boundflow/runtime/crown_ibp.py`
  - forward IBP 支持 `relu_split_state` 做 best-effort 的 pre-activation 区间收缩。
- 新增：`tests/test_phase6e_bab_mlp.py`
  - split 约束收紧回归、toy complete 演示（注：该演示依赖 1D Linf 的输入域收缩补丁）。
- 新增：`gemini_doc/change_2025-12-23_phase6e_bab_mlp_split_state_and_driver.md`

**验证**
- `python -m pytest -q tests/test_phase6e_bab_mlp.py`

---

## 2025-12-23：Phase 6F PR-1（β/αβ-CROWN MLP）——αβ oracle 闭环 + feasibility 接口 + β 梯度钉子 + 非平凡空域（pairwise）

**动机**
- 先把 αβ oracle 的接口/可微闭环/可剪枝形态落地：让 BaB 能通过 `feasibility` 将空域当成一等公民剪枝，并用 β 梯度钉子避免 silent bug。

**主要改动**
- 新增：`boundflow/runtime/alpha_beta_crown.py`
  - PR-1：β 以 conservative 占位符进入计算图，并提供 `feasibility`。
- 更新：`boundflow/runtime/bab.py`
  - `BabConfig.oracle={"alpha","alpha_beta"}`：允许切换 oracle，并在 `infeasible` 时 prune。
- 新增：`tests/test_phase6f_alpha_beta_crown_pr1.py`
  - β 梯度钉子、pairwise 的非平凡空域回归。
- 新增：`gemini_doc/change_2025-12-23_phase6f_pr1_alpha_beta_oracle_beta_grad_and_infeasible.md`

**验证**
- `python -m pytest -q tests/test_phase6f_alpha_beta_crown_pr1.py`

---

## 2025-12-23：Phase 6F PR-2（β/αβ-CROWN MLP）——β 真实 split-constraint encoding + 可证空域（非 pairwise）+ BaB 1D patch 降级

**动机**
- 将 β 从 PR-1 的占位符升级为真实的 split-constraint encoding，使 BaB 的 complete 支点回到 β 本身，并加强空域识别（不仅限于 pairwise）。

**主要改动**
- 更新：`boundflow/runtime/crown_ibp.py`
  - `run_crown_ibp_mlp(..., relu_pre_add_coeff_*)`：提供对 pre-activation 的线性系数注入槽位。
- 更新：`boundflow/runtime/alpha_beta_crown.py`
  - PR-2：β 以 Lagrangian 形式注入 split 约束（`s*z>=0` ⇒ `-β*s*z`），并新增 convex-combo infeasible 证书搜索（first-layer）。
- 更新：`boundflow/runtime/bab.py`
  - 1D Linf 输入域收缩补丁改为可选开关 `use_1d_linf_input_restriction_patch=False`（默认关闭）。
- 更新：`tests/test_phase6e_bab_mlp.py`、`tests/test_phase6f_alpha_beta_crown_pr1.py`
  - 回归/DoD：非 pairwise 空域证书、BaB 在不启用 patch 时由 αβ 恢复 complete。
- 新增：`gemini_doc/change_2025-12-23_phase6f_pr2_beta_split_constraint_encoding_and_bab_patch_demoted.md`

**验证**
- `python -m pytest -q tests/test_phase6f_alpha_beta_crown_pr1.py`

---

## 2025-12-24：Phase 6G PR-1（αβ oracle）——multi-spec 真 batch 回归 + `spec_reduce` 口径固化

**动机**
- 在 Phase 6F 把 β 语义闭环钉死后，Phase 6G 开始做“系统化收益”。PR-1 先在 oracle 层把 multi-spec 真 batch 的语义与优化目标口径（mean vs worst）固定下来，避免后续性能化/缓存化返工。

**主要改动**
- 更新：`boundflow/runtime/alpha_crown.py`
  - `run_alpha_crown_mlp(..., spec_reduce={"mean","min","softmin"}, soft_tau=...)`：固化多 spec 的目标聚合口径（默认 mean 保持兼容）。
- 更新：`boundflow/runtime/alpha_beta_crown.py`
  - `run_alpha_beta_crown_mlp(..., spec_reduce=..., soft_tau=...)`：同步支持；并对 infeasible 检测增加 `m==1` 快速路径。
- 新增：`tests/test_phase6g_alpha_beta_multispec_batch.py`
  - batch vs serial 一致性、forward 复用计数（S=1 vs S=32）、multi-spec 梯度链路回归。
- 新增：`gemini_doc/change_2025-12-24_phase6g_pr1_alpha_beta_multispec_true_batch_and_spec_reduce.md`

**验证**
- `python -m pytest -q tests/test_phase6g_alpha_beta_multispec_batch.py`
- `python -m pytest -q tests/test_phase6f_alpha_beta_crown_pr1.py tests/test_phase6g_alpha_beta_multispec_batch.py`

---

## 2025-12-24：Phase 6G PR-2（BaB）——node-batch（batch pick + batch eval）回归 + 梯度隔离钉子 + 吞吐 microbench

**动机**
- 在 Phase 6F 把 β 编码语义闭环钉死后，Phase 6G PR-2 只做系统化收益：把 BaB 从 “一次评估 1 个节点” 升级到 “一次评估 K 个节点”，让 oracle 吃满 batch，提高吞吐，并用测试钉死正确性与 autograd 语义。

**主要改动**
- 更新：`boundflow/runtime/alpha_beta_crown.py`
  - node-batch 支持：`relu_split_state:[B,H]` 与 `beta:[B,H]` 的 β 注入（`_beta_to_relu_pre_add_coeff`）。
  - infeasible detector 仅对 `B==1` 启用；node-batch（`B>1`）跳过该 best-effort 检测以避免错误口径。
- 新增：`tests/test_phase6g_bab_node_batch.py`
  - `node_batch_size=1` vs `node_batch_size=4` 在 1D toy 上 verdict 一致（`proven`）。
- 新增：`tests/test_phase6g_node_batch_grad_isolation.py`
  - node-batch 梯度隔离：只对 node0 的 loss 反传时，其它 node 的梯度为 0（防串味）。
- 新增：`scripts/bench_phase6g_bab_node_batch_throughput.py`
  - microbench：对同一组 split states，对比 batched node-eval vs serial node-eval 的 p50 耗时与 speedup（stdout JSON）。
- 新增：`gemini_doc/change_2025-12-24_phase6g_pr2_bab_node_batch_eval.md`

**验证**
- `python -m pytest -q tests/test_phase6g_bab_node_batch.py`
- `python -m pytest -q tests/test_phase6g_node_batch_grad_isolation.py`

---

## 2025-12-24：Phase 6G PR-3A（BaB/αβ）——NodeEvalCache（split+config+spec）与命中回归钉子

**动机**
- 在 PR-2 完成 node-batch 后，最稳的系统化收益来自“减少重复评估”：同一 split pattern 在同一 run 内被重复 tighten/重复触达时，不应重复调用 oracle。

**主要改动**
- 更新：`boundflow/runtime/bab.py`
  - 新增 `NodeEvalCache`：以 `(module,input_spec,C,oracle_config,split_state)` 为 key 的进程内缓存。
  - 新增 `eval_bab_alpha_beta_node(...)`：封装 “cache→oracle→writeback” 的统一入口。
  - `solve_bab_mlp` 接入 cache，并在 node-batch 下支持 partial hit（batch 内部分节点直接复用）。
- 新增：`tests/test_phase6g_bab_node_eval_cache.py`
  - cache hit/miss 回归钉子（同 node 不重算；改一处 split 必 miss）。
- 新增：`gemini_doc/change_2025-12-24_phase6g_pr3a_node_eval_cache.md`

**验证**
- `python -m pytest -q tests/test_phase6g_bab_node_eval_cache.py`

---

## 2025-12-26：Phase 6G PR-4（BaB/αβ）——microbench 开关矩阵 + 计数器口径固化（可归因收益）

**动机**
- 将 Phase 6G 的系统化收益拆成可归因对照（cache / branch hint / infeasible prune），避免仅用 p50 计时导致“收益来源不清”的 reviewer 质疑。

**主要改动**
- 更新：`scripts/bench_phase6g_bab_node_batch_throughput.py`
  - 引入开关矩阵：`enable_node_eval_cache/use_branch_hint/enable_batch_infeasible_prune`。
  - stdout JSON 固化输出关键计数器：`oracle_calls/forward_trace_calls/cache_hits/cache_misses/cache_hit_rate/pruned_infeasible_count/evaluated_nodes_count`。
  - 输出结构：`rows`（每个开关组合一行）+ `meta`（设备/形状/计时口径/版本等）。
- 新增：`gemini_doc/change_2025-12-26_phase6g_pr4_microbench_switch_matrix_and_counters.md`

**验证**
- `python scripts/bench_phase6g_bab_node_batch_throughput.py --device cpu --nodes 32 --node-batch-size 8 --specs 16 --steps 0 --warmup 1 --iters 3`

---

## 2025-12-29：Phase 6H PR-1（BaB/αβ）——端到端 time-to-verify 基准（开关矩阵 + JSONL 工件）

**动机**
- 将 Phase 6G 的“node-eval throughput”归因口径升级为端到端 “time-to-verify” 闭环证据链（verdict/节点数/队列行为/计数器），并产出可直接用于论文复现实验的 JSON/JSONL 工件。

**主要改动**
- 更新：`boundflow/runtime/bab.py`
  - `BabConfig` 新增 `use_branch_hint`（默认 True），用于 E2E ablation。
  - `BabResult` 补齐端到端统计字段：`nodes_evaluated/nodes_expanded/batch_rounds/avg_batch_fill_rate`（不改变语义，仅增强可观测性）。
- 新增：`scripts/bench_phase6h_bab_e2e_time_to_verify.py`
  - 端到端 BaB bench：对 `enable_node_eval_cache/use_branch_hint/enable_batch_infeasible_prune` 做 2×2×2 开关矩阵；
  - 输出 `{meta, rows}` JSON，且支持 `--jsonl-out` 追加写入 JSONL 工件。
- 新增：`gemini_doc/change_2025-12-29_phase6h_pr1_e2e_bab_time_to_verify_bench.md`

**验证**
- `python scripts/bench_phase6h_bab_e2e_time_to_verify.py --device cpu --workload 1d_relu --oracle alpha_beta --steps 0 --max-nodes 256 --node-batch-size 32 --warmup 1 --iters 3`

---

## 2025-12-29：Phase 6H PR-1 DoD 补强——meta schema 固化 + 可比性标注 + bench schema 钉子

**动机**
- 钉死 E2E bench 的复现口径与可比性，避免 verdict 不一致时误解 speedup；并用测试长期守住输出 schema。

**主要改动**
- 更新：`scripts/bench_phase6h_bab_e2e_time_to_verify.py`
  - `rows` 增加 `comparable/note`；`batch_stats/serial_stats` 增加 `popped_nodes_total/queue_peak` 别名；`meta` 增加 `git_sha/device_name/spec_reduce/torch_num_threads`。
- 新增：`tests/test_phase6h_bench_e2e_schema.py`
- 新增：`gemini_doc/change_2025-12-29_phase6h_pr1_dod_hardening_meta_comparable_schema_test.md`

**验证**
- `python -m pytest -q tests/test_phase6h_bench_e2e_schema.py`

---

## 2025-12-29：Phase 6H PR-2——sweep 汇总 + 出图出表（JSONL → CSV/fig），闭环“可发表工件”

**动机**
- 将 6H PR-1 的 JSONL 工件升级为可批量 sweep、可汇总、可出图/出表的流水线，形成论文可直接引用的证据链。

**主要改动**
- 新增：`scripts/sweep_phase6h_e2e.py`
  - 批量运行 E2E bench 并追加 JSONL（每 run 一行 `{meta,rows}`）。
- 新增：`scripts/report_phase6h_e2e.py`
  - JSONL 展平为 CSV（switch 组合 × {batch,serial}）并生成 `summary.md`（仅以 comparable 行为主表）。
- 新增：`scripts/plot_phase6h_e2e.py`
  - 从 JSONL 自动出图（speedup、counters 对照、fill-rate 散点；需要 `matplotlib`）。
- 新增：`tests/test_phase6h_report_csv_schema.py`
- 新增：`tests/test_phase6h_plot_smoke.py`（无 `matplotlib` 则 skip）
- 新增：`gemini_doc/change_2025-12-29_phase6h_pr2_sweep_report_plot_pipeline.md`

**验证**
- `python -m pytest -q tests/test_phase6h_report_csv_schema.py`
- `python -m pytest -q tests/test_phase6h_plot_smoke.py`

---

## 2025-12-29：Phase 6H PR-3（AE/论文工件准备）——一键 runner + meta 补全 + sweep 失败记录

**动机**
- 让“复现实验”更接近 AE 交付形态：一键运行主结果；meta 补齐 OS/Python；sweep 失败不静默且可审计。

**主要改动**
- 更新：`scripts/bench_phase6h_bab_e2e_time_to_verify.py`
  - `meta` 增加 `python_version/platform`。
- 更新：`scripts/sweep_phase6h_e2e.py`
  - 增加 `--fail-fast`；失败时写入 JSONL 失败记录（`meta.run_status=error` + stderr_tail），并在结束返回非 0。
- 更新：`scripts/report_phase6h_e2e.py`
  - `summary.md` 增加失败运行区块。
- 新增：`scripts/run_phase6h_artifact.sh`
- 新增：`gemini_doc/change_2025-12-29_phase6h_pr3_artifact_runner_meta_and_sweep_failure_records.md`

**验证**
- `bash scripts/run_phase6h_artifact.sh`

---

## 2025-12-30：Phase 6H PR-4（AE 打包准备）——AE README + claims 映射 + schema_version + 环境审计

**动机**
- 将 6H 的可复现流水线升级为 AE 友好的交付口径：Kick-the-tires（≤30min）+ Claim→产物映射；并固化 `schema_version` 与 runner 环境审计信息，降低 “fresh machine” 排错成本。

**主要改动**
- 更新：`scripts/bench_phase6h_bab_e2e_time_to_verify.py`
  - `meta.schema_version="phase6h_e2e_v1"`。
- 更新：`scripts/bench_phase6g_bab_node_batch_throughput.py`
  - `meta.schema_version="phase6g_node_eval_v1"`。
- 更新：`scripts/run_phase6h_artifact.sh`
  - 输出 `env.txt/pip_freeze.txt/conda_list.txt` 环境审计信息（best-effort）。
- 新增：`gemini_doc/ae_readme_phase6h.md`
- 新增：`gemini_doc/change_2025-12-30_phase6h_pr4_ae_readme_schema_version_and_env_audit.md`

**验证**
- `python -m pytest -q tests/test_phase6h_bench_e2e_schema.py`
- `bash scripts/run_phase6h_artifact.sh /tmp/phase6h_artifact_run`

---

## 2025-12-31：Phase 6H PR-5（不动语义）——扩展 E2E bench workload suite（小型非 toy MLP）

**动机**
- 在不改 runtime 语义的前提下，把 time-to-verify 的主图从 toy 扩展到小型非 toy MLP case，使图表更接近论文主结果。

**主要改动**
- 更新：`scripts/bench_phase6h_bab_e2e_time_to_verify.py`
  - 新增 `--workload`：`mlp2d_2x16`、`mlp3d_3x32`。
  - 新增 `_make_chain_mlp(...)`：按 seed 构造可复现链式 MLP（Linear+ReLU）。
- 新增：`tests/test_phase6h_workload_suite_smoke.py`
- 新增：`gemini_doc/change_2025-12-31_phase6h_pr5_workload_suite_mlp_small.md`

**验证**
- `python -m pytest -q tests/test_phase6h_workload_suite_smoke.py`

---

## 2025-12-31：Phase 6H PR-4 补丁——runner 支持 workloads 覆盖（kick-the-tires 默认不变）

**动机**
- 同时满足 kick-the-tires（默认快）与更像论文主结果的 workload suite（可选扩展），避免强制拉长 AE 路径。

**主要改动**
- 更新：`scripts/run_phase6h_artifact.sh`
  - 增加第二参数/环境变量 `PHASE6H_WORKLOADS` 覆盖 workload 列表，默认仍为 `1d_relu`。
- 更新：`gemini_doc/ae_readme_phase6h.md`
  - 增加 workload 覆盖用法说明。
- 新增：`gemini_doc/change_2025-12-31_phase6h_pr4_runner_workloads_override.md`

**验证**
- `bash scripts/run_phase6h_artifact.sh /tmp/phase6h_artifact_run "1d_relu,mlp2d_2x16"`

---

## 2025-12-31：Phase 6 收尾 PR（测试收集卫生）——可选依赖 onnx/tvm 不再导致 collection 崩溃

**动机**
- 让 `pytest -q tests` 在缺少 `onnx/tvm` 等大依赖时仍可收集与出报告（相关用例可 skip，但不能在 collection 阶段崩溃），满足 AE/CI 的硬门槛。

**主要改动**
- 更新：`tests/test_phase4d_onnx_frontend_matches_torch.py`
  - 将 `import_onnx` 延后到 `importorskip("onnx")` 之后导入，避免无 onnx 时 collection 失败。
- 更新：`tests/test_phase5d_pr8_relax_lowering_skeleton.py`、`tests/test_phase5d_pr11c1_save_function_closure.py`、`tests/test_phase5d_pr12_2_tir_var_upper_bound_effect.py`
  - 模块级 `pytest.importorskip("tvm")` + `llvm` backend gate，避免无 tvm 时 collection 失败。
- 更新：`tests/test_env.py`
  - core imports 仅要求 `torch/boundflow`；`tvm/auto_LiRPA` 缺失则 skip。
- 更新：`gemini_doc/ae_readme_phase6h.md`
  - 增加 optional deps/skip 说明。
- 新增：`gemini_doc/change_2025-12-31_phase6h_pr6_test_collection_hygiene_optional_deps.md`

**验证**
- `pytest -q tests`

---

## 2025-12-31：Phase 6 收尾 PR（E2E 统计口径加固）——p90/p99 tail latency + timeout 计数 + schema v2

**动机**
- Phase 6 已具备 AE/论文交付形态，但 reviewer/AE 常追问 “是否只报 p50 掩盖尾部？” 与 “异常样本/timeout 是否被透明记录？”。
- 同时修复 `torch.utils.benchmark` 在 PyTorch 2.8+ 下的字段差异（`Measurement.number` 不存在）。

**主要改动**
- 更新：`scripts/bench_phase6h_bab_e2e_time_to_verify.py`
  - `meta.schema_version` 升级为 `phase6h_e2e_v2`。
  - 每个开关组合输出 `batch/serial` 的 `p50/p90/p99`、`runs/valid_runs/timeouts` 与 `speedup_p90/speedup_p99`。
  - 新增 `--timeout-s`（perf_counter best-effort）与 `--torch-benchmark-repeats`（估计 p90/p99）。
- 更新：`scripts/report_phase6h_e2e.py`
  - CSV/summary 同步新增 `p90/p99` 与 run 计数字段；summary 主表展示 `p50+p90`。
  - 修复：sweep 失败记录（`rows=[]`）也能在 `summary.md` 显示。
- 更新：`scripts/plot_phase6h_e2e.py`
  - 新增 `*_speedup_p90.png`（p90 speedup 图）。
- 更新：`scripts/bench_phase6c_crown_ibp_multispec_throughput.py`、`scripts/bench_phase6g_bab_node_batch_throughput.py`
  - `torch_benchmark` 计时元信息改用 `Measurement.number_per_run`（兼容 PyTorch 2.8+）。
- 更新：`tests/test_phase6h_bench_e2e_schema.py`、`tests/test_phase6h_report_csv_schema.py`
  - schema 钉子同步覆盖新增字段。
- 更新：`gemini_doc/ae_readme_phase6h.md`
  - Claim/产物映射同步纳入 `p90` 与 `*_speedup_p90.png`。
- 新增：`gemini_doc/change_2025-12-31_phase6h_pr7_e2e_tail_latency_p90_timeout_schema_v2.md`

**验证**
- `python -m pytest -q tests/test_phase6h_bench_e2e_schema.py tests/test_phase6h_report_csv_schema.py tests/test_phase6h_plot_smoke.py`

---

## 2025-12-31：Phase 6 收官文档——新增 Phase 6 总结（phase6_summary.md）

**动机**
- Phase 6 的细节已分散记录在多份 `gemini_doc/change_*.md` 与 `docs/change_log.md` 中，但缺少一份“横向串联”的收官总结，便于论文/答辩叙事与研发接手。

**主要改动**
- 新增：`gemini_doc/phase6_summary.md`
  - 汇总 Phase 6 的目标/计划基线（三轴解耦 + stage pipeline）、6A→6H 里程碑、关键代码落点、DoD/回归钉子、可复现工件链与已知限制。
- 更新：`gemini_doc/README.md`
  - 将 `gemini_doc/phase6_summary.md` 纳入“关键交付文档”索引。
- 新增：`gemini_doc/change_2025-12-31_phase6_summary_doc.md`

**验证**
- 文档变更（无额外运行时验证）。

---

## 2025-12-31：Phase 4 收官文档——新增 Phase 4 总结（phase4_summary.md）

**动机**
- Phase 4 的细节已记录在 `docs/change_log.md` 与 `gemini_doc/change_2025-12-17_phase4*.md`，但缺少一份“横向串联”的总结文档，便于论文/答辩与工程接手。

**主要改动**
- 新增：`gemini_doc/phase4_summary.md`
  - 汇总 Phase 4 的目标/完成定义、4/4A/4B/4C/4D 里程碑、关键代码落点与回归钉子（含可选依赖说明）。
- 更新：`gemini_doc/README.md`
  - 将 `gemini_doc/phase4_summary.md` 纳入索引。
- 新增：`gemini_doc/change_2025-12-31_phase4_summary_doc.md`

**验证**
- 文档变更（无额外运行时验证）。

---

## 2025-12-31：Phase 5 收官文档——新增 Phase 5 总结（phase5_summary.md）

**动机**
- Phase 5 的交付重点是“可复现评测产线 + schema_version=1.0 冻结”，细节分散在 `docs/phase5_done.md`、`docs/bench_jsonl_schema.md`、`gemini_doc/artifact_claims_phase5d.md` 与 `docs/change_log.md` 的 Phase 5 系列条目中；补一份总结便于论文叙事与工程接手。

**主要改动**
- 新增：`gemini_doc/phase5_summary.md`
  - 总结 Phase 5 的 bench→JSONL→postprocess→artifact 产线闭环、TVM/Relax 可观测性与消融矩阵、schema contract tests、复现入口与已知限制。
- 更新：`gemini_doc/README.md`
  - 将 `gemini_doc/phase5_summary.md` 纳入索引。
- 新增：`gemini_doc/change_2025-12-31_phase5_summary_doc.md`

**验证**
- 文档变更（无额外运行时验证）。

---

## 2025-12-31：Phase 3 收官文档——新增 Phase 3 总结（phase3_summary.md）

**动机**
- Phase 3 的交付核心是 “Interval IBP reference + auto_LiRPA 对齐（MLP/CNN）”，作为后续 Phase 4/5/6 的 correctness 地基；补一份总结便于论文叙事与工程接手。

**主要改动**
- 新增：`gemini_doc/phase3_summary.md`
  - 总结 Phase 3 的目标/完成定义、里程碑（MLP→CNN 扩展）、关键代码落点与回归钉子。
- 更新：`gemini_doc/README.md`
  - 将 `gemini_doc/phase3_summary.md` 纳入索引。
- 新增：`gemini_doc/change_2025-12-31_phase3_summary_doc.md`

**验证**
- 文档变更（无额外运行时验证）。

---

## 2025-12-31：Phase 2 收官文档——新增 Phase 2 总结（phase2_summary.md）

**动机**
- Phase 2 的交付核心是 “TorchFrontend：torch.export → Primal IR + 最小 normalize 起步”，作为后续 Phase 3/4/5/6 的前端地基；补一份总结便于论文叙事与工程接手。

**主要改动**
- 新增：`gemini_doc/phase2_summary.md`
  - 总结 Phase 2 的目标/完成定义、关键代码落点（`frontends/pytorch/frontend.py`、`frontends/normalize.py`）与回归钉子。
- 更新：`gemini_doc/README.md`
  - 将 `gemini_doc/phase2_summary.md` 纳入索引。
- 新增：`gemini_doc/change_2025-12-31_phase2_summary_doc.md`

**验证**
- 文档变更（无额外运行时验证）。

---

## 2025-12-31：Phase 1 收官文档——新增 Phase 1 总结（phase1_summary.md）

**动机**
- Phase 1（总账中以 Phase 0/1 合并记录）的交付核心是 “工程止血 + Primal IR 加固（Node/Value + validate）”，作为后续 Phase 2/3/4/5/6 的工程/IR 地基；补一份总结便于论文叙事与工程接手。

**主要改动**
- 新增：`gemini_doc/phase1_summary.md`
  - 总结工程止血（editable install/包结构清理）与 Primal IR 加固（Node/Value + validate）以及最小回归钉子。
- 更新：`gemini_doc/README.md`
  - 将 `gemini_doc/phase1_summary.md` 纳入索引。
- 新增：`gemini_doc/change_2025-12-31_phase1_summary_doc.md`

**验证**
- 文档变更（无额外运行时验证）。

---

## 2025-12-31：Phase 0 收官文档——新增 Phase 0 总结（phase0_summary.md）

**动机**
- Phase 0（工程止血：editable install/包结构清理/最小 smoke）在总账中与 Phase 1 合并记录；补一份独立总结以保持 phase 总结文档体系一致。

**主要改动**
- 新增：`gemini_doc/phase0_summary.md`
  - 聚焦工程止血与最小可用开发基线（不展开 IR 设计细节）。
- 更新：`gemini_doc/README.md`
  - 将 `gemini_doc/phase0_summary.md` 纳入索引。
- 新增：`gemini_doc/change_2025-12-31_phase0_summary_doc.md`

**验证**
- 文档变更（无额外运行时验证）。

---

## 2026-01-03：更新全流程总览文档（从 claims 到 AE，对齐 Phase 0~6）

**动机**
- `gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md` 原版本停留在 Phase 5（schema 1.0）视角，并把 Phase 6（αβ-CROWN + BaB + E2E 工件链）写成 TODO；随着 Phase 6 收官，需要把“从研究主张到 AE 交付”的总导览升级为覆盖 Phase 0~6 的现状。

**主要改动**
- 重写：`gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md`
  - 版本升级为 v2.0，明确两条可复现主线：
    - Phase 5：interval IBP + TVM（bench JSONL `schema_version=1.0`）
    - Phase 6：αβ oracle + BaB（E2E JSON `schema_version=phase6h_e2e_v2`）
  - 补齐两套工件链的 Mermaid 流水线图，并更新 Phase 0~6 的导航入口（链接到 `gemini_doc/phase0_summary.md`~`gemini_doc/phase6_summary.md`）。
- 新增：`gemini_doc/change_2026-01-03_update_full_pipeline_from_claims_to_ae_v2.md`

**验证**
- 文档变更（无额外运行时验证）。

---

## 2026-01-03：修正文档 Mermaid 兼容性（全流程总览改用纯文本流水线图）

**动机**
- 部分 Markdown 渲染器不支持 Mermaid，导致 `gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md` 中的图无法正常显示；为保证在任意 Markdown 环境可读，改为纯文本图。

**主要改动**
- 更新：`gemini_doc/boundflow_full_pipeline_from_claims_to_ae.md`
  - 将 Phase 5/Phase 6 的 Mermaid 流水线图替换为 `text` 纯文本流水线图。
- 新增：`gemini_doc/change_2026-01-03_fix_mermaid_in_full_pipeline_doc.md`

**验证**
- 文档变更（无额外运行时验证）。

---

## 2025-12-24：Phase 6G PR-3B（BaB/αβ）——消除分支选择二次 forward（复用 forward trace / branch hint）

**动机**
- 节点评估（oracle）已跑 forward IBP，但分支选择 `_pick_branch` 仍会再次调用 `_forward_ibp_trace_mlp`，导致每个节点额外一次 forward，吞吐收益被抵消；PR-3B 目标是复用 node eval 的 forward trace/分支提示，避免二次 forward。

**主要改动**
- 更新：`boundflow/runtime/crown_ibp.py`
  - 新增 `run_crown_ibp_mlp_from_forward_trace(...)`：给定 `interval_env/relu_pre` 的 backward-only 入口，用于在优化循环/分支选择中复用 forward trace。
- 更新：`boundflow/runtime/alpha_beta_crown.py`
  - `run_alpha_beta_crown_mlp`：forward trace 只算一次，优化循环内复用；并在 `AlphaBetaCrownStats` 暴露 `branch_choices`（每个 batch/node 的分支提示）。
- 更新：`boundflow/runtime/bab.py`
  - `NodeEvalCacheValue` 增加 `branch`；`eval_bab_alpha_beta_node` 返回 `branch_hint`；`solve_bab_mlp` 分支阶段优先使用 hint（无 hint 才回退 `_pick_branch`）。
- 新增：`tests/test_phase6g_branch_pick_reuses_forward_trace.py`
  - monkeypatch 统计 `_forward_ibp_trace_mlp` 调用次数：oracle=1 次，branch pick=0 次。
- 新增：`gemini_doc/change_2025-12-24_phase6g_pr3b_branch_pick_reuse_forward_trace.md`

**验证**
- `python -m pytest -q tests/test_phase6g_branch_pick_reuses_forward_trace.py`

---

## 2025-12-24：Phase 6G PR-3C（BaB/αβ）——per-node infeasible/reason/witness + node-batch partial prune（first-layer）

**动机**
- node-batch（`B>1`）场景下 infeasible detector 被跳过会降低剪枝效率；同时 `feasible/reason/witness` 必须作为 per-node 元信息进入 cache/reuse（不影响 soundness，仅影响效率与可解释性）。

**主要改动**
- 更新：`boundflow/runtime/alpha_beta_crown.py`
  - 新增 `check_first_layer_infeasible_split(...)`：对 first-layer split halfspaces 做 best-effort infeasible 检测并返回 `AlphaBetaCrownStats`（含 witness）。
- 更新：`boundflow/runtime/bab.py`
  - 新增 `BabConfig.enable_batch_infeasible_prune: bool=False`（默认关闭）。
  - 新增 `prune_infeasible_first_layer_items(...)`：batch 内逐 node 检测并 partial prune，且把 infeasible 元信息写入 `NodeEvalCache`。
- 新增：`tests/test_phase6g_node_batch_partial_infeasible_prune.py`
  - 混合 infeasible/feasible 节点的 prune 回归，并验证 infeasible 元信息进入 cache。
- 新增：`gemini_doc/change_2025-12-24_phase6g_pr3c_per_node_infeasible_and_partial_prune.md`

**验证**
- `python -m pytest -q tests/test_phase6g_node_batch_partial_infeasible_prune.py`

## 2025-12-22：新增 Phase 6 评审备忘（无外链版）

**动机**
- 将三轴解耦 + stage pipeline 的评审建议改为只引用仓库内证据、统一指代、面向工程落地的版本，便于长期维护与引用。

**主要改动**
- 新增：`gemini_doc/phase6_review_three_axis_stage_pipeline.md`
  - 无外链版评审备忘（优势/避坑/落地顺序），并对齐 `docs/stage_4_critical_review.md`、`docs/p4_p5.md`、`docs/bench_jsonl_schema.md`。
- 新增：`gemini_doc/change_2025-12-22_phase6_review_notes_no_external_links.md`
  - 记录本次新增评审备忘的动机与摘要。
- 更新：`gemini_doc/change_2025-12-22_phase6a_inputspec_lpball_perturbation.md`
  - 补充指向评审备忘的链接。

**验证**
- 文档变更（无额外运行时验证）。

---

## 2026-03-15：新增本机 AI CLI 更新脚本

**动机**
- 本机已安装 `gemini`、`claude`、`codex` 三个 AI CLI，且均以 npm 全局包形式存在；手动逐个更新不便于统一检查当前版本、最新版本与 PATH 中实际命令位置。

**主要改动**
- 新增脚本：`scripts/update_ai_clis.sh`
  - 内置三组映射：
    - `gemini` -> `@google/gemini-cli`
    - `claude` -> `@anthropic-ai/claude-code`
    - `codex` -> `@openai/codex`
  - 支持 `--check`：只读显示当前版本、最新版本、命令路径和状态。
  - 支持指定目标更新：如 `bash scripts/update_ai_clis.sh codex`。
  - 缺失包默认跳过，`--install-missing` 时才安装。
  - 更新结束后回显最终状态，便于确认。
- 新增变更记录：`gemini_doc/change_2026-03-15_add_ai_cli_update_script.md`

**验证**
- `bash -n scripts/update_ai_clis.sh`
- `bash scripts/update_ai_clis.sh --check`

---

## 2026-03-15：新增研发脉络总览文档（project_evolution_overview）

**动机**
- 现有文档已经覆盖总账、阶段总结、全流程与设计评审，但缺少一篇面向“项目演化主线”的总整理文档，导致接手者需要在多份文档之间来回跳转。

**主要改动**
- 新增：`gemini_doc/project_evolution_overview.md`
  - 从项目目标、阶段推进、代码落点、现有记录分工与下一步路线五个维度整理 Phase 0~6 的研发主线。
- 更新：`gemini_doc/README.md`
  - 新增“研发演化/接手视角”阅读路径，并将 `project_evolution_overview.md` 纳入长期有效文档索引。
- 新增：`gemini_doc/change_2026-03-15_add_project_evolution_overview.md`

**影响面**
- 仅影响文档导航与接手效率，无代码语义变更，无 Python API、CLI、schema 变化。

**验证**
- 检查 `gemini_doc/project_evolution_overview.md` 内引用路径存在。
- 检查 `gemini_doc/README.md` 索引可读且定位准确。
- 确认新文档与 `phase*_summary.md` 形成互补，而非重复替代关系。

---

## 2026-03-15：Phase 7A PR-1——LinearOperator 输入侧基础抽象（concretize_affine foundation）

**动机**
- Phase 6 的 CROWN/alpha-beta 路径已能在链式 MLP 子集上跑通，但输入处 `concretize_affine(...)` 仍要求显式稠密 `A:[B,K,I]`。如果直接继续扩 Conv 或一般图，会把算子覆盖问题和线性形式表达问题耦合在一起，返工风险高。

**主要改动**
- 新增：`boundflow/runtime/linear_operator.py`
  - 提供运行时内部 `LinearOperator` 协议与 `DenseLinearOperator` 实现。
- 更新：`boundflow/runtime/perturbation.py`
  - `PerturbationSet.concretize_affine(...)` 改为接受 `torch.Tensor | LinearOperator`，但保持 tensor 路径数值行为不变。
- 更新：`boundflow/runtime/crown_ibp.py`
  - `run_crown_ibp_mlp(...)` 与 `run_crown_ibp_mlp_from_forward_trace(...)` 的最终输入 concretize 统一走 `DenseLinearOperator`。
- 更新：`boundflow/runtime/alpha_beta_crown.py`
  - first-layer infeasible helper 同步走 operator 路径。
- 新增：`tests/test_phase7a_linear_operator_concretize.py`
- 新增：`gemini_doc/change_2026-03-15_phase7a_pr1_linear_operator_concretize_foundation.md`

**影响面**
- 仅引入 runtime 内部基础抽象，不改公开 CLI、schema、artifact 口径。
- 不把 `LinearOperator` 混入 IR/Task/Planner，当前 backward 中间态仍保持 dense tensor。

**验证**
- `python -m pytest -q tests/test_phase7a_linear_operator_concretize.py`
- `python -m pytest -q tests/test_phase6b_crown_ibp_mlp.py`
- `python -m pytest -q tests/test_phase6d_alpha_crown_mlp.py`
- `python -m pytest -q tests/test_phase6f_alpha_beta_crown_pr1.py`
- `python -m pytest -q tests/test_phase6g_alpha_beta_multispec_batch.py`

---

## 2026-03-15：新增 BoundFlow 工作流 skill

**动机**
- 仓库里已经有 `gemini_doc/llm_collaboration_workflow.md` 这份协作文档，但它还只是文档，不是一个可被 Codex 直接触发和复用的 skill。需要把这套长期使用的 PR-by-PR 工作流固化成真正的 skill。

**主要改动**
- 新增 skill：`/home/lee/.codex/skills/boundflow-workflow/SKILL.md`
  - 触发范围限定在 BoundFlow 仓库内的工程迭代任务。
  - 固化执行顺序：读 `AGENTS.md` 与 `gemini_doc/llm_collaboration_workflow.md`、先做 DoD、实现最小闭环、跑定向测试、写 `gemini_doc/change_*.md`、追加 `docs/change_log.md`。
- 新增：`gemini_doc/change_2026-03-15_add_boundflow_workflow_skill.md`

**影响面**
- 不改仓库代码语义，不改 CLI、schema、artifact 口径。
- 将现有文档化工作流提升为可调用的 skill。

**验证**
- 检查 `~/.codex/skills/boundflow-workflow/SKILL.md` 已创建且内容覆盖 BoundFlow 工作流要点。
- 检查仓库中已记录本次变更，并追加总账。

---

## 2026-03-16：Phase 7A PR-2——线性 MLP backward A 状态 operator 化

**动机**
- Phase 7A PR-1 已经让输入边界 `concretize_affine(...)` 支持 `LinearOperator`，但 CROWN/alpha-beta 主路径里的 backward `A_u/A_l` 仍是全程 dense tensor。继续直接做 Conv 或一般图会把表达抽象和算子覆盖耦合在一起，返工风险高。

**主要改动**
- 更新：`boundflow/runtime/linear_operator.py`
  - 扩展 `LinearOperator` 内部能力，新增 `contract_last_dim(...)` 与 `matmul_right(...)`。
  - 新增 `RightMatmulLinearOperator`，用于表达 linear backward 的 lazy 右乘组合。
- 更新：`boundflow/runtime/crown_ibp.py`
  - 新增共享 `AffineBackwardState` 与 `_run_crown_backward_from_trace(...)`，收敛 `run_crown_ibp_mlp(...)` 和 `run_crown_ibp_mlp_from_forward_trace(...)` 的 backward 逻辑。
  - linear backward 改为 operator 路径；ReLU backward 保持显式 dense barrier。
  - 最终输入 concretize 直接接收 `LinearOperator`，不再只在最后包装 `DenseLinearOperator`。
- 新增：`tests/test_phase7a_pr2_linear_operator_backward_state.py`
  - 覆盖 lazy operator 数值等价、嵌套融合、非法输入防御，以及主 backward 路径确实调用 `matmul_right(...)`。
- 更新：`tests/test_phase7a_linear_operator_concretize.py`
  - 将对 CROWN 主路径的断言从 `DenseLinearOperator` 放宽到 `LinearOperator`。
- 新增：`gemini_doc/change_2026-03-16_phase7a_pr2_operatorize_linear_backward_state.md`

**影响面**
- 不改 CLI、schema、artifact、IR、planner。
- `alpha_crown.py` 与 `alpha_beta_crown.py` 通过共享 CROWN backward 自动继承这一改动，无需改变公开语义。
- 这一步的重点是表达层解耦，不承诺立即带来性能收益。

**验证**
- `python -m pytest -q tests/test_phase7a_pr2_linear_operator_backward_state.py`
- `python -m pytest -q tests/test_phase7a_linear_operator_concretize.py`
- `python -m pytest -q tests/test_phase6b_crown_ibp_mlp.py`
- `python -m pytest -q tests/test_phase6d_alpha_crown_mlp.py`
- `python -m pytest -q tests/test_phase6f_alpha_beta_crown_pr1.py`
- `python -m pytest -q tests/test_phase6g_alpha_beta_multispec_batch.py`

---

## 2026-03-16：为 Codex 全局安装 Superpowers，并补跨主机复用文档

**动机**
- 需要把 `superpowers` 装成主机级 Codex skill，使这台机器上的所有工程都能复用。
- 还需要留下一份可跨主机复用的安装文档，让另一台机器上的 Codex 读完后能自动完成安装，而不是死写某个固定路径。

**主要改动**
- 主机级配置：
  - clone `superpowers` 到 `~/.codex/superpowers`
  - 建立软链接：`~/.codex/skills/superpowers -> ~/.codex/superpowers/skills`
  - 更新 `~/.codex/config.toml`，启用：
    - `[features]`
    - `collab = true`
- 新增：`gemini_doc/codex_superpowers_global_install.md`
  - 记录目录探测规则、主机级安装步骤、验证/更新/卸载命令，以及一段可直接发给 Codex 的执行指令。
- 新增：`gemini_doc/change_2026-03-16_install_codex_superpowers_global.md`
- 更新：`gemini_doc/README.md`

**影响面**
- 不改 BoundFlow 代码语义，不改测试、schema、artifact 口径。
- 重启 Codex 后，这台机器上的所有工程都可发现 `superpowers`。

**验证**
- `ls -la ~/.codex/skills/superpowers`
- `git -C ~/.codex/superpowers rev-parse --short HEAD`
- `rg -n '^\[features\]|^collab = true$' ~/.codex/config.toml`

---

## 2026-03-16：Phase 7A PR-3——原生 NCHW contract 与 Conv-ready CROWN-IBP

**动机**
- PR-1/PR-2 已经把输入 concretize 和线性 MLP backward `A` 状态 operator 化，但 runtime contract 仍然默认输入是扁平 `[B,I]`。如果继续做 Conv 扩展而不升级 contract，后续会在 flatten/unflatten 适配上反复返工。

**主要改动**
- 更新：`boundflow/runtime/linear_operator.py`
  - `LinearOperator` 协议新增 `input_shape/input_numel/spec_dim/contract_input/reshape_input/conv2d_right`。
  - `DenseLinearOperator` 现在显式携带 `input_shape`。
  - 新增 `ReshapeInputLinearOperator` 与 `Conv2dLinearOperator`，让 runtime 内部可以原生表达 NCHW 输入与 `A @ Conv2d(x)`。
- 更新：`boundflow/runtime/perturbation.py`
  - `PerturbationSet.concretize_affine(...)` 现在原生接受 `center:[B,*input_shape]`。
  - tensor `A` 允许 `[B,K,I]` 或 `[B,K,*input_shape]`；operator `A` 则通过 `input_shape` 做显式校验。
- 更新：`boundflow/runtime/crown_ibp.py`
  - plain CROWN-IBP 从 `{linear,relu}` 扩到链式 `{conv2d,relu,flatten,linear}`。
  - forward trace 新支持 `conv2d` 与 `flatten(start_dim=1,end_dim=-1)`。
  - backward 新增 conv2d/flatten 分支；高维 ReLU pre-bound 在 barrier 处临时 flatten，再恢复原始 `input_shape`。
  - `get_crown_ibp_mlp_stats(...)` 现在接受 chain CNN 子集，但继续拒绝 skip/branch 非链式图。
- 更新：`boundflow/runtime/alpha_crown.py`
  - 对含 `conv2d/flatten` 的图显式 fail-fast，避免误把 PR-3 理解成 alpha-CROWN 也支持 CNN。
- 更新：`boundflow/runtime/alpha_beta_crown.py`
  - 对含 `conv2d/flatten` 的图显式 fail-fast；PR-3 只扩 plain CROWN-IBP，不扩 alpha-beta/BaB。
- 新增测试：
  - `tests/test_phase7a_pr3_highdim_concretize.py`
  - `tests/test_phase7a_pr3_conv_linear_operator.py`
  - `tests/test_phase7a_pr3_crown_ibp_cnn.py`
- 新增文档：
  - `gemini_doc/change_2026-03-16_phase7a_pr3_native_nchw_contract_and_conv_crown.md`

**影响面**
- runtime public contract 现在原生支持高维输入，但当前承诺范围只到 rank-2 flat 和 rank-4 `NCHW`。
- 不改 IR、planner、CLI、artifact schema。
- `alpha_crown.py`、`alpha_beta_crown.py`、`bab.py` 仍然是 MLP-only。

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase7a_pr3_highdim_concretize.py tests/test_phase7a_pr3_conv_linear_operator.py tests/test_phase7a_pr3_crown_ibp_cnn.py`
- `conda run -n boundflow python -m pytest -q tests/test_phase7a_pr2_linear_operator_backward_state.py tests/test_phase7a_linear_operator_concretize.py tests/test_phase6b_crown_ibp_mlp.py tests/test_phase6d_alpha_crown_mlp.py tests/test_phase6f_alpha_beta_crown_pr1.py tests/test_phase6g_alpha_beta_multispec_batch.py`

---

## 2026-03-18：Phase 7A PR-4——Conv2dLinearOperator 的 exact lazy row-norm 归约

**动机**
- PR-3 已经把 `Conv2dLinearOperator` 接到高维 `NCHW` contract 和 plain CROWN-IBP 上，但它的 `row_abs_sum / row_l2_norm / row_abs_max` 仍然直接走 `to_dense()` 再归约。语义正确，但过早摊平了 Conv operator 的结构信息。

**主要改动**
- 更新：`boundflow/runtime/linear_operator.py`
  - 新增 `_materialize_feature_map_rows(...)`，把 operator 递归 materialize 成 `[B*K,C,H,W]` feature-map rows。
  - 新增 `_reduce_feature_map_rows(...)`，直接在 NCHW rows 上做 `l1/l2/linf` 归约。
  - `Conv2dLinearOperator.row_abs_sum / row_l2_norm / row_abs_max` 改为 exact lazy 路径，不再直接调用 `self.to_dense()`。
  - `Conv2dLinearOperator.to_dense()` 本身保持不变，继续作为 debug/reference path。
- 新增测试：
  - `tests/test_phase7a_pr4_conv_lazy_norms.py`
  - 覆盖单层 conv、嵌套 conv、禁止回退到 `Conv2dLinearOperator.to_dense()`、以及 `concretize_affine(...)` 在 `p in {inf,2,1}` 下与 dense reference 完全一致。
- 新增文档：
  - `gemini_doc/change_2026-03-18_phase7a_pr4_conv_lazy_row_norms.md`

**影响面**
- 不改 public API，不改 `LinearOperator` protocol，不改 `PerturbationSet.concretize_affine(...)` 签名。
- 不扩 `CROWN` / `alpha` / `alpha-beta` / `BaB` 的公开语义。
- 这次 PR 的“lazy”指结构化递归归约，而不是零 materialization，也不承诺最坏复杂度优化。

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase7a_pr4_conv_lazy_norms.py`
- `conda run -n boundflow python -m pytest -q tests/test_phase7a_pr3_conv_linear_operator.py tests/test_phase7a_pr3_highdim_concretize.py tests/test_phase7a_pr3_crown_ibp_cnn.py tests/test_phase7a_pr2_linear_operator_backward_state.py tests/test_phase6b_crown_ibp_mlp.py tests/test_phase6d_alpha_crown_mlp.py tests/test_phase6f_alpha_beta_crown_pr1.py tests/test_phase6g_alpha_beta_multispec_batch.py`

---

## 2026-03-19：Phase 7A PR-5——将 alpha-CROWN 从 MLP 扩到 chain CNN

**动机**
- PR-3/PR-4 已经把 plain CROWN-IBP 和 `Conv2dLinearOperator` 扩到链式 CNN 子集 `{conv2d,relu,flatten,linear}`，但 `alpha_crown.py` 仍然是 MLP-only，`run_crown_ibp_mlp(..., relu_alpha=...)` 也还拒绝 rank>2 的 ReLU alpha。

**主要改动**
- 更新：`boundflow/runtime/crown_ibp.py`
  - `_broadcast_relu_alpha(...)` 从 rank-2 扩到支持高维 ReLU pre bound。
  - 允许 shared alpha 形状：`[]`、`[*S]`、`[I]`、`[1,*S]`、`[1,I]`。
  - 明确拒绝 batch-specific alpha：`[B,*S]`、`[B,I]`。
  - `_forward_ibp_trace_mlp(...)` 对 conv `relu_split_state` 报更清晰的未支持信息。
- 更新：`boundflow/runtime/alpha_crown.py`
  - 删除原来的线性层静态维度推断，改为从 `_forward_ibp_trace_mlp(...)` 的 `relu_pre` 读取逻辑 shape。
  - `AlphaState.alpha_by_relu_input[name]` 现在既支持 MLP 的 `[H]`，也支持 conv ReLU 的 `[C,H,W]`。
  - `run_alpha_crown_mlp(...)` 改成 forward-trace reuse，优化循环内调用 `run_crown_ibp_mlp_from_forward_trace(...)`。
  - warm-start 现在支持高维 shared alpha 的 shape 归一。
- 更新：`boundflow/runtime/alpha_beta_crown.py`
  - 对 conv 图继续 fail-fast，但文案改成“PR5 只扩 alpha-CROWN，alpha-beta-CROWN 仍然是 MLP-only”。
- 更新：`boundflow/runtime/bab.py`
  - 在 `solve_bab_mlp(...)` 入口处显式拒绝 `conv2d/flatten` 图，避免 `run_alpha_crown_mlp(...)` 扩容后误开 BaB。
- 新增测试：
  - `tests/test_phase7a_pr5_alpha_crown_cnn.py`
- 新增文档：
  - `gemini_doc/change_2026-03-19_phase7a_pr5_alpha_crown_chain_cnn.md`

**影响面**
- `run_alpha_crown_mlp(...)` 现在支持链式 CNN 子集 `{conv2d,relu,flatten,linear}`。
- `run_crown_ibp_mlp(..., relu_alpha=...)` 现在支持高维 shared alpha。
- `alpha-beta-CROWN` 与 `BaB` 仍然保持 MLP-only。
- conv `relu_split_state` 仍未开放。

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase7a_pr5_alpha_crown_cnn.py`
- `conda run -n boundflow python -m pytest -q tests/test_phase6d_alpha_crown_mlp.py tests/test_phase6f_alpha_beta_crown_pr1.py tests/test_phase6g_alpha_beta_multispec_batch.py tests/test_phase6e_bab_mlp.py`
- `conda run -n boundflow python -m pytest -q tests/test_phase7a_pr3_crown_ibp_cnn.py tests/test_phase7a_pr4_conv_lazy_norms.py`

---

## 2026-03-19：Phase 7A PR-6——将 alpha-beta-CROWN 从 MLP 扩到 chain CNN

**动机**
- PR-5 已经把 plain CROWN-IBP 和 alpha-CROWN 扩到链式 CNN 子集 `{conv2d,relu,flatten,linear}`，但 `alpha_beta_crown.py` 仍然停在 MLP-only：conv split/beta/first-layer detector 都还没打通。

**主要改动**
- 新增：`boundflow/runtime/relu_shape_utils.py`
  - 抽出高维 ReLU 公共 shape/broadcast helper：`shape_numel(...)`、`relu_input_shapes(...)`、`coerce_relu_param_shape(...)`、`broadcast_relu_split_like_pre(...)`。
- 更新：`boundflow/runtime/crown_ibp.py`
  - `_apply_relu_split(...)` 从 rank-2 扩到 rank-agnostic，conv `relu_split_state` 现在真正进入 `_forward_ibp_trace_mlp(...)`。
  - `relu_pre_add_coeff_l` 现在允许高维 structured 形状并在 backward 时 flatten 到 `[B,I]`。
  - `relu_alpha` 广播现在接受 per-batch 高维形状，供 alpha-beta oracle 的 `per_batch_params=True` 使用。
- 更新：`boundflow/runtime/alpha_crown.py`
  - 改用 `relu_shape_utils.py`，删除本地重复的高维 alpha shape helper。
- 更新：`boundflow/runtime/alpha_beta_crown.py`
  - 去掉 conv 图 fail-fast，`run_alpha_beta_crown_mlp(...)` 现在支持链式 CNN 子集 `{conv2d,relu,flatten,linear}`。
  - `AlphaState/BetaState` 从“隐藏维度 `H`”切到基于 `relu_pre` 的逻辑 shape `[*S]`。
  - `per_batch_params=False` 时，conv alpha/beta 为 shared `[*S]`；`per_batch_params=True` 时为 `[B,*S]`。
  - `_beta_to_relu_pre_add_coeff(...)` 现在支持高维 split + 高维 beta，统一输出 flat `[B,I]`。
  - `_branch_choices_from_relu_pre(...)` 改成任意 rank flatten 后选最大 gap，继续返回 `(relu_input_name, flat_idx)`。
  - `check_first_layer_infeasible_split(...)` 与 `_collect_first_layer_split_halfspaces(...)` 新增对 direct-input `conv2d -> relu` 的支持。
  - first-layer conv 证书通过 one-hot output row + `DenseLinearOperator(...).conv2d_right(...).to_dense()` 提取 affine row。
  - deeper-than-first-layer conv split 不进入 halfspace 证书，只返回 `ok (no first-layer split halfspaces)`。
- 更新：`boundflow/runtime/bab.py`
  - conv 图继续 fail-fast，但文案改成：`BaB conv graphs not yet supported; PR6 only extends alpha-beta-CROWN oracle`。
- 新增测试：
  - `tests/test_phase7a_pr6_alpha_beta_crown_cnn.py`
- 更新测试：
  - `tests/test_phase7a_pr5_alpha_crown_cnn.py`
  - 把 PR-5 里“conv split_state 仍不支持”的旧断言改成 PR-6 新行为检查。
- 新增文档：
  - `gemini_doc/change_2026-03-19_phase7a_pr6_alpha_beta_crown_chain_cnn.md`

**影响面**
- `run_alpha_beta_crown_mlp(...)` 现在支持 chain CNN 子集 `{conv2d,relu,flatten,linear}`。
- conv `relu_split_state` 支持 shared + batch-specific 高维形状，并统一到 `[B,I]`。
- conv `alpha/beta` 完全沿用 `per_batch_params` 语义。
- first-layer direct-input conv split 可被 infeasible detector 证伪。
- `AlphaBetaCrownStats.branch_choices` 在 conv 图上继续返回 flat idx。
- conv `BaB` 仍然未开放。

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase7a_pr6_alpha_beta_crown_cnn.py`
- `conda run -n boundflow python -m pytest -q tests/test_phase7a_pr5_alpha_crown_cnn.py tests/test_phase7a_pr4_conv_lazy_norms.py tests/test_phase7a_pr3_crown_ibp_cnn.py tests/test_phase6d_alpha_crown_mlp.py tests/test_phase6f_alpha_beta_crown_pr1.py tests/test_phase6g_alpha_beta_multispec_batch.py tests/test_phase6e_bab_mlp.py`

---

## 2026-03-19：Phase 6G 零 split-state detector 短路修正

**动机**
- 在清理并验证剩余未提交改动时，`tests/test_phase6g_branch_pick_reuses_forward_trace.py` 暴露出一个回归：root 节点空 split 也会触发 alpha-beta oracle 的 first-layer infeasible detector，导致多跑一次 `_forward_ibp_trace_mlp(...)`。

**主要改动**
- 更新：`boundflow/runtime/alpha_beta_crown.py`
  - 新增 `_has_nonzero_split_state(...)`
  - `_collect_first_layer_split_halfspaces(...)` 在 `relu_split_state` 为空或“全部为 0”时直接返回，不再额外跑 forward trace
  - `run_alpha_beta_crown_mlp(...)` 的 `do_infeasible_check` 改成只在存在非零 split 时开启
- 新增文档：
  - `gemini_doc/change_2026-03-19_phase6g_zero_split_detector_short_circuit.md`

**影响面**
- root node 的 `ReluSplitState.empty(...)` 不再误触发 detector
- `branch_choices` 继续复用 alpha-beta oracle 已有的 forward trace
- Phase 6G 的 branch picking forward reuse 回归恢复通过

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase6g_branch_pick_reuses_forward_trace.py`
- `conda run -n boundflow python -m pytest -q tests/test_env.py tests/test_phase4d_onnx_frontend_matches_torch.py tests/test_phase5d_pr8_relax_lowering_skeleton.py tests/test_phase5d_pr9_tvm_executor_linear_equiv.py tests/test_phase5d_pr10_tvm_compile_instruments.py tests/test_phase5d_pr11a_task_relax_ops_equiv.py tests/test_phase5d_pr11c1_save_function_closure.py tests/test_phase5d_pr11c_vm_cache_and_opt_passes.py tests/test_phase5d_pr12_2_tir_var_upper_bound_effect.py tests/test_phase5d_pr12_static_plan_modes.py tests/test_phase6c_crown_ibp_multispec_batch.py tests/test_phase6g_bab_node_batch.py tests/test_phase6g_bab_node_eval_cache.py tests/test_phase6g_branch_pick_reuses_forward_trace.py tests/test_phase6g_node_batch_grad_isolation.py tests/test_phase6g_node_batch_partial_infeasible_prune.py tests/test_phase6h_artifact_runner_smoke.py tests/test_phase6h_bench_e2e_schema.py tests/test_phase6h_plot_smoke.py tests/test_phase6h_report_csv_schema.py tests/test_phase6h_workload_suite_smoke.py`

---

## 2026-03-20：Phase 7A PR-7——BaB on chain CNN（含 node-batch，与真实样本 batch 共存）

**动机**
- PR-6 已经把 `alpha-beta-CROWN` oracle 扩到 chain CNN，但 `bab.py` 仍然停在 MLP-only：conv 图 fail-fast、`ReluSplitState` 只支持 `[H]`、node-batch 仍假设输入 `B==1`。

**主要改动**
- 更新：`boundflow/runtime/bab.py`
  - `solve_bab_mlp(...)` 现在支持 chain CNN 子集 `{conv2d,relu,flatten,linear}`，但仅在 `oracle="alpha_beta"` 时开放。
  - `oracle="alpha"` 在 conv 图上继续 fail-fast，文案改为：`alpha-only BaB does not yet support conv graphs`。
  - `ReluSplitState.empty(...)` 新增 `input_spec=`，改为从 forward trace 的 `relu_pre` 推断高维逻辑 shape。
  - `ReluSplitState.with_split(...)` 继续接收 flat idx，但现在对逻辑 shape 做 flatten 更新后再恢复。
  - `_QueueItem` 新增 `example_idx`，host 侧改成“每样本独立搜索树 + 全局 heap 调度”。
  - node-batch 可以混合不同样本的节点进入一次 `alpha-beta` oracle 调用。
  - `max_nodes` 口径改成每样本独立预算。
  - 新增 `BabPerExampleResult`，`BabResult` 保留旧聚合字段并增加 `per_example`。
  - `_pick_branch(...)` 改成 rank-agnostic，对任意 `relu_pre` flatten 成 `[B,I]` 后返回 flat idx。
  - `prune_infeasible_first_layer_items(...)` 改成按 `item.example_idx` 切样本，并按样本分隔 `NodeEvalCache`。
- 更新脚本：
  - `scripts/bench_phase6g_bab_node_batch_throughput.py`
    - 适配 `_QueueItem.example_idx` 与 `cache_by_example`
  - `scripts/bench_phase6h_bab_e2e_time_to_verify.py`
    - 修正 instrumentation 对 `prune_infeasible_first_layer_items(...)` 新签名的兼容
- 更新测试：
  - `tests/test_phase6g_node_batch_partial_infeasible_prune.py`
  - `tests/test_phase7a_pr5_alpha_crown_cnn.py`
- 新增测试：
  - `tests/test_phase7a_pr7_bab_chain_cnn.py`
  - `tests/test_phase7a_pr7_bab_batch_examples.py`
- 新增文档：
  - `gemini_doc/change_2026-03-20_phase7a_pr7_bab_chain_cnn.md`

**影响面**
- BaB 现在支持 chain CNN，但不扩 skip/branch/general DAG。
- `B>1` 采用混合方案：
  - host/BaB：每样本独立搜索树
  - oracle：batch 维表示一批待评估节点/domain
- 现有 bench/schema 不因为 `BabResult.per_example` bump 版本；旧脚本继续读取聚合字段。

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase7a_pr7_bab_chain_cnn.py tests/test_phase7a_pr7_bab_batch_examples.py tests/test_phase7a_pr5_alpha_crown_cnn.py tests/test_phase6g_node_batch_partial_infeasible_prune.py tests/test_phase6g_bab_node_batch.py tests/test_phase6g_bab_node_eval_cache.py tests/test_phase6g_node_batch_grad_isolation.py tests/test_phase6g_branch_pick_reuses_forward_trace.py tests/test_phase6e_bab_mlp.py tests/test_phase6f_alpha_beta_crown_pr1.py tests/test_phase6g_alpha_beta_multispec_batch.py tests/test_phase6d_alpha_crown_mlp.py tests/test_phase7a_pr6_alpha_beta_crown_cnn.py tests/test_phase7a_pr3_crown_ibp_cnn.py tests/test_phase7a_pr4_conv_lazy_norms.py tests/test_phase6h_bench_e2e_schema.py`
- 结果：`49 passed in 7.35s`

---

## 2026-03-21：Phase 7A PR-8——solver 栈从 chain 扩到 residual/general DAG（含 Torch/ONNX 前端）

**动机**
- PR-7 之前，BoundFlow 的 solver 栈虽然已经覆盖 chain MLP / chain CNN / conv alpha / conv alpha-beta / conv BaB，但 runtime 与前端仍默认“图是链式的”，无法承接最小 ResNet/basic-block 风格 general DAG：
  - residual add
  - projection skip
  - feature/channel concat

**主要改动**
- 更新：`boundflow/runtime/crown_ibp.py`
  - `_forward_ibp_trace_mlp(...)` 新增 `add` / `concat`
  - backward 从链式回扫改成 reverse-topo DAG adjoint 聚合
  - DAG 汇合点统一走 exact dense barrier
  - `run_crown_ibp_mlp(...)` 去掉 chain-only 结构限制
  - `get_crown_ibp_mlp_stats(...)` 改成接受 general DAG 子集 `{linear,conv2d,relu,flatten,add,concat}`
- 更新：`boundflow/runtime/task_executor.py`
  - `PythonTaskExecutor.run_ibp(...)` / `run_ibp_task(...)` 新增 `concat`
  - `add` 改为显式拒绝 broadcast
- 更新：`boundflow/frontends/pytorch/frontend.py`
  - 新增 `aten.cat.default` / `aten.concat.default -> concat`
  - 提取 `concat.axis`
- 更新：`boundflow/frontends/onnx/frontend.py`
  - 新增 `Concat` 导入
- 更新：`boundflow/runtime/alpha_crown.py`
  - 当所有 ReLU 都已 stable、loss 不依赖 alpha 时，直接把 step-0 结果作为最优返回，不再在 `backward()` 处报无梯度错误
- 更新：`boundflow/runtime/alpha_beta_crown.py`
  - 同样处理“所有 split/relaxation 参数都无梯度可学”的情形
- 新增测试：
  - `tests/test_phase7a_pr8_general_dag_runtime.py`
  - `tests/test_phase7a_pr8_general_dag_frontends.py`
- 更新测试：
  - `tests/test_phase6b_crown_ibp_mlp.py`
  - `tests/test_phase7a_pr3_crown_ibp_cnn.py`
- 新增文档：
  - `gemini_doc/change_2026-03-21_phase7a_pr8_general_dag_solver_stack.md`

**影响面**
- solver 栈现在支持单 task、静态 shape 的 general DAG 子集：
  - `linear`
  - `conv2d`
  - `relu`
  - `flatten`
  - `add`
  - `concat`
- `add` 只支持 exact same-shape，不支持 broadcast
- `concat` 只支持：
  - rank-2 `[B,F]` feature axis
  - rank-4 `NCHW` channel axis
- infeasible detector 数学边界不扩，仍只对 direct-input first-layer affine producer 生效
- 公开 API 名字不变；这是行为扩展，不引入 `*_dag` 新函数族

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase7a_pr8_general_dag_runtime.py tests/test_phase7a_pr8_general_dag_frontends.py tests/test_phase6b_crown_ibp_mlp.py::test_crown_ibp_mlp_supports_general_dag_branch_graph tests/test_phase7a_pr3_crown_ibp_cnn.py::test_crown_ibp_stats_supports_chain_cnn_and_branch_like_cnn_dag`
- 结果：`9 passed, 4 warnings in 2.32s`
- `conda run -n boundflow python -m pytest -q tests/test_phase7a_pr8_general_dag_runtime.py tests/test_phase7a_pr8_general_dag_frontends.py tests/test_phase7a_pr7_bab_chain_cnn.py tests/test_phase7a_pr7_bab_batch_examples.py tests/test_phase7a_pr6_alpha_beta_crown_cnn.py tests/test_phase7a_pr5_alpha_crown_cnn.py tests/test_phase7a_pr4_conv_lazy_norms.py tests/test_phase7a_pr3_crown_ibp_cnn.py tests/test_phase6b_crown_ibp_mlp.py tests/test_phase6d_alpha_crown_mlp.py tests/test_phase6e_bab_mlp.py tests/test_phase6f_alpha_beta_crown_pr1.py tests/test_phase6g_alpha_beta_multispec_batch.py tests/test_phase6g_bab_node_batch.py tests/test_phase6g_bab_node_eval_cache.py tests/test_phase6g_node_batch_grad_isolation.py tests/test_phase6g_branch_pick_reuses_forward_trace.py tests/test_phase6g_node_batch_partial_infeasible_prune.py tests/test_phase6h_bench_e2e_schema.py tests/test_phase4d_onnx_frontend_matches_torch.py`
- 结果：`65 passed, 8 warnings in 3.68s`

---

## 2026-03-21：清理旧 worktree 与无效草稿文件

**动机**
- PR-8 合回主线后，仓库中仍残留一个已过时的 feature worktree，以及两份无关草稿文件，继续保留只会污染工作区状态。

**主要改动**
- 删除旧 worktree 与分支：
  - `.worktrees/phase7a-pr6-alpha-beta-crown-chain-cnn`
  - `phase7a-pr6-alpha-beta-crown-chain-cnn`
- 删除无效草稿：
  - `gemini_doc/Untitled-1.md`
  - `gemini_doc/chatlog.md`
- 新增文档：
  - `gemini_doc/change_2026-03-21_cleanup_stale_worktree_and_drafts.md`

**影响面**
- `git worktree list` 现在只剩主仓库 `main`
- 工作区不再保留无关草稿文件

**验证**
- `git rev-list --left-right --count main...phase7a-pr6-alpha-beta-crown-chain-cnn`
- 结果：`4 0`
- `git worktree list`
- 结果：只剩 `/home/lee/Codes/boundflow  e4cd789 [main]`

---

## 2026-03-25：刷新 BoundFlow 工作流 skill

**动机**
- 仓库里已有 `gemini_doc/llm_collaboration_workflow.md` 长版工作流文档，也已有早期版本的 `boundflow-workflow` skill。
- 但旧 skill 只覆盖了简版入口，还缺少几项已经在实际协作中固定下来的关键动作：
  - 用户输入模板
  - 先写 DoD 再实现
  - 失败时先分流 `contract/pipeline` 与 `shape/dtype/数值语义`
  - 收尾时明确“已验证/残余风险/下一步 PR”

**主要改动**
- 更新本机 skill：
  - `/home/lee/.codex/skills/boundflow-workflow/SKILL.md`
  - 把 frontmatter 描述收紧成触发条件
  - 把 `llm_collaboration_workflow.md` 的关键动作压缩进可执行入口
- 更新：
  - `gemini_doc/README.md`
  - 在“研发协作流程”入口里补上本机 skill 路径
- 新增文档：
  - `gemini_doc/change_2026-03-25_refresh_boundflow_workflow_skill.md`

**影响面**
- BoundFlow 仓库现在同时有：
  - 长版工作流文档：`gemini_doc/llm_collaboration_workflow.md`
  - 可执行短版入口：`~/.codex/skills/boundflow-workflow/SKILL.md`
- 以后在仓库内触发该 skill 时，会更接近当前真实使用的 PR-by-PR 工作流，而不是只停留在简版提醒。

**验证**
- 检查 `~/.codex/skills/boundflow-workflow/SKILL.md` 已更新；
- 检查 skill 已覆盖：
  - 触发条件
  - 仓库硬约束
  - 默认执行顺序
  - 用户输入模板
  - 完成标准
- 检查 `gemini_doc/README.md` 已补上 skill 路径；
- 检查仓库侧已写本次变更记录并追加总账。

---

## 2026-03-25：将 Codex 配置从 collab 切换到 multi_agent

**动机**
- Codex 已提示 `[features].collab` 废弃，要求使用 `[features].multi_agent`。
- 继续保留旧字段会产生持续告警，也会增加后续版本兼容风险。

**主要改动**
- 更新用户级配置：
  - `~/.codex/config.toml`
  - 将 `[features]` 段中的 `collab = true` 替换为 `multi_agent = true`
- 新增文档：
  - `gemini_doc/change_2026-03-25_replace_codex_collab_with_multi_agent.md`

**影响面**
- Codex 后续将按新的 `multi_agent` 特性开关工作；
- 当前仓库代码与测试行为不受影响。

**验证**
- `sed -n '1,200p' ~/.codex/config.toml`
- `rg -n '^\[features\]|^(collab|multi_agent) = ' ~/.codex/config.toml`
- 结果：
  - `[features]` 段存在
  - `multi_agent = true` 存在
  - `collab = true` 不再存在

---

## 2026-03-26：Phase 7A PR-9——operator-preserving DAG backward

**动机**
- PR-8 已经把 solver 栈扩到最小 residual/general DAG，但 DAG backward 热路径还保留两个 dense barrier：
  - adjoint merge 通过 `to_dense() + sum`
  - `concat` backward 通过 dense slice 回落 `DenseLinearOperator`
- 如果不先去掉这两个热路径 barrier，后续继续扩图/扩 merge 语义会先被显式大张量 `A` 的 materialize 吃掉。

**主要改动**
- 新增：`boundflow/runtime/dag_utils.py`
  - 统一 `concat` 的 axis 归一化与 shape 校验
- 更新：`boundflow/runtime/linear_operator.py`
  - `LinearOperator` 协议新增 `add(...)` / `slice_input(...)`
  - 新增 `AddLinearOperator` 与 `SliceInputLinearOperator`
  - `Dense/RightMatmul/ReshapeInput/Conv2d` 全部补齐组合入口
  - rank-3 row norms 对新 operator 保持 anti-dense 归约路径
- 更新：`boundflow/runtime/crown_ibp.py`
  - `_accumulate_backward_state(...)` 改成 operator `add(...)`
  - `concat` backward 改成 operator `slice_input(...)`
  - `_split_bias_once(...)` 增加 guard
  - `get_crown_ibp_mlp_stats(...)` 复用 concat helper
- 更新：`boundflow/runtime/task_executor.py`
  - 复用新的 concat helper，去掉重复 axis/shape 校验逻辑
- 新增测试：
  - `tests/test_phase7a_pr9_dag_linear_operator.py`
  - `tests/test_phase7a_pr9_operator_preserving_dag_backward.py`
- 新增文档：
  - `gemini_doc/change_2026-03-26_phase7a_pr9_operator_preserving_dag_backward.md`

**影响面**
- 不改 public API，不扩 PR-8 已承诺的 general DAG 语义范围。
- ReLU backward 的 dense barrier 继续保留；PR-9 只去掉 DAG merge / concat backward 的 dense barrier。
- alpha / alpha-beta / BaB 通过共享的 CROWN backward 自动继承这次改动。

**验证**
- `conda run -n boundflow python -m pytest -q tests/test_phase7a_pr9_dag_linear_operator.py tests/test_phase7a_pr9_operator_preserving_dag_backward.py tests/test_phase7a_pr8_general_dag_runtime.py tests/test_phase7a_pr8_general_dag_frontends.py tests/test_phase7a_pr4_conv_lazy_norms.py tests/test_phase7a_pr3_crown_ibp_cnn.py tests/test_phase7a_pr5_alpha_crown_cnn.py tests/test_phase7a_pr6_alpha_beta_crown_cnn.py tests/test_phase7a_pr7_bab_chain_cnn.py tests/test_phase7a_pr7_bab_batch_examples.py`
- 结果：`47 passed, 4 warnings in 2.52s`
- `conda run -n boundflow python -m pytest -q tests/test_phase6b_crown_ibp_mlp.py tests/test_phase6d_alpha_crown_mlp.py tests/test_phase6e_bab_mlp.py tests/test_phase6f_alpha_beta_crown_pr1.py tests/test_phase6g_alpha_beta_multispec_batch.py`
- 结果：`21 passed in 1.66s`
- 新增下一步文档：
  - `gemini_doc/next_plan_after_phase7a_pr9.md`
  - 将 PR-10 主线固定为 “ReLU barrier 结构化”，而不是把更强 lazy row-norm 写成并列路线

---

## 2026-07-12：PR-11 有界分层 placement retry

**动机**
- 原 Global Retry 在 `spec=128/domain=8` held-out 上最坏需要 56 次预算拒绝 replay，无法作为真实
  runtime 的有限重试策略。

**主要改动**
- Planner 新增 `latency_rank_stratified_v1`：两个最快候选、80%/90% latency-rank 候选、最低
  predicted-peak fallback，默认最多 5 次；
- scheduler 与 plain CROWN wrapper 接入 bounded ranking + real CUDA OOM retry；
- evaluator 新增 `global_bounded_retry`，real-OOM runner 改走相同入口；
- 新增记录：`gemini_doc/change_2026-07-12_pr11_bounded_stratified_retry.md`。

**证据**
- s32/d8：7/7 feasible、0 unexpected、median/p90 regret 1.159×/1.722×、最多 3 次；
- s128/d8：7/7 feasible、0 unexpected、median/p90 regret 1.171×/1.221×、最多 5 次；
- 380 MiB cap 下 dense real OOM → structured success，3/3 独立进程稳定恢复。

---

## 2026-07-12：PR-11 独立 topology held-out No-Go

**主要改动与结论**
- 新增 7-barrier `branched_resnet`（parallel residual branches + add + concat + fuse）profile workload；
- 128/128 placement combinations 与 dense reference 对齐；
- bounded retry 为 9/9 feasible、0 unexpected，但 median/p90 regret 为 1.976×/4.494×，失败；
- evaluator 仍读取 candidate-specific trace logical bytes，只能定位为 profile-guided replay；
- 下一切片冻结为 static topology/liveness-aware barrier cost summary。

**记录**
- `gemini_doc/change_2026-07-12_pr11_independent_topology_nogo.md`

---

## 2026-07-12：PR-11 static topology/liveness cost

**主要改动**
- 新增 candidate-independent barrier schema：shape/bytes、fanout、live span、depth、merge/branch/path；
- profile/cost-model/evaluator 分别升级到 v2/v2/v3，禁止再用 held-out candidate trace feature；
- all-structured 作为显式 conservative fallback；
- 新增 topology-density-stratified v3、3× replicated profile 聚合与 6-family LOO；冻结
  ridge=.001、factor=1.30。

**结果**
- 3 轮共 1,416/1,416 placement executions correctness 通过，聚合为 472 patterns；
- 三组 held-out 共 23/23 feasible、0 unexpected；median regret 最坏为 1.880×；
- p90/max 最坏为 2.377×/3.160×；候选上限为 6；
- static model loader/candidate generator/plain-CROWN runtime 已连通，统一 QueryState/BaB 尚未完成。

**记录**
- `gemini_doc/change_2026-07-12_pr11_static_topology_cost.md`
- `gemini_doc/pr11_closure_audit_2026_07_12.md`

---

## 2026-07-13：PR-12E/F runtime Pareto 与 frozen held-out

**主要改动**
- 新增 calibration-only fused backend Planner，按 family、bytes-per-region、预算和 eligibility
  选择 PyTorch eager 或 TVM fused TIR；
- 新增 default/custom-stream runtime benchmark，分离 compile-first/cold/warm、CUDA Events 与
  allocator peak；
- 新增 JSONL→CSV→Pareto figure→manifest 后处理和三组 contract tests；
- 新增 fanout graph-ineligible fallback control，Planner/Oracle 均必须保持 eager。

**结果与判定**
- calibration 12/12、held-out 24/24 candidate rows correctness 通过；
- 5/5 held-out 预算可行、0 unsafe，Planner median/p90 regret 1.000×/1.262×；
- fused 在所有 held-out 降低 peak，但 memory-sensitive Linear、unseen Conv、mini-ResNet latency
  分别为 eager 的 4.21×、1.26×、1.03×；
- PR-12E 证据链 PASS、性能目标 FAIL；PR-12F execution PASS、quality guarded/partial；PR-12
  overall 保持 IN PROGRESS，PR-13 blocked。

**记录**
- `gemini_doc/change_2026-07-13_pr12ef_runtime_pareto_heldout.md`

---

## 2026-07-13：PR-12G budgeted chunked backend 与多后端 Planner

**主要改动**
- 新增 `pytorch_chunked` backend，限制 scaled-A query-row workspace 并复用 cuBLAS/cuDNN；
- 新增 eager/chunked/TIR calibration-only Planner、runtime selection/backend step contract；
- 冻结全新 multibackend-v2 split，并新增 benchmark replay、三候选 CSV/Pareto/manifest 工具；
- 新增 chunk correctness、custom stream、split 隔离、Planner replay 与 postprocess tests。

**结果与判定**
- calibration 48/48、held-out 36/36 candidate rows correctness 通过；
- 5/5 预算可行、0 unsafe，exact Oracle 3/5，median/p90 regret 1.000×/1.054×；
- Planner 选择 eager/chunked/TIR 为 1/2/2，selected geomean 相对 eager 1.081×；
- reduced Planner quality PASS，但 structured-eager/TVM-unfused、2× headline 和 repeated-query
  E2E 未关闭；PR-12 overall 保持 IN PROGRESS，PR-13 blocked。
- 收尾门禁：focused 41 passed；全量 318 passed、1 skipped；mypy 14 files success；pylint
  7 files 10.00/10；Black/diff check 通过。

**记录**
- `gemini_doc/change_2026-07-13_pr12g_multibackend_planner.md`

---

## 2026-07-14：PR-12H benchmark contract freeze

**主要改动**
- 冻结 `pr12g-validated-reduced` tag；
- 新增 kernel、region-runtime、end-to-end final-bound 三层机器可读 contract；
- 历史 fused-sanity 与 runtime-Pareto 明确声明 `compliant=false` 及缺失 inclusion；
- 新增 PR-12H–N 中长期计划和跨会话执行状态。

**边界**
- 不修改 canonical3 数值、不增加性能 claim；
- 下一阶段固定为 structured eager/TVM-unfused baseline；PR-13 继续 blocked。
- 收尾门禁：focused 7 passed；全量 321 passed、1 skipped；mypy 5 files success；pylint
  3 files 10.00/10；Black/diff check 通过。

**记录**
- `docs/pr12_benchmark_contract.md`
- `gemini_doc/change_2026-07-14_pr12h_benchmark_contract.md`

---

## 2026-07-14：PR-12I structured/TVM-unfused 公平 baseline

**主要改动**
- 新增显式 `scaled_u/scaled_l` workspace 的 TVM-unfused Linear/Conv2d baseline；
- 在 frozen region-runtime 与 complete final-bound 合同下统一比较 eager、structured、chunked、
  TVM unfused 与 TVM fused；
- 对 `torch.compile(fullgraph=True)` 做未改写 workload 的条件 probe，并保留结构化失败；
- 新增 JSONL→CSV→Pareto/summary/manifest 工具与回归。

**结果与判定**
- 权威 v2 共 72 rows：54 ok、18 N/A、0 correctness failure；
- TVM fused E2E geomean 为 eager 的 0.546×，median peak ratio 0.512，3/3 Pareto；
- TVM unfused E2E 为 0.481×、0/3 Pareto；`torch.compile` 因 `ContextVar.set` 无法 fullgraph
  capture，未进入性能表；
- PR-12I baseline/correctness PASS，但不宣称 latency headline；下一阶段为 compile amortization，
  PR-12 overall 仍 IN PROGRESS，PR-13 blocked。
- 收尾门禁：focused 9 passed；全量 327 passed、1 skipped；mypy 6 files success；pylint
  6 files 10.00/10；Black/diff check 通过。

**记录**
- `gemini_doc/change_2026-07-14_pr12i_fair_baselines.md`

---

## 2026-07-14：PR-12J compile/load/cache amortization

**主要改动**
- 新增带 canonical signature/target/code-schema/TVM-version key 的 fused CROWN `.so + manifest`
  cache，验证 library SHA；
- 分离 TIR generation、schedule、compile、serialization、module load、first/warm、memory hit 与
  独立进程 disk hit；
- 固定 Q=1..1024，输出 fresh/disk/process/memory-cache 对 eager/chunked 的 break-even 与图表。

**结果与判定**
- v1 暴露 Conv tuple/list manifest 比较 bug，v2 暴露 warm hit 重复 SHA `.so` 的测量污染；均保留；
- authoritative v4 为 3/3 correct、0 hidden recompile；
- Linear/Conv warm 较慢，not amortizable；mini-ResNet 对 eager 的 fresh/disk-first/process
  break-even 为 4668/1062/4450，均超过 Q=1024，且对 chunked 不可摊销；
- 阶段拆分/cache correctness PASS，目标区间摊销 FAIL；PR-12 overall 仍 IN PROGRESS，PR-13
  blocked，下一阶段为 profiler。
- 收尾门禁：focused/integration 5 passed；全量 330 passed、1 skipped；mypy 6 files success；
  pylint 6 files 10.00/10；Black/diff check 通过。

**记录**
- `gemini_doc/change_2026-07-14_pr12j_compile_amortization.md`

---

## 2026-07-14：PR-12K CUPTI activity profile

**主要改动**
- 新增 6 workload×5 backend 的 complete final-bound profiler runner、raw Chrome trace 与
  JSONL→CSV/图/summary/manifest 后处理；
- 排除 profiler inclusive annotation range，保留真实 kernel activity；top-kernel raw count
  使用整数；
- 审计 Nsight Compute 2026.1.1、CUPTI 与 driver counter 权限，不修改系统配置。

**结果与判定**
- 30/30 rows correct；fusion 相对 TVM-unfused 最大整体 launch 降幅仅 1.96%；
- 按 5% device-time 阈值为 3/6 退化、1/6 改善、2/6 中性；
- ncu 实测 `ERR_NVGPUCTRPERM`，因此禁止 bandwidth/cache、occupancy、stall 等硬件 counter
  claim；
- PR-12L 唯一选择 `E_STOP_OPTIMIZING_TIR`；保留 fused candidate，下一阶段转 compile-aware
  Planner；PR-12 overall 仍 IN PROGRESS，PR-13 blocked。
- 收尾门禁：focused 2 passed；全量 332 passed、1 skipped；mypy 2 scripts success；pylint
  2 scripts 10.00/10；Black/diff check 通过。

**记录**
- `gemini_doc/change_2026-07-14_pr12k_cupti_profile.md`

---

## 2026-07-14：PR-12L 冻结停止孤立 TIR 调优

**决策**
- 唯一选择 `E_STOP_OPTIMIZING_TIR`；
- 不再扩 Linear tile、CUDA Graph、chunk-size family 或 Conv capability；
- 不删除 fused backend，由下一阶段的 compile-aware Planner 决定其适用 regime；
- PR-12M 必须使用全新 split、16/32/64/128 MiB/unbounded 多预算和 expected reuse，禁止回写
  旧 final held-out 或回到 TIR 试参。

**边界**
- 本阶段没有 TIR、schedule 或 runtime 代码变化；
- PR-12 overall 仍 IN PROGRESS，PR-13 blocked。

**记录**
- `gemini_doc/change_2026-07-14_pr12l_stop_tir_optimization.md`

---

## 2026-07-14：PR-12M compile-aware 多预算 Planner

**主要改动**
- 新增 capability→budget→risk→amortized latency Planner，输入 expected reuse 与 memory/disk
  cache probability；
- 新增 v3 split/model 无泄漏冻结、calibration/final runner、replay、CSV/figure/manifest；
- baseline runner 可显式读取 calibration 或 final-heldout，合同不变。

**结果与判定**
- calibration/final candidate 各 25/25 correct，冻结/回放 model SHA 一致；
- 75 decisions，72/72 feasible opportunities 选到可行 backend，0 unsafe；
- feasible median/p90/max regret 1.000×/1.000×/1.016×；
- eager/chunked/structured/fused 均被选择，fused 从 cold/mixed 各 1 次增至 warm-Q1024 11 次；
- 3 个 16 MiB capacity failure 单列；PR-12M validated-reduced PASS，PR-12 overall 仍
  IN PROGRESS，下一阶段仅 PR-12N closure audit。
- 收尾门禁：focused 9 passed；全量 340 passed、1 skipped；mypy 7 source files success；
  pylint 6 core/script files 10.00/10；Black/diff check 通过。

**记录**
- `gemini_doc/change_2026-07-14_pr12m_compile_aware_planner.md`

---

## 2026-07-14：PR-12N closure 与 artifact

**审计与交付**
- 重算 I/J/K/M primary hash，核对 manifest、负结果、third-party SHA 与无关来源污染；
- 新增 closure audit 和 reduced Artifact Appendix；
- 更新 Claims Map、执行 memo、PR-12 状态、长期计划和文档索引；
- 创建 annotated tag `pr12-validated-reduced`。

**最终判定**
- PR-12：`VALIDATED-REDUCED`；
- 不升级 full validated：Q≤1024 compile 0/3 可摊销、counter unavailable、收益为局部、无真实
  BaB/VNN-COMP；
- 不降级 mechanism-only：non-toy E2E Pareto、预算价值、自动多 regime selection 与独立
  held-out 已成立；
- PR-13：GO/READY，尚未启动。

**记录**
- `gemini_doc/pr12_closure_audit_2026_07_14.md`
- `gemini_doc/pr12_artifact_appendix_2026_07_14.md`

---

## 2026-07-14：PR-13A Query/State Contract 与真实固定流 Replay

**主要改动**
- 新增 state-versioned `BoundQuery`、完整 compatibility key、四级 state-validity manager、
  owned payload/result 和 fixed recorder/replay；
- 现有 host-side `solve_bab_mlp` 增加可选 observer，保留 parent link 和真实 split/warm-start
  版本，不改变默认 solver 路径；
- 新增 contract/replay tests、artifact runner 和生成式 AI 使用记录。

**结果与判定**
- 真实 solver 生成 8-query two-ReLU smoke：8/8 replay、max abs diff 0、0 loss/duplicate；
- αβ/split capability 固定为 dense，不会误选 plain-CROWN fused TIR；
- PR-13A PASS（foundation only），PR-13 overall IN PROGRESS；下一阶段仅 PR-13B dynamic
  BatchManager，不宣称性能或 non-toy 结果。

**记录**
- `gemini_doc/change_2026-07-14_pr13a_query_contract_fixed_replay.md`
- `gemini_doc/pr13_execution_status.md`

---

## 2026-07-14：PR-13B Dynamic BatchManager

**主要改动**
- 新增 exact-key compatibility buckets、memory first-fit、partial/timeout/deadline flush、host
  wakeup、确定性 OOM 二分和 queue/fill/latency/no-loss metrics；
- 新增 physical αβ dense batch executor，pack/unpack center/spec/split/α/β 并按 query ID 恢复；
- compatibility 加入 input name、perturbation 和 execution-options hash。

**结果与判定**
- 真实 8-query stream 动态 3 batches：8/8、max diff 0、0 loss/invalid；
- OOM fault 8→4+4→2+2+2+2：3 split，最终 8/8；
- PR-13B validated foundation；CPU/逻辑 clock/fault OOM，不是性能或真实 GPU OOM；
- 下一阶段仅 PR-13C same-solver adapter。

**记录**
- `gemini_doc/change_2026-07-14_pr13b_dynamic_batch_manager.md`

---

## 2026-07-14：PR-13C Same-Solver Adapter

**主要改动**
- 新增同步 same-solver query adapter；原 solver 继续拥有 branch/heap/order/termination；
- single/batched bound calls 可选走 query runtime，返回真实 bounds/α/β/branch；
- capability dispatch 在 executor 前拒绝不合法 plain-CROWN forged query。

**结果与判定**
- αβ steps=3/batch=4：original/runtime query IDs 与 per-query bounds/branch/αβ state 7/7，
  solver status/node counters/best bounds 一致，0 loss；
- alpha-only serial 也对齐；forged capability 下 physical αβ executor 调用 0；
- PR-13C validated foundation，下一阶段 PR-13D 双层正式评估；单次 wall time 不作性能 claim。

**记录**
- `gemini_doc/change_2026-07-14_pr13c_same_solver_adapter.md`

---

## 2026-07-14：PR-13D/E Reduced GPU 与 Closure

**主要改动**
- 增加 fixed-stream / true-E2E CUDA benchmark、time/query、throughput、p50/p90/p99、peak
  memory、status/node-count 与公平 baseline 汇总；
- 修复 per-batch αβ Adam gradient scaling、query split lineage version、GPU state hot-path hash；
- 增加 dispatch-plan cache counters 与 non-default CUDA stream event-only 回归；
- 完成 PR-13 closure audit、Artifact Appendix、Claims Map 和状态索引。

**结果与判定**
- fixed 16-query：runtime / per-node 96.52×，runtime / batched original 1.024×；
- hard E2E 16-node：9.93× / 0.980×；safe/unsafe/unknown status 与 node count 一致；
- 0 correctness failure/loss/invalid；custom stream PASS；dispatch cache 1 miss/4 hits；
- runtime rejected/missing/reordered result 采用 fail-closed，不得把失败节点当作已证明；
- 收益主要来自 ordinary batching，non-toy、真实 OOM、PR-12 compiled Planner dispatch 未完成；
- PR-13 以 `VALIDATED-REDUCED` 关闭，不升级 full C3 claim；
- 收尾：326 passed/30 skipped；custom CUDA stream 1 passed；PR-13 Mypy success、Pylint
  10.00/10、changed-file 污染扫描 0 match。

**记录**
- `gemini_doc/change_2026-07-14_pr13d_fixed_e2e_gpu.md`
- `gemini_doc/pr13_closure_audit_2026_07_14.md`
- `gemini_doc/pr13_artifact_appendix_2026_07_14.md`

---

## 2026-07-18：启动 PR-14 Verification-Aware Execution on Real Verification Workloads

**状态纠正**
- 完整审计 research branches 与 annotated tags，确认真实冻结基线为 `57a854b` / tag
  `pr13-validated-reduced`，不再把 `main@263ea81` 误当项目最新状态；
- 从该 tag 创建 `feat/pr14-real-verification`，停止历史 PR-10B.2 路线。

**计划冻结**
- 新增 PR-13 后当前状态文档与 PR-14 执行计划；
- 下一主线为真实 verifier/workload coverage adapter → fixed real-query replay/eligibility →
  complete verification evaluation；
- PR-14 复用已有 `BoundQuery`/recorder/replay，不重写 solver，不恢复孤立 TIR 调优；
- 公平 baseline 固定为 same-solver original batched executor，C3 是否保留为核心贡献由真实
  workload 证据决定。
- PR-13 focused 回归在新分支/当前环境下为 15 passed。

**记录**
- `gemini_doc/current_status_after_pr13.md`
- `gemini_doc/pr14_execution_plan.md`
- `gemini_doc/change_2026-07-18_start_pr14_real_verification.md`

---

## 2026-07-19：冻结 PR-14 Coverage-First 执行模型

**决策**
- PR-14 正式名称改为 `Verification-Aware Execution on Real Verification Workloads`；
- 第一门禁从“先接 executor”收紧为 MLP/CNN/ResNet-block 真实 query coverage profile；
- 新 `VerificationQueryProfile` 只能从 PR-13 `BoundQuery` 派生，禁止创建第二套 query schema；
- 先报告 method/stage 与 backend eligibility，再进入 fixed replay 和 full E2E；
- ASPLOS 当前为执行 `CONDITIONAL GO`，但在真实 workload 闭环前仍是 `ASPLOS-ready NO`。

**记录**
- `gemini_doc/change_2026-07-19_finalize_pr14_coverage_first_plan.md`

---

## 2026-07-20：IR—Planner—Schedule—Runtime 架构重置

**状态纠正**
- 当前 `Bound IR` 仍为占位骨架，runtime LinearOperator 不能代替一等语义 IR；
- `PlanBundle` 与 PR-11/12 局部计划不能代替统一 Plan IR；
- TaskGraph 拓扑循环不能代替 Schedule IR；
- PR-13 ordinary batching 和计划中的 JIT 不再被提前包装成系统贡献。
- 独立审计发现的两处旧 story-freeze 入口已补为历史决定，并显式链接 IR-first 现行契约。

**新路线**
- 冻结 Bound/Plan/Task/Schedule/Query/Runtime 的所有权和数据协议；
- 下一工程分支改为 `feat/compiler-ir-stack-v1`；
- 顺序为 Bound IR → Plan IR → Task/Schedule IR → runtime/backend 迁移 → adaptive evaluation；
- C1/C2 Claims Map 随代码现实降级，PR-10—14 历史数值和负面证据继续保留。

**记录**
- `gemini_doc/boundflow_ir_planner_schedule_runtime_contract_v1_2026_07_20.md`
- `gemini_doc/change_2026-07-20_ir_planner_schedule_runtime_contract.md`

---

## 2026-07-28：Bound IR v1 schema/verifier foundation

**主要改动**
- 将 `ir/bound.py` 从 Any/dict 占位结构升级为 typed value/type/spec/domain/op/graph/module；
- 增加 sample/spec/domain batch axes、lower/upper polarity、representation 与 state identity；
- 增加 SSA/use-def、类型/极性、materialization、reshape、method-state verifier；
- 增加 canonical JSON 与 SHA-256 stable hash；
- 保留旧 runtime `DomainState` 继承兼容，Bound IR 不依赖 runtime/backend/torch/TVM。

**证据与边界**
- focused：15 passed；
- 关键 IBP/CROWN/DAG/PR-14 兼容回归：42 passed；
- 全量：384 passed、1 skipped；
- Mypy 0 issues，Pylint 10.00/10，Black clean；
- 只关闭 IR-1A schema/verifier foundation；builder、reference interpreter、runtime lowering
  和 IR-driven E2E 仍待 IR-1B。

**记录**
- `gemini_doc/change_2026-07-28_bound_ir_v1_schema_foundation.md`

---

## 2026-07-28：Bound IR v1 plain-CROWN dense semantic closure

**主要改动**
- 将真实 CROWN state 明确为 `A_u/b_u/A_l/b_l` 四元 SSA state；
- 增加 residual/concat backward route、bias-once 与 fanout compose 的 typed op/verifier；
- 增加 Task/IBP trace → Bound IR lowering、参数/目标 fingerprint 和 deterministic module；
- 增加不依赖 `crown_ibp.py` 的 dense Bound IR reference interpreter；
- identity/multi-spec MLP、chain CNN、residual/concat fanout 与旧 oracle final bounds 对齐。

**证据与边界**
- 专属 schema/lowering/interpreter：20 passed；
- 相邻 CROWN/DAG/CNN/env：32 passed；
- 全量：392 passed、1 skipped；
- Mypy 0 issues，Pylint 10.00/10，Black clean；
- 关闭 IR-1B dense semantic closure；materialize/structured rewrite、生产 runtime 迁移、
  IR-driven artifact 和 Plan/Schedule IR 仍未完成。

**记录**
- `gemini_doc/change_2026-07-28_bound_ir_v1_plain_crown_lowering.md`

---

## 2026-07-28：Bound IR v1 representation rewrite 与 IR-1 closure

**主要改动**
- verifier 禁止 affine transform/route/compose 隐式改变 coefficient representation；
- 新增 deterministic structured-region rewrite；
- affine region 入口显式 dense→structured cast，ReLU/concretize 前显式 materialize；
- reference interpreter 执行 structured Linear/Conv/Reshape/Add/Concat/Compose；
- dense/structured rewritten IR 在 MLP、CNN、residual、concat 上 final bounds 对齐。

**证据与边界**
- 专属：25 passed；
- 相邻 IR/CROWN/DAG/CNN/LinearOperator/env：47 passed；
- 全量：397 passed、1 skipped；
- Mypy 0 issues，Pylint 10.00/10，Black clean；
- IR-1 最小 reference semantic closure 门禁通过；
- 完整 C1 仍待 Plan/Schedule IR、backend/runtime migration 和 IR-driven E2E artifact。

**下一阶段**
- IR-2 Plan IR v1：PlanTemplate/PlanInstance、统一 decision/verifier、旧 PR-11/12 adapter、
  deterministic plan replay。

**记录**
- `gemini_doc/change_2026-07-28_bound_ir_v1_representation_rewrite.md`

---

## 2026-07-28：Plan IR v1 schema/verifier/replay 与旧计划迁移

**主要改动**
- 新增 typed `PlanTemplate` 静态候选空间和 `PlanInstance` 动态完整选择；
- 分离 region、representation、materialization、backend、domain/spec/sample batch、
  storage/lifetime、state decisions；
- 新增 Bound hash、partition、capability、memory、state、storage alias/lifetime 跨决策 verifier；
- 新增 canonical JSON/hash 和 strict instance replay；
- 为 PR-11/12 的 MaterializationPlan、PlacementPlan、ExecutionCandidate、StoragePlan、
  FusedStep、PlanBundle.meta 提供 adapter/partial/unsupported 迁移结论。

**证据与边界**
- 专属 Plan IR/migration：12 passed；
- 相邻 Bound IR、PR-11/12、storage/env：88 passed；
- 全量：409 passed、1 skipped；
- Mypy 0 issues，Pylint 10.00/10，Black clean；
- 只关闭 IR-2A foundation；reference template builder、query selector、多预算和 artifact
  仍待 IR-2B/2C。

**记录**
- `gemini_doc/change_2026-07-28_plan_ir_v1_schema_and_legacy_migration.md`

---

## 2026-07-28：Plan IR v1 reference builder、selector 与 artifact

**主要改动**
- 新增 typed `ReferencePlanEvidence` 及 region/representation/transition/backend/batch/storage/state
  evidence；
- 从 Bound IR use-def、tensor type 和 state version 自动推导 region boundary、storage lifetime/
  size/alignment 与稳定 candidate/template identity；
- 新增有界 deterministic selector，交叉选择 partition、representation、transition、backend、
  batch、storage/state，并应用 memory/deadline；
- 新增不可变 Bound/Template/Instance artifact、逐文件 SHA-256、精确 replay 与 tamper rejection。

**证据与边界**
- Plan IR 专属：11 passed；连同 legacy migration：16 passed；
- 相邻 Bound IR、PR-11/12、storage/env：92 passed；
- 全量：413 passed、1 skipped；
- Mypy 0 issues，Pylint 10.00/10，Black clean；
- 关闭 IR-2B reference path；真实旧 artifact 批量 assembly/report、query-time state-validity、
  独立 replay CLI 和 IR-2 closure audit 仍待 IR-2C。

**记录**
- `gemini_doc/change_2026-07-28_plan_ir_v1_reference_builder_selector.md`

---

## 2026-07-28：Plan IR v1 state-validity、legacy assembly 与 IR-2 closure

**主要改动**
- `PlanInstance` 新增 canonical query-time state validity，`REUSE` 只接受 exact valid version；
- stale state 选择 recompute，伪造 valid stale state fail closed；
- legacy migration groups 原子加入同一 template，输出 accepted/unsupported/rejected 稳定报告；
- 新增 fresh-process reference artifact generate/replay CLI；
- 新增 legacy record schema inventory；当前 artifacts 扫描 58 文件/4,911 objects，PR-11/12
  planner raw records 为 0。

**证据与判定**
- 专属：21 passed；相邻：97 passed；全量：418 passed、1 skipped；
- Mypy 0 issues，Pylint 10.00/10，Black clean；
- IR-2 reference closure 判定 `VALIDATED-REDUCED`；
- raw historical migration 不可审计，C2 仍待 Schedule/runtime/backend/E2E。

**下一阶段**
- IR-3 Schedule IR v1 + reference executor。

**记录**
- `gemini_doc/change_2026-07-28_plan_ir_v1_closure_audit.md`

---

## 2026-07-28：Schedule IR v1 schema/lowering/verifier foundation

**主要改动**
- 新增 typed `ScheduleModule`、`ScheduleBuffer` 与 budget/allocate/materialize/launch/emit/free
  actions；
- Schedule 锁定 Bound/Template/Instance 三重 hash，并有 canonical JSON/stable hash；
- 新增 PlanInstance→同步 Schedule lowering；
- verifier 检查 storage arena/peak、budget、use-before-def、region/transition 全覆盖、query
  accounting 和 allocation leak。

**证据与边界**
- 专属：3 passed；相邻：100 passed；全量：421 passed、1 skipped；
- Mypy 0 issues，Pylint 10.00/10，Black clean；
- 只关闭 IR-3A foundation；batch/retry/event/state/reference executor/trace replay 仍未完成。

**记录**
- `gemini_doc/change_2026-07-28_schedule_ir_v1_schema_lowering.md`

---

## 2026-07-28：Schedule IR control actions、reference executor 与 trace artifact

**主要改动**
- 新增 typed BatchLoop、Record/WaitEvent、StateLoad/Store/Invalidate、Retry/Fallback、
  RequestReplan；
- verifier 检查 query slice 完整性、cross-stream happens-before、state/Plan 一致、bounded
  OOM ladder 与 replan semantic preservation；
- 新增同步 reference executor、动态 memory ledger、launch attempt trace、canonical replay；
- 新增 immutable Schedule/Trace artifact 和 fresh-process generate/replay CLI；
- Schedule+Bound reference smoke 的 final lower/upper 与直接 Bound interpreter 对齐。

**证据与边界**
- 专属：10 passed；相邻：107 passed；全量：428 passed、1 skipped；
- Mypy 0 issues，Pylint 10.00/10，Black clean；
- IR-3B foundation validated；一等 typed Task IR、逐 Task/backend execution 尚缺，IR-3 未关闭。

**下一阶段**
- IR-3C Task IR v1 + Plan region lowering + per-task reference executor。

**记录**
- `gemini_doc/change_2026-07-28_schedule_ir_v1_control_executor.md`

---

## 2026-07-28：Task IR v1 schema/lowering/linkage foundation

**主要改动**
- 新增不依赖旧 `ir/task.py`/runtime/Any 的 typed TaskIRModule/TaskIRUnit；
- 每个 selected Plan region lower 为 task，显式 op、parameter、external/state dependency、
  memory effect、backend capability/artifact/reference implementation；
- 新增 Task↔Schedule launch 双向逐字段 verifier；
- 新增 deterministic task dispatch trace；
- Schedule artifact 加入 TaskModule/TaskTrace payload 与 hash。

**证据与边界**
- Task IR 专属：4 passed；Task+Schedule：14 passed；相邻：111 passed；
- 全量：432 passed、1 skipped；
- Mypy 0 issues，Pylint 10.00/10，Black clean；
- IR-3C foundation validated；数学语义仍由 whole-Bound oracle 提供，per-task semantic executor
  与 IR-3 closure 尚缺。

**下一阶段**
- IR-3D per-task semantic executor + closure audit。

**记录**
- `gemini_doc/change_2026-07-28_task_ir_v1_foundation.md`

---

## 2026-07-28：Task/Schedule IR v1 逐任务语义闭环

**主要改动**
- 新增 stateful Bound IR session，每个 TaskIRUnit 只执行自己的连续 Bound op partition；
- Task trace 新增真实 boundary value hashes，覆盖 MLP/CNN/residual/concat/structured materialize；
- Task IR 新增 typed input/output tensor/shape constraints；
- Schedule IR 新增 typed Transfer action；
- fresh-process artifact 升级 v2，重算 Task trace 与 final lower/upper hashes。

**证据与边界**
- Task/Schedule/Artifact 专属：24 passed；
- 全量：442 passed、1 skipped、6 warnings；
- IR-3 synchronous reference closure validated-reduced；
- production backend/runtime、state payload reuse、多设备/异步执行仍属于 IR-4+。

**下一阶段**
- IR-4 现有 backend/runtime 迁移，先做 typed PyTorch dense dispatch/cache contract。

**记录**
- `gemini_doc/change_2026-07-28_task_schedule_ir_v1_semantic_closure.md`

---

## 2026-07-28：IR-4A typed backend dispatch/cache foundation

**主要改动**
- 新增锁定 Bound/Plan/Instance/Task/backend/capability/artifact 的 canonical dispatch key；
- 新增只消费 TaskIRUnit 的 PyTorch reference backend adapter；
- prepared-task cache 使用完整 dispatch SHA-256，hit 时继续核对 typed payload；
- Task trace/artifact 新增每个 task 的 backend dispatch key；
- 增加 cache hit、stale hash 与 illegal capability fail-closed 测试。

**证据与边界**
- Task/Artifact 定向：15 passed；
- 全量：443 passed、1 skipped、6 warnings；
- IR-4A foundation validated；chunked/structured/TVM/query-runtime 尚未迁移。

**下一阶段**
- IR-4B chunked/structured typed backend registry，然后迁移 TVM compile cache。

**记录**
- `gemini_doc/change_2026-07-28_ir4a_typed_backend_dispatch.md`

---

## 2026-07-28：IR-4B PyTorch typed backend registry

**主要改动**
- Task lowering 按 BackendKind 生成并验证独立 implementation ID；
- Bound session 新增真实 ReLU→Linear/Conv fused task execution；
- 新增 PyTorch reference/dense/structured/chunked typed registry；
- chunked selected task 真实调用旧 TorchChunkedFusedCrownExecutor；
- 增加 MLP/CNN/structured/CUDA chunked 对齐与非法 non-fused rejection。

**证据与边界**
- IR-4B 专属：5 passed；相邻：42 passed；
- 全量：448 passed、1 skipped、6 warnings；
- PyTorch 三类 backend migration validated；TVM/query-runtime/semantic fallback pending。

**下一阶段**
- IR-4C TVM fused/unfused typed backend 与完整 dispatch-key compile cache。

**记录**
- `gemini_doc/change_2026-07-28_ir4b_pytorch_backend_registry.md`

---

## 2026-07-28：IR-4C TVM typed backend/cache 与 semantic fallback

**主要改动**
- 新增 TVM fused/unfused typed Task registry 和 PyTorch/TVM composite registry；
- fused cache schema v2 把完整 backend dispatch key 加入 memory/disk namespace；
- 增加两个独立 Python 进程的 miss→disk_hit 重放；
- Schedule Retry/Fallback 现在真实切换 semantic backend；
- Task trace 记录 attempted backend ladder 与最终成功 backend。

**证据与边界**
- IR-4C 新增：7 passed；相邻：43 passed；
- 全量：455 passed、1 skipped、6 warnings；
- Query Runtime、state payload 和旧 solver-facing entry migration 仍未完成。

**下一阶段**
- IR-4D Query Runtime + state payload migration，然后执行 IR-4 closure audit。

**记录**
- `gemini_doc/change_2026-07-28_ir4c_tvm_backend_cache_fallback.md`

---

## 2026-07-28：IR-4D typed compiler query 与 exact state runtime

**主要改动**
- 新增只接受已验证 plain-CROWN 子集的 typed compiler query 入口；
- query 经 PlanInstance→TaskIR→ScheduleIR→typed backend 执行并保持原始 ID 顺序；
- 新增绑定 Bound module/value/version/content hash 的 dense runtime state store；
- Schedule StateLoad/Store/Invalidate 具有真实语义，完整 state outputs 可跳过对应 Task；
- legacy PR-13 α/β capability 在 compiler 入口显式 PR-14 No-Go；
- 新增 fresh-process query/state artifact generate/replay。

**证据与边界**
- IR-4D + 相邻定向：42 passed；
- 全量：462 passed、1 skipped、6 warnings；
- Mypy 0 issues，Pylint 10.00/10；
- 不宣称跨 query physical batching；
- 旧 SameSolverQueryRuntime α/β executor 仍待 IR-4 closure audit。

**下一阶段**
- 执行 IR-4 closure audit；在旧 α/β 路径迁移/退役/validated-reduced 边界明确前不进入 IR-5。

**记录**
- `gemini_doc/change_2026-07-28_ir4d_compiler_query_state_runtime.md`

---

## 2026-07-28：IR-4E PR-13 query migration 与 IR-4 closure

**主要改动**
- `BoundQuery` 新增唯一 compiler-eligible 的 `plain_crown_typed_ir` capability；
- 新增 PR-13 query identity 与完整 compiler payload 的交叉验证 adapter；
- PR-13 DynamicBatchManager 负责 compatibility/deadline/memory/OOM/order，executor
  只进入 PlanInstance→TaskIR→ScheduleIR→typed backend；
- legacy α/β SameSolver runtime 默认拒绝，仅历史脚本显式 opt-in；
- fresh-process query/state artifact 升级 v2。

**证据与边界**
- IR-4E/PR-13 定向：24 passed；
- 全量：464 passed、1 skipped、6 warnings；
- Mypy 0 issues，Pylint 10.00/10；
- IR-4 narrow plain-CROWN scope validated-reduced closure；
- α/β/split external integration 仍明确不成立。

**下一阶段**
- IR-5 adaptive PlanInstance 与公平 held-out 对比；IR-6 继续 gated。

**记录**
- `gemini_doc/change_2026-07-28_ir4e_pr13_query_migration_closure.md`

---

## 2026-07-28：IR-5A adaptive PlanInstance query context

**主要改动**
- selector 新增 query distribution、expected query count 与 exact compile-cache context；
- 按 runtime + uncached compile/setup amortization 选择 plan；
- deadline 使用同一 amortized latency；
- compiler runtime 支持 per-query memory/budget/deadline/selection context；
- context 进入 PlanInstance identity/provenance 与 Plan/Task cache namespace。

**证据与边界**
- 定向：29 passed；
- 全量：466 passed、1 skipped、6 warnings；
- cold/repeated/warm-cache 可切换不同合法 plan；
- 尚无 fixed/local/global/oracle 新 held-out 证据，IR-5 不关闭。

**下一阶段**
- IR-5B 公平策略 evaluator 与 typed held-out artifact。

**记录**
- `gemini_doc/change_2026-07-28_ir5a_adaptive_plan_context.md`

---

## 2026-07-28：IR-5B 公平 adaptive policy evaluator

**主要改动**
- 新增 frozen context/plan observation 评估契约；
- fixed/local/global/oracle 共享 legality、budget、cache 与 measured outcomes；
- 统一输出 p50/p90/p99、TTV、peak 和 Oracle regret；
- fixed 不可行时明确记录 infeasible；
- 新增 fresh-process synthetic contract artifact。

**证据与边界**
- 定向：25 passed；
- 全量：468 passed、1 skipped、6 warnings；
- artifact 明确标注 synthetic contract，不是 held-out 性能证据；
- IR-5 仍 pending。

**下一阶段**
- IR-5C typed measured held-out workload/artifact。

**记录**
- `gemini_doc/change_2026-07-28_ir5b_fair_policy_evaluator.md`

---

## 2026-07-28：IR-5C0 typed measured workload foundation

**主要改动**
- 新增正式 typed MLP benchmark workload/candidate builder；
- 同语义生成 reference 与 fused backend 的完整 Plan/Task/Schedule；
- evaluator 分离 predicted/measured compile，避免 held-out leakage。

**证据与边界**
- 定向 3 passed；
- reference/dense final bounds 一致、PlanInstance hash 不同；
- 尚无 measured held-out 数字，IR-5C 继续。

**记录**
- `gemini_doc/change_2026-07-28_ir5c0_typed_measured_workload_foundation.md`

---

## 2026-07-28：IR-5C1 leakage-free measurement runner

**主要改动**
- 新增 typed candidate cold/warm/CUDA peak/TVM compile-phase 测量；
- calibration-only latency/setup model 明确拒绝 held-out leakage；
- 冻结 workload split、resource context 与 query/cache contexts；
- 新增目录级 artifact manifest、integrity replay 与 reference semantic replay。

**证据与边界**
- 定向 4 passed，Mypy 0 issues，Pylint 10.00/10；
- 开发期后验 memory-budget pilot 明确废弃；
- 本切片不宣称最终 held-out 性能，fresh CUDA v2 artifact 仍 pending。

**记录**
- `gemini_doc/change_2026-07-28_ir5c1_leakage_free_measurement_runner.md`

---

## 2026-07-28：IR-5C2 CUDA typed held-out（PARTIAL）

**主要结果**
- 16/16 typed CUDA candidate measurements correctness allclose；
- 8 contexts × 4 policies，Global 8/8 feasible；
- Global Oracle regret p50/p90/max 为 1.000×/1.00766×/1.00766×；
- 64 MiB 选择 PyTorch dense，冻结低内存预算选择 TVM fused；
- artifact manifest 绑定 `1be9c19`，integrity + semantic replay 通过。

**未关闭门禁**
- calibration/held-out 仍属于同一 MLP family；
- ordinary batching/fair batched-original 未接入；
- 低内存切换是 feasibility-driven，尚非多个可行候选间的 Global 优势；
- CNN/残差/non-toy 与跨层收益归因仍缺。

**下一阶段**
- IR-5C3 independent workload-family + fair batching baselines；不启动 IR-6。

**记录**
- `gemini_doc/change_2026-07-28_ir5c2_cuda_heldout_partial.md`

---

## 2026-07-28：IR-5C3A independent CNN workload family

**主要改动**
- 新增 two-convolution chain-CNN typed workload builder；
- 新增显式 `chain_cnn` measured spec 与 MAC work feature；
- 支持 MLP calibration→CNN held-out 的同一 measurement/evaluator 接口。

**验证与边界**
- CPU reference/dense 对齐；
- CUDA reference/dense/chunked/TVM fused 对齐且 TVM trace 命中 fused；
- 定向 4 passed，Mypy 0 issues；
- 尚无 fair batching baseline 或新 held-out artifact。

**记录**
- `gemini_doc/change_2026-07-28_ir5c3a_independent_cnn_family.md`

---

## 2026-07-28：IR-5C3B fair batching evaluator/runner contract

**主要改动**
- v2 evaluator 显式加入 fixed-single、ordinary-batching、batched-original；
- compiler selection pool 与全方案 Oracle pool 分离；
- physical batch latency 按 query 数归一，compile/setup 不除；
- 新增 MLP calibration→CNN held-out runner 与 batch-first-query semantic gate。

**验证与边界**
- 定向 2 passed，Mypy 0 issues，Pylint 10.00/10；
- warm=1 pilot 只作 runner smoke，不进入证据链；
- 正式 9-sample fresh CUDA artifact 仍 pending。

**记录**
- `gemini_doc/change_2026-07-28_ir5c3b_fair_batching_contract.md`

---

## 2026-07-28：IR-5C3C fair architecture-held-out（VALIDATED-NO-GO）

**正式结果**
- MLP calibration→chain-CNN held-out，8 compiler rows/2 original/2 batch checks 全 correct；
- Global 8/8 feasible，但 fair Oracle regret p50/p90/max =
  68.065×/70.263×/70.263×；
- batched-original 始终为 Oracle；
- 64/512 MiB 均选 chunked，无多预算切换，无 memory Pareto。

**归因与判定**
- profile 指向 query hot path 重复 validate/hash/dispatch-key；
- 当前 IR-5 v1 VALIDATED-NO-GO，IR-6 blocked；
- 唯一补救为一次验证、query-time 复用的 prepared execution capsule，并要求新 final split。

**记录**
- `gemini_doc/change_2026-07-28_ir5c3c_family_fair_nogo.md`

---

## 2026-07-28：IR-5D prepared execution capsule

**主要改动**
- 静态 Bound/Task/Plan validate、hash 与 primary/fallback dispatch key 移出 query hot path；
- prepared program 冻结参数快照，动态 Schedule 只允许 query binding 重写；
- 新增 AUDIT/PRODUCTION trace mode，timed production path 跳过中间 tensor SHA；
- compiler query cache 和 measured runner 接入 prepared capsule；
- 新增 from-forward-trace legacy baseline，使双方只计 CROWN backward。

**验证与边界**
- 全量 `476 passed, 1 skipped`，Mypy 0 issues，Pylint 10.00/10；
- 已消费 CNN calibration 诊断中最快 typed/legacy median 比值为 `0.880×`/`0.896×`；
- 诊断不是 fresh artifact，IR-5C3 `70.263×` No-Go 不撤销，IR-6 仍 blocked；
- 下一步必须冻结并一次性消费新的 residual-CNN final split。

**记录**
- `gemini_doc/change_2026-07-28_ir5d_prepared_execution_capsule.md`

---

## 2026-07-28：IR-5E residual final protocol freeze

**主要改动**
- 新增带真实 fanout/add merge 的 residual-CNN typed workload 与 measured spec；
- fair batching/measurement 支持 chain-CNN 与 residual-CNN；
- runner 新增 CUDA-only `residual-final-v2` suite；
- 冻结 chain-CNN calibration→全新 residual-CNN final 的 shapes/IDs/seeds；
- final baseline 固定为 from-forward-trace，新增 p90≤1.20 与双 workload Pareto 字段。

**验证与边界**
- residual CPU reference/dense 与临时 CUDA 四后端语义对齐；
- 旧 v1 artifact replay 兼容；
- 临时 smoke 的 `7301/7302` 已废弃，正式 `7401/7402` 未执行；
- 正式 artifact 必须在 protocol commit 后一次性生成，失败不得按 final 数据继续调参。

**记录**
- `gemini_doc/change_2026-07-28_ir5e_residual_final_protocol_freeze.md`

---

## 2026-07-28：IR-5F residual-final-v2 protocol invalid

**结果**
- clean protocol commit `b3762bf` 上首次正式生成在 fixed-single semantic gate 中止；
- 参数完全一致，但同 seed、不同 batch shape 生成的 input center 不是前缀；
- 两个 workload 的 input max diff 为 `3.73509`/`2.16740`，不是浮点 tolerance 问题；
- 未生成 summary/manifest，未读取 held-out 性能数字。

**处置**
- v2 标记 PROTOCOL-INVALID，`7401/7402` 永久退役；
- 保留 strict semantic gate，只允许显式 slice batched input 的方法学修复；
- 修复必须升级 suite/schema、旋转 fresh final identity 后重新冻结。

**记录**
- `gemini_doc/change_2026-07-28_ir5f_residual_final_v2_protocol_invalid.md`

---

## 2026-07-28：IR-5G exact-input-slice residual final v3 freeze

**主要改动**
- convolutional builder 支持显式 `input_center`，严格校验 shape/dtype；
- fixed-single 从 batched query zero 精确 clone，不再依赖 RNG 前缀假设；
- semantic gate 前新增 `torch.equal` input identity 门禁并写入 artifact；
- 冻结 v3 schema 与 fresh `7501/7502` identities。

**边界**
- 不改 backend、预算、final shape、p90/Pareto 阈值；
- 未读取 v2 held-out timing；
- dummy residual test 通过，v3 exact identities 未执行；
- protocol commit 后只允许一次正式生成。

**记录**
- `gemini_doc/change_2026-07-28_ir5g_exact_input_slice_v3_freeze.md`

---

## 2026-07-28：IR-5H residual-final-v3（VALIDATED-NO-GO）

**正式证据**
- fresh chain-CNN calibration→residual-CNN final，8+8 measurements、48 outcomes；
- exact input identity、compiler/baseline correctness、Global 8/8 feasibility 全通过；
- manifest 绑定 `971a317`，integrity + semantic replay 通过。

**失败门禁**
- Global p50/p90 regret = `1.00385×/1.26160×`，p90 超过 `1.20×`；
- color warm-cache 错选 TVM，regret `1.26160×`；
- gray compiler frontier 只有 TVM 一个点，双 workload Pareto 失败；
- 64/512 MiB 均选 dense，无 multi-budget switch。

**判定**
- IR-5 最终保持 VALIDATED-NO-GO，IR-6 不启动；
- 停止当前 ASPLOS system-performance 路线，不再旋转 final 或按 final 调参。

**记录**
- `gemini_doc/change_2026-07-28_ir5h_residual_final_v3_nogo.md`

---

## 2026-08-03：IR-5 路线封存与发布交接

**主要改动**
- 修复权威当前状态中“下一步补 IR-5 / 当前进入 IR-5 / 只允许 IR-5D”的过期指令；
- 总体计划置顶最终 residual-v3 No-Go，而不是中间 IR-5C3 状态；
- 执行备忘录把 IR-5A—H 的中间“待执行”统一为历史完成语义；
- 新增 closure 范围、外部 replay 命令和新研究路线准入条件。

**最终状态**
- IR-1—4 validated-reduced 机制保留；
- IR-5 Global p90/Pareto 门禁失败，最终 VALIDATED-NO-GO；
- IR-6 不启动，当前 ASPLOS system-performance 路线封存；
- 后续只允许独立的真实 Verifier IR correctness/integration 研究路线。

**本机复核**
- residual-final-v3 integrity replay：PASS；
- 全量回归：`445 passed, 37 skipped`；
- 当前 NVIDIA 驱动不可通信，semantic replay 未现场复跑，沿用正式 artifact 的历史通过证据并明确该审计边界。

**记录**
- `gemini_doc/change_2026-08-03_ir5_route_closure_and_publish.md`

---

## 2026-08-03：启动真实 Verifier IR 集成路线

- 从 IR-5 closure 新建 `feat/real-verifier-ir-integration-v1`；
- 冻结 external intermediate bounds、adaptive relaxation policy 与 activation-BaB
  external exact backend 的 Bound/Plan/Task/Schedule 所有权；
- 独立复现 ResNet 根因：external pre-activation bounds + adaptive slope 将 max diff
  从 `796.765` 降到 `2.15e-6`，sign 从 `3/9` 提升到 `9/9`；
- 本阶段只确立 correctness 契约，不形成性能 claim。

**记录**
- `gemini_doc/change_2026-08-03_start_real_verifier_ir_integration.md`

---

## 2026-08-03：RVIR-1 External Intermediate-Bound Semantics

- ReLU intermediate-bound source 与 lower-slope policy 成为稳定 Bound IR 字段；
- PR-14 capture 拥有逐 ReLU external bounds、identity 与 aggregate hash；
- count/order/shape 失配 fail closed，fixed replay 使用 external bounds + adaptive policy；
- ResNet-2B prop0 CPU fresh replay 从 max diff `796.765` 修复到 `3.10e-6`，sign `9/9`；
- focused IR/compiler/PR-14 回归 `89 passed`，不形成 CUDA/性能 claim。

**记录**
- `gemini_doc/change_2026-08-03_rvir1_external_intermediate_semantics.md`

---

## 2026-08-03：RVIR-2 Typed External Verifier Calls

- Bound/Plan/Task/Schedule 新增强类型 external αβ-CROWN exact-call 路径；
- α/β/split identity、requested lower/upper 与 external semantics ownership 显式化；
- profiler 通过 typed schedule 调用原 provider，并保留嵌套 query parent lineage；
- 真实 CPU BaB observer on/off 对照均访问 380 domains、final lower 一致；
- 377/377 调用编译、调度并完成，343 个 activation 调用 effective method 均为 αβ-CROWN；
- 当前只形成 correctness/integration claim，不形成性能 claim。

**记录**
- `gemini_doc/change_2026-08-03_rvir2_typed_external_calls.md`

---

## 2026-08-03：RVIR-3/4 CPU correctness artifact

- 冻结 394/394 历史 activation query 的五层 typed IR admission/hash；
- 冻结真实 CPU 377/377 exact-call dispatch、347 parent links 与 lower-only identity；
- observer on/off status、380 domains 与 final lower 一致；
- 合并 ResNet lower max diff `3.10e-6`、sign `9/9` correctness evidence；
- artifact 自包含 query identity，可 fresh-process 重算，不形成 CUDA/性能 claim；
- 历史 split values、requested polarity、parent lineage 缺失均逐行标注，不补写证据。

**记录**
- `gemini_doc/change_2026-08-03_rvir3_cpu_correctness_artifact.md`

---

## 2026-08-03：真实 Verifier IR 路线关闭

- RVIR-1—4 在 CPU correctness/integration 范围内全部通过；
- 全量 `452 passed, 37 skipped`，artifact fresh-process replay 通过；
- 更新执行备忘录、当前状态、claims map、总体计划与 README 阅读顺序；
- 最终判定 VALIDATED-REDUCED；历史 fused `0/394`、v1 identity limitation、GPU/性能缺口
  均保留；
- IR-5 仍为 VALIDATED-NO-GO，IR-6 不启动，ASPLOS-ready 仍为 NO。

**记录**
- `gemini_doc/real_verifier_ir_integration_closure_2026_08_03.md`

---

## 2026-08-03：默认启用 DocOps Logic

- 初始化 `.docops` 低 token 状态、规则、事件与知识卡；
- 后续每次代码/文档/计划/流程修改必须记录 `ch`，验证后记录 `va`；
- 交接前必须执行 DocOps soft lint；
- RVIR PR #4 的 closure/validation 已补录。

**记录**
- `gemini_doc/change_2026-08-03_adopt_docops_default.md`

---

## 2026-08-03：RVIR 外部审计交接

- 汇总 RVIR 起点、6-commit 链、实现、artifact、验证与 claim 边界；
- 明确区分 fused `0/394`、typed admission `394/394`、实时 execution `377/377`；
- 给出独立审计顺序与审核输出格式；
- 通过 DocOps exchange/handoff 形成异步审计入口。

**记录**
- `gemini_doc/change_2026-08-03_rvir_external_audit_handoff.md`

---

## 2026-08-03：RVIR 在线原始证据 replay v2

- 重建固定 αβ-CROWN commit 与官方 simple-MLP CPU workload；
- 新运行的 377 条 queries/typed records SHA 与原 RVIR 摘要逐字节一致；
- v2 artifact 冻结两份原始 JSONL，fresh replay 重算 lineage、accounting 与五层 IR hash；
- v1 artifact 保持不可变并继续支持 replay；CPU-only、无性能 claim 的边界不变。

**记录**
- `gemini_doc/change_2026-08-03_rvir_online_raw_replay_v2.md`

---

## 2026-08-04：Production Schedule IR + Memory P0

- 新增 deterministic production Schedule ownership/memory audit 与 semantic replay；
- 复算 residual 2 workload × 4 backend：完整 Bound-op/arena/launch coverage 成立，但
  MaterializeAction、storage choice、预算决策切换均缺失；
- 复算 VNN-COMP ResNet 51/51 activation call 五层 IR hash，确认主计算仍为单 external op/
  launch；
- P0 最终 `NO_GO`，不启动 production schedule-memory headline；下一分支冻结为
  `feat/native-real-network-bound-ir-v1`。

**记录**
- `gemini_doc/BOUNDFLOW_PRODUCTION_SCHEDULE_MEMORY_P0_CHANGELOG_2026_08_04.md`

---

## 2026-08-04：Native Real-Network Bound IR v1

- 将固定 VNN-COMP ResNet2B 的 17-op Primal graph lower 为 21-op native plain-CROWN
  Bound graph，并生成 21 个 Task units 和 21 次 Schedule launch；Bound/Task external-call 为 0；
- external intermediate bounds 新增 portable safe-load payload、逐 tensor/aggregate digest 与
  tamper rejection；aggregate identity 进入 ReLU state version 和 Plan provenance；
- 新增 native compiler integration、pinned input fetch、artifact generate/replay 与 tests；
- real CPU final lower 对 αβ-CROWN oracle max diff `7.152557373046875e-07`、sign 9/9，五层
  hash fresh replay 一致；
- 状态为 correctness/compiler ownership VALIDATED-REDUCED；单 storage/batch、0
  materialization、external forward bounds 与无 GPU/performance claim 的限制保留。

**记录**
- `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_BOUND_IR_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-04：Native Real-Network Memory Plans v1

- 在同一固定 ResNet Bound IR/PlanTemplate 中加入 retain-all 与 lifetime-reuse 两个 storage
  candidate；预算从 `1,860,912` bytes 降到 `442,656` bytes 时切换 PlanInstance 与 Schedule；
- lifetime-reuse 使用 verified last-use 做 deterministic physical alias，真实图有 386 对合法
  alias；Task runtime 按 selected lifetime 删除中间引用，85 个值在最终 Task 前释放；
- 双计划 lower/upper bitwise equal，对 external lower max diff
  `7.152557373046875e-07`、sign 9/9；`442,655` bytes 以
  `memory_budget_exceeded` fail closed；
- 新增 deterministic generate/replay artifact，并确认 NRIR-1 原 artifact 五层 hash 不变；
- 状态为 CPU storage-plan correctness/ownership VALIDATED-REDUCED；不声称 CUDA allocator
  peak、OOM rescue、latency、speedup、real materialization 或 sliced batching。

**记录**
- `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_MEMORY_PLANS_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-04：Native CUDA Physical-Memory Protocol v1

- 冻结 retain/reuse 的 5 fresh-process repeats × 5 warmup × 20 measured、alternating order、
  prepared lower-only timing 与 allocated/reserved counter；
- 新增模型/intermediate-bound/environment/worker/IR/result identity、20% memory reduction、
  1.20× latency ratio、raw→summary→manifest replay 门禁；
- 新增 prepared storage capsule，避免每次 query 重做静态 Schedule/Plan validation/hash；
- 本机 CUDA driver/device unavailable：probe artifact 为 `performance_claimed=false`，正式
  benchmark 在创建输出目录前 exit 2，未生成任何 measured row 或性能结论；
- 聚焦 `17 passed`，全量 `484 passed, 37 skipped`；Mypy clean、Pylint 10.00/10、Black/
  diff check 通过；下一路线为 representation semantic binding bridge。

**记录**
- `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_CUDA_MEMORY_PROTOCOL_V1_CHANGELOG_2026_08_03.md`

---

## 2026-08-04：Native Representation Semantic Binding v1

- 新增 dense/structured-affine 两个全局 source Plan policy；用 storage-compatible prefix
  pruning 避免真实 21-region 图的指数混合枚举，同时保持可行 Plan 集合不变；
- 新增 fail-closed binder，将每个 selected transition 与 source Schedule
  `MaterializeAction`、rewritten execution Bound op 一一绑定，并为新 Bound hash 重建独立
  Plan/Task/Schedule stack；
- 固定 ResNet structured 路径插入 14 cast + 14 materialize，21-op source 变为 49-op execution，
  49/49 ops 均进入 Task 与 Launch；
- dense/structured lower 最大差 `9.5367431640625e-07`，均对 external lower allclose、sign
  9/9；artifact digest、semantic replay 与 tamper tests 通过；
- 当前 structured operator/storage 仍为 dense-equivalent，明确禁止 compression、memory、
  latency、CUDA、OOM、Pareto 或 speedup claim；下一路线为 real-network sliced batch execution。

**记录**
- `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_REPRESENTATION_BINDING_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-04：Native Real-Network Sliced Batch Execution v1

- `PlanSelectionContext` 新增互相独立的 domain/spec/sample 上限，非默认值进入 PlanInstance
  provenance/hash；历史默认路径 identity 保持兼容；
- source Schedule 对 reduced spec candidate 生成连续半开 objective ranges，并由 verifier 强制
  无重叠、完整覆盖和 width 上限；
- 每个 selected range 编译独立 native Bound/Plan/Task/Schedule child，runtime 校验完整 objective
  digest 后顺序执行并沿 spec 轴聚合 lower/upper；
- 固定 ResNet full=1×21-op，sliced=3×21-op、ranges `0:3/3:6/6:9`、63 Task/Launch；
  full/sliced max diff `1.9073486328125e-06`，external sign 9/9；
- 新增 generate/replay artifact 与 rehashed range/query/trace/gate/claim tamper tests；当前只关闭
  spec-axis correctness/ownership，不声称 memory/latency/CUDA/OOM/Pareto/speedup；
- domain/sample 与 representation × batch composition 仍 pending，后者为下一联合门禁。
- 新旧 native/Plan/Task/Schedule 聚焦 `89 passed`；全量 `508 passed, 37 skipped`；Black、
  Mypy、Pylint 10.00/10、diff check 通过。

**记录**
- `gemini_doc/BOUNDFLOW_NATIVE_REAL_NETWORK_SLICED_BATCH_EXECUTION_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-04：Native Representation × Batch Composition v1

- 同一 source template 联合 representation/storage 与 full/spec-size-3 batch candidates；memory
  budget × max spec 由 generic selector 选择 dense/structured × full/sliced 四组合；
- 新增 required storage policy context/provenance/verifier，source policy 显式传播到 child，避免
  child shape 改变后重新打分换 policy；
- 四组合共享 source Bound/PlanTemplate、拥有四个 PlanInstance/Schedule；真实 ResNet child
  op/task/launch=`21/63/49/147`，structured 28 transitions 与 sliced exact ranges 同时保留；
- 四路径 external lower max diff 均 ≤`1.9073486328125e-06`、sign 9/9；artifact generate/replay
  与 tamper gates 通过；
- 聚焦 `103 passed`，全量 `522 passed, 37 skipped`；Black/Mypy clean、Pylint 10.00/10、
  diff check 通过；
- 状态为 cross-axis correctness/ownership VALIDATED-REDUCED；跨 query/domain batching、cache、
  memory/latency/CUDA/OOM/Pareto/speedup 仍 pending。

**记录**
- `gemini_doc/BOUNDFLOW_NATIVE_REPRESENTATION_BATCH_COMPOSITION_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-04：Native Repeated-Query Batching and Cache v1

- 将 frozen ResNet 的 9 个不同 property objectives 显式建模为 9 条 query，冻结 ID、objective
  digest、range、workload/input/state identity；
- packed path 以 size-3 执行 3 child，same-policy serial reference 分别执行 9 child，并按 exact
  ranges 恢复 9/9 query results；
- exact in-process compile cache first miss/second hit；objective content、query order、state
  identity 变化均产生不同 key/miss；
- packed/cache hit bitwise equal；packed/serial max diff `3.2186508178710938e-06`；二者对
  external 均 allclose、sign 9/9；
- artifact generate/replay 与 query/cache/result/semantic/claim tamper tests 通过；聚焦
  `121 passed`，全量 `540 passed, 37 skipped`；Black/Mypy clean、Pylint 10.00/10、diff 通过；
- 状态为 repeated-query formation/packing/cache/lineage VALIDATED-REDUCED；3 vs 9 是机制计数，
  不声称 speedup；BaB parent/child domain state 与性能证据仍 pending。

**记录**
- `gemini_doc/BOUNDFLOW_NATIVE_REPEATED_QUERY_BATCHING_CACHE_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-04：Native BaB Input-Domain Batching v1

- 固定 root box 按前三个正宽输入坐标三层确定性二分，形成 8 个不同 leaf queries 与完整
  parent/branch lineage；
- 每个 leaf 独立重算 IBP interval/ReLU state；parent state 只允许 `warm_start_only`，typed
  compilation/runtime trace 禁止作为 child exact input；
- source Plan 加入 full-domain 与 size-4 candidates；Schedule 实际形成 `[0,4)/[4,8)`，编译并
  执行两个 child IR stacks；full 与 same-policy serial 分别执行 1/8 stacks；
- fixed ResNet packed/full/serial 8×1 lower/upper bitwise equal，8/8 result/parent lineage 恢复；
  artifact generate/replay 与 lineage/range/state/claim tamper gates 通过；
- 聚焦 `19 passed`；全量 `559 passed, 37 skipped`；Black/Mypy clean、Pylint 10.00/10、
  diff check 通过；
- 只关闭 input-box domain batching/state ownership，不声称 ReLU/β BaB、queue/prune/termination、
  memory/latency/CUDA/OOM/Pareto/speedup；下一门禁为 native ReLU-split BaB queue/state v1。

**记录**
- `gemini_doc/BOUNDFLOW_NATIVE_BAB_DOMAIN_BATCHING_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-04：Native ReLU-Split BaB Queue v1

- CROWN Bound IR 新增 first-class per-ReLU int8 split inputs，内容 identity 进入 ReLU attrs、
  state version 与 Bound hash；Plan/Task/Schedule 支持 mixed float32/int8、split workload/
  capability 与 actual execution；
- runtime exact binding 对 split key/shape/dtype/device/range/hash、active/inactive preactivation
  fail closed；local split-constrained IBP 与 external verifier provenance 分离；
- 新增 deterministic widest-ambiguous branch、best-first bounded queue、typed parent/branch/prune/
  expand/terminal trace 与 node/depth budget；child 仅继承离散 split，exact state 独立重算；
- queue node batches 实际编译/执行 representation-bound Bound/Plan/Task/Schedule stacks；toy complete
  queue packed/serial 为 5/15 stacks，固定 ResNet bounded queue 为 3/7 stacks；
- fixed ResNet 执行 7 nodes、3 expand、4 frontier 并正确报告 budget-exhausted/not-claimed；packed/
  serial lower/upper max diff `1.8310546875e-04/1.220703125e-04`，queue/split identity 一致；
- 新增 generate/replay artifact、manifest、runtime/artifact/tamper tests；聚焦 `68 passed`、
  全量 `577 passed, 37 skipped`；Black/Mypy/Pylint/diff 通过；
- 只关闭 first-class split/queue/control-flow correctness ownership，不声明 α/β optimization、
  完整 verdict、memory/latency/CUDA/OOM/Pareto/speedup。下一门禁为 native α/β state 与
  warm-start validity。

**记录**
- `gemini_doc/BOUNDFLOW_NATIVE_RELU_SPLIT_BAB_QUEUE_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-04：Native Alpha/Beta Optimization State v1

- optimized ReLU BoundOp 新增 split/alpha/beta typed inputs；每个 fixed ResNet ReLU 使用 7-port
  contract，state identity 进入 Bound/Plan/Task/Schedule 与 compiler hash；
- reference interpreter 实际消费 optimized ambiguous lower slope 和 `-beta*split` lower-dual
  coefficient；key/shape/dtype/device/range/hash/linkage tamper fail closed；
- 新增 model/input/objective/intermediate/split/policy/payload scope 与 warm-start classifier：exact
  scope 可 exact reuse，monotonic split refinement 仅 initialization，reversal/removal/drift 拒绝；
- fixed ResNet 共 19 inputs、6 optimized ReLU ops、21 Task/Launch；native/legacy bounds max diff
  `0.0/0.0`，beta sum `0.04999999701976776`，对 zero-beta lower 改善 `0.34039306640625`；
- artifact generate/replay hash
  `302f536685885e75248582698589d49f667d7709ca3258c043310e02278e6884`；聚焦 `50 passed`，
  全量 `591 passed, 37 skipped`；Black/Mypy clean、Pylint 10.00/10、diff check 通过；
- 只关闭 frozen-state/beta/warm-start correctness；Adam loop 尚未编译，无完整 BaB/verdict/
  performance claim。下一门禁为 native optimizer-step Task/Schedule control v1。

**记录**
- `gemini_doc/BOUNDFLOW_NATIVE_ALPHA_BETA_OPTIMIZATION_STATE_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-04：Native Alpha/Beta Optimizer-Step Schedule v1

- 新增 typed optimizer Plan/Task/Schedule IR；Plan 绑定 NRIR-10 10 个 source compiler hashes、
  initial state/scope、policy、ReLU keys、warm-start kind 与固定 step budget；
- 固定步数 lower 为 evaluate/reduce/backward/Adam/project/select-best Task/Action；executor 严格按
  Schedule 执行，并记录逐 value hash chain、alpha/beta gradient、projection、evaluation 与 best state；
- toy 2-step 为 13 actions，与 legacy bounds/alpha/beta 逐张量一致，selected state 再经 NRIR-10
  native compiler stack 执行一致；order/linkage/hash/scope/warm-start tamper fail closed；
- fixed ResNet 1-step 为 8 actions；alpha/beta gradient L1=
  `169.23175295069814/12.862210273742676`，Schedule/legacy/final native max diff 均为 `0.0`；
- artifact generate/replay hash
  `31261b63d80a7b11dc14484ddab2fe37bbafcc86866aaeaaa53d6af70ea40a19`；聚焦 `35 passed`，
  全量 `612 passed, 37 skipped`；Black/Mypy/Pylint/diff 全过；
- 只关闭 fixed-step optimizer control ownership；dynamic early stop、multi-node BaB integration、
  complete verdict 与 performance 仍 pending。下一门禁为 optimizer Schedule × ReLU-split queue。

**记录**
- `gemini_doc/BOUNDFLOW_NATIVE_ALPHA_BETA_OPTIMIZER_STEP_SCHEDULE_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-04：Native Optimized ReLU-Split BaB v1

- 新增独立 optimized queue runtime：每个 node batch 执行 8-action optimizer Schedule，selected
  alpha/beta state 再执行 21-task native Bound/Plan/Task/Schedule stack；
- parent per-node selected states 按 child batch layout 重组并重建 scope，只允许 monotonic-refinement
  initialization；parent exact state 永不作为 child exact input；
- toy complete queue 为 15 nodes，packed/serial 5/15 stacks，queue/bounds/state hash 一致；
- fixed ResNet 为 7 nodes/3 expands/4 frontier，packed/serial 3/7 stacks；bounds max diff=
  `1.220703125e-04/1.8310546875e-04`，alpha/beta tensor max diff=
  `4.172325134277344e-07/7.450580596923828e-09`；exact batch-layout state hash 不伪称相等；
- active child beta gradients 非零，每个 selected state 对 native re-execution diff 为 0；artifact
  replay hash=`e813826c8fe74161505ab2379b37fa67247fd40c3bd0cb8f82b77880ce403787`；
- 聚焦 `18 passed`、全量 `630 passed, 37 skipped`；Black/Mypy/Pylint/diff 全过；
- 只关闭 optimized queue integration；fixed run 仍 budget-exhausted/not-claimed。下一门禁为 sound
  property termination/verdict，不启动性能路线。

**记录**
- `gemini_doc/BOUNDFLOW_NATIVE_OPTIMIZED_RELU_SPLIT_BAB_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-04：Native Property Termination and Verdict v1

- 新增独立 three-state verdict runtime：verified 必须所有 leaf sound-pruned，任何
  frontier/depth/unproven prune 保持 unknown；
- 新增 concrete primal Task IR executor，覆盖 ResNet 所需 conv2d/ReLU/residual/flatten/linear
  语义并保留 intermediate value trace；
- unsafe 只能由重执行通过的 concrete witness 产生；input box、ReLU split path、
  output/objective 与 tensor/value-trace hashes 全部绑定；
- toy verified/unsafe/unknown 与非 root split witness 通过；固定 ResNet center objective=
  `0.8564349412918091`，7-node/4-frontier 仍为 explicit unknown；
- artifact generate/replay hash=
  `9e3dceed23c8759c910938ba7c9f84caaeb949c8f19b72fab104ce4e1b733405`；聚焦
  `19 passed`、全量 `649 passed, 37 skipped`；Black/Mypy/Pylint/diff 全过；
- 只关闭 verdict soundness/control ownership；candidate search、multi-clause property、timeout/
  dynamic early stop、real complete closure 和性能证据仍 pending。

**记录**
- `gemini_doc/BOUNDFLOW_NATIVE_PROPERTY_TERMINATION_VERDICT_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-04：Complete Verifier Query v1

- concrete Task IR executor 新增可选 autograd-preserving path；新增 deterministic center-start
  projected-gradient candidate search，search failure 明确不是 proof，found candidate 必须 concrete replay；
- 新增 multi-clause complete query：conjunction aggregation、ascending clause order、verified 全闭合、
  unsafe short-circuit、unresolved/pending unknown 与 cooperative deadline；
- query/clause trace 绑定 objective、threshold、search/optimizer/queue policy 与 search/queue/verdict
  hashes；claim、status、pipeline 与 numeric tamper 均 fail closed；
- 修复 optimized native re-execution trace 的 scale-aware 容差不一致：execution 继续使用
  `allclose(atol=2e-6, rtol=2e-6)`，serialized trace 使用 `2e-3` 独立 ceiling 并拒绝 non-finite；
- toy verified/unsafe/unknown/deadline 全闭环；固定 ResNet 九个真实 clauses 全部执行，但 native
  scalarized lower bounds 过松，9/9 unresolved，整体为 sound unknown；
- artifact generate/replay hash=
  `d17f7d7e960491ad7ef3f33bad41a4cfbf21a9fd5213df3637584b6a753968f1`；相关
  `39 passed`、全量 `670 passed, 37 skipped`；Black/Mypy/Pylint 全过；
- 只关闭 complete-query correctness/control `VALIDATED-REDUCED`，无 performance claim。下一阶段
  先冻结 end-to-end phase/tightness baseline，再按证据推进 bound strength 与执行优化。

**记录**
- `gemini_doc/BOUNDFLOW_COMPLETE_VERIFIER_QUERY_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-04：End-to-End Tightness and Performance Baseline v1

- typed external intermediate semantics 接入 optimizer Schedule、selected-native compiler、
  queue child batches 与 complete query；external bounds/provenance 与 tensor schema fail closed；
- adaptive α 初始化与 external initial-CROWN policy 对齐，默认 constant canonical hash 保持不变；
- fixed ResNet 从 local 0/9 提升为 external-adaptive 6/9 verified，仅 clauses 0/2/4 unknown；
  九个 lower 对 external initial 无退化、最大改善 `0.0072252750`、sign 9/9；
- 三组轮换 CPU audit queue median 为 `6.7178/6.7969/6.7317 s`，candidate/verdict 约
  `3.6/3.9 ms`，定位 compile/hash/selected-native re-execution 为主耗时；
- 新增 generate/replay runner、frozen artifact 与 semantic/timing/claim tamper tests；fresh replay
  hash=`14c3b9dc2e5376156be1f33f3e8804ec21f60e11096bd3bdc95225b7e1474376`；
- focused `35 passed`、全量 `684 passed, 37 skipped`；Mypy clean、Pylint 10.00/10、
  Black/diff check 通过；
- 只关闭单 workload CPU diagnosis `VALIDATED-REDUCED`；下一门禁是 prepared production fast
  path，随后处理三个 hard clauses，不声明 CUDA/竞品 speedup 或完整 verifier closure。

**记录**
- `gemini_doc/BOUNDFLOW_END_TO_END_TIGHTNESS_PERFORMANCE_BASELINE_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-04：Prepared Production Fast Path v1

- 新增 exact prepared optimizer program 与 root conjunction capsule；cold phase 冻结/验证
  Plan/Task/Schedule/source compiler/scope/hash，warm phase 保持 Schedule action 数值执行；
- production 明确省略逐 action audit hash chain 与 selected-native validation re-execution，任何
  program/module/input/objective/intermediate source/scope drift fail closed；
- fixed ResNet 三组 audit raw=`58.713/59.078/59.587 s`，prepared warm raw=
  `111.166/110.262/110.950 ms`，median ratio=`532.47×`；
- cold prepare=`14.724 s`、first=`1.415 s`、合计=`16.139 s`；retained payload=
  `2,076,372 B`，不隐藏 cold/memory 成本；
- prepared/audit lower max diff=`1.90735e-6`、candidate/status exact，仍为 6/9 verified；
  artifact replay hash=`e14fcd62b322c0bc60d45c726cf94a7aa6cfb8d7aa3212662d08996db169b6b2`；
- focused `25 passed`、全量 `698 passed, 37 skipped`；Black/Mypy/Pylint/diff 通过；
- 只关闭单 workload CPU internal-overhead diagnosis；不是 competitor speedup 或完整 verifier。
  下一门禁为 clauses 0/2/4 branching/stronger-bound。

**记录**
- `gemini_doc/BOUNDFLOW_PREPARED_PRODUCTION_FAST_PATH_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-04：Hard-Clause Objective Branching v1 启动

- clause 0 widest 7-node/depth-2 probe 的最差叶 lower 约 `-0.5168`，相对 root
  `-0.5357` 改善有限；
- 48-candidate batched fixed-state probe 约 `14.7 ms`，objective-aware 选择 `31:17`，估计
  worst-child=`-0.4309`；实际 optimizer children=`-0.4210/-0.4239`；
- 冻结下一路线为 typed objective branching policy + first-class score Plan/Task/Schedule；校准
  probe 不作为论文 speedup/完整证明结论。
- 已实现并接入完整 branch score IR；同预算 clauses `0/2/4` objective-vs-widest worst-leaf
  改善=`0.120752/0.071564/0.057901`，但三者仍为 unknown；
- frozen artifact fresh replay hash=
  `1193bee8817e4acc9ec33f8ddadc00a671d0ac3c9411f14f62978eb5ab1a95bd`；focused
  `15 passed`、全量 `707 passed, 37 skipped`，静态门禁全过；
- 关闭等级为 branch IR/control + bounded-tree tightness `VALIDATED-REDUCED`，不声明完整
  verifier、GPU、竞品或单次 audit timing 性能。

**记录**
- `gemini_doc/BOUNDFLOW_HARD_CLAUSE_OBJECTIVE_BRANCHING_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-04：Multiworkload Competitor E2E Baseline v1

- 新增 fail-closed VNNLIB box/property frontend 与 Query IR；三份真实 VNN-COMP property 的
  lower/upper/C/rhs 与固定 αβ-CROWN parser parity；
- 新增三 workload Plan/Task/Schedule：MNISTFC、CIFAR ResNet2B、OVAL21，21 tasks、6 个
  fresh-process native/competitor execution action；
- 修复 flatten/reshape-first BoxPerturbation bounds/shape trace 未随 shape op 变换的问题；
- 新增双后端 generate/replay runner与 448 KiB 完整 worker logs；artifact evidence hash=
  `473b287bb88e4c52426b405aeb4164aa72a98d7b1bbd74c00471fe1d1451deb0`；
- BoundFlow 状态为 `unknown/unknown/unknown`，αβ-CROWN 为 `verified/unknown/verified`；
  CPU 单次 E2E 只作诊断，`performance_claimed=false`；
- focused `16 passed`、全量 `723 passed, 37 skipped`；Black、targeted Mypy、Pylint 10.00/10、
  source-to-IR fresh replay 与 diff check 通过；
- 关闭 ingest/control/workload coverage `VALIDATED-REDUCED`；下一门禁是 native
  intermediate-bound refinement，不声明 GPU/speedup/ASPLOS-ready。

**记录**
- `gemini_doc/BOUNDFLOW_MULTIWORKLOAD_COMPETITOR_E2E_BASELINE_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-04：Native Intermediate-Bound Refinement v1

- plain CROWN 支持任意已产生中间值的 selected-row bound；新增 refinement
  Plan/Task/Schedule、top-width target、分块 backward、单调 intersection/propagation 与 trace；
- 新增独立 `native_refined` provenance，接入 optimizer、Bound IR 和 BaB child batches，禁止
  冒充 external verifier bounds；
- 正式 same-policy 结果：MNISTFC unresolved `3→1`、nodes `31→21`；OVAL21
  `unknown→verified`、nodes `15→11`；ResNet 两 root lower 改善 `+70.496/+160.551` 但仍 unknown；
- artifact fresh replay hash=
  `f6e6996608abacefb929ee88b05b45b3a16043cfca10f7a5d393e83bcd8bf14b`；focused
  `9 passed`、全量 `732 passed, 37 skipped`；Black/Mypy/Pylint 10.00/10 全过；
- 以 native refinement IR/control 与 multiworkload tightness `VALIDATED-REDUCED` 关闭；
  `performance_claimed=false`、ASPLOS-ready=NO。当时的下一门禁 objective-directed target
  selection 已由下节 NRIR-20 完成。

**记录**
- `gemini_doc/BOUNDFLOW_NATIVE_INTERMEDIATE_BOUND_REFINEMENT_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-04：Objective-Directed Intermediate Refinement v1

- 新增 single-clause `objective_influence_width_per_relu_v1`：从 CROWN backward state 捕获
  逐 ReLU influence，并以 influence×ambiguous width 选择固定预算 targets；
- objective hash、target influence/score 与 selection 输入进入 refinement Plan/Task/Schedule；
  多子句、schema、score、dependency 和 objective tamper 均 fail closed，旧 width hash 兼容；
- 旧 NRIR-19 runner 改为披露 code-revision mismatch 后执行 source-to-IR semantic replay，
  三 workload 的旧 width Plan/Task/Schedule 均精确恢复；
- fixed ResNet clauses 0/1 各以相同 96-target 预算对照，objective 相对 width root lower 改善
  `+55.928741/+26.228943`，但最终仍负，不声称 property closure；
- artifact fresh semantic replay hash=
  `8fce1c7c3e5c63adb14a7ab5b9f23407e4a7a1406353750e4f150ee745b4e88e`；focused
  `16 passed`、全量 `739 passed, 37 skipped`；Black、targeted Mypy、Pylint 10.00/10 通过；
- 关闭 objective-directed IR/control + fixed-root tightness `VALIDATED-REDUCED`；下一门禁为
  per-child exact-state refinement，`performance_claimed=false`、ASPLOS-ready=NO。

**记录**
- `gemini_doc/BOUNDFLOW_OBJECTIVE_DIRECTED_INTERMEDIATE_REFINEMENT_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-04：Per-Child Objective Refinement v1

- 每个 optimized BaB node 依据 exact split state 独立执行 objective-directed refinement
  Plan/Task/Schedule，再把 child-specific intermediate bounds 拼为 optimizer batch；
- queue/evaluation trace 绑定 split、三层 refinement IR、semantic trace、initial/final bounds 与
  target count；parent alpha/beta 只作 warm initialization，旧默认 payload 条件兼容；
- fixed ResNet clauses 0/1、同 96-target/5-step、7-node/depth-2 预算下，root lower 完全一致，
  但 per-child worst leaf lower 相对 root-global 退化 `-0.847961/-0.936646`；
- 按预设门禁以 `VALIDATED-NO-GO` 关闭，不声明 property closure、CUDA、competitor speedup、
  repeated performance 或 ASPLOS-ready；下一路线为 ancestral-constraint carry-forward refinement。

**记录**
- `gemini_doc/BOUNDFLOW_PER_CHILD_OBJECTIVE_REFINEMENT_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-04：Ancestral-Constraint Refinement v1

- child refinement Plan/Task/Schedule 显式绑定并消费 validated parent refinement execution 的
  final bounds、Plan 与 semantic trace；不接受裸 mapping，source 只作 sound constraint；
- local exact-split forward→constrained initial→final refinement 双重单调，queue parent lineage、
  Task dependency、source hash 与 tamper 全 fail closed；
- fixed ResNet clauses 0/1、同 96-target/5-step、7-node/depth-2 预算下，ancestral worst leaf 相对
  independent 提升 `+73.615173/+75.022095`，相对 root-global 提升
  `+72.767212/+74.085449`，root exact；
- 以 fixed bounded-tree tightness `VALIDATED-REDUCED` 关闭；叶 lower 仍负，不声明 complete
  property、CUDA、competitor speedup、repeated performance 或 ASPLOS-ready；下一路线为
  hard-clause convergence expansion。

**记录**
- `gemini_doc/BOUNDFLOW_ANCESTRAL_CONSTRAINT_REFINEMENT_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-04：External-Seeded Ancestral Refinement v1

- 新增 external-owned typed constraint seed IR；绑定图/输入、external/effective constraints 与
  source artifact/model/property/objective-set，raw external 先与 local forward 求可行交集；
- refinement Plan/Task/Schedule/action trace 显式消费 seed；queue root seed 与 child validated
  parent execution 严格互斥，逐节点 provenance/Plan/semantic/final hash tamper fail closed；
- fixed ResNet clauses `0/2/4` 的 seeded ancestral worst leaf 相对 external baseline 改善
  `+0.001512/+0.001133/+0.000534`，相对 seeded root-global 为
  `+0.000823/+0.000004/0`；三条仍 unknown；
- artifact semantic replay hash=
  `9f52b99a74dab448626061f5b8f060f3b8c43b6c03f6deb0899d9fe91883d9f7`；focused
  `33 passed`、全量 `766 passed, 37 skipped`，Black/Mypy/Pylint 10.00/10 通过；
- 以 typed seed/control + fixed-tree tightness `VALIDATED-REDUCED` 关闭；不声明 complete
  property、CUDA、performance、multi-workload 或 ASPLOS-ready。下一门禁为 depth/node convergence。

**记录**
- `gemini_doc/BOUNDFLOW_EXTERNAL_SEEDED_ANCESTRAL_REFINEMENT_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-04：External-Seeded Depth/Node Convergence v1

- 固定 ResNet clauses `0/2/4`，只把 external-seeded ancestral 完整树从 7/depth2 扩为
  15/depth3 与 31/depth4；每单元 fresh process、原子 checkpoint、严格 resume；
- 九个 shard 冻结 source/seed/policy、queue、refinement lineage 与 objective-branch IR hash；
  aggregate 按 split-state logical domains 校验嵌套，避免把 best-first execution order 冒充语义；
- 三条 worst terminal lower 均持续改善到 `-0.282360/-0.401845/-0.459939`，但仍无 fixed-tree
  closure；关闭为 convergence `VALIDATED-REDUCED`，不升级 property/performance/ASPLOS-ready；
- fresh semantic replay hash=
  `db0401bef0d938773fed04a173e49cae0ad0b4fdc4ffdd49450cc86fae7f0db6`；下一门禁为 dynamic
  ancestral refinement budget/multi-pass。

**记录**
- `gemini_doc/BOUNDFLOW_EXTERNAL_SEEDED_DEPTH_NODE_CONVERGENCE_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-04：Dynamic Ancestral Refinement Budget v1

- 新增 first-class parent-lower generated-group budget policy/decision，冻结 16/24/8 cap、tie、
  group/node/split/parent identity 与 exact conservation；assigned cap 进入实际 refinement
  Plan/Task/Schedule/execution，不是旁路调参日志；
- fixed16 与 dynamic8_24 在相同 31-node/depth-4、planned cap=`496` 下执行三 hard clauses；dynamic
  worst lower 分别改善 `+0.0003859997/+0.0002329946/+0.0002717972`；
- 六分片 artifact 固定 source、budget/decision、逐 node refinement IR、queue/branch/lower，支持 atomic
  checkpoint、strict resume、aggregate 和 fresh-process semantic replay；
- fresh replay 6/6、focused `34 passed`、全量 `778 passed, 37 skipped`、Black/Mypy/Pylint
  `10.00/10` 通过；
- 以 same-planned-cap dynamic tightness `VALIDATED-REDUCED` 关闭；所有树仍 unknown，不升级
  complete property/performance/ASPLOS-ready。下一门禁为 typed multi-pass refinement/termination。

**记录**
- `gemini_doc/BOUNDFLOW_DYNAMIC_ANCESTRAL_REFINEMENT_BUDGET_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-04：Typed Multi-Pass Refinement v1

- 新增 total-cap partition/reselection/termination policy 与逐 pass decision；每 pass 的
  enumerate/select/stop/backward/intersect/propagate 显式进入 Plan/Task/Schedule 与 action trace；
- dynamic 8/16/24 per-node cap 拆为 4+4/8+8/12+12，prior-target ledger 保证 disjoint selection，
  no-unseen target 走 sound passthrough；legacy lowering/hash 条件兼容；
- fixed ResNet clauses `0/2/4` 上，single 与 split-two-pass 的 worst lower delta 全为 `0.0`；总
  planned cap=`496`、actual targets=`2976`、logical tree=`31/31` 均相同；
- fresh replay 6/6、focused `50 passed`、全量 `787 passed, 37 skipped`、Black/Mypy/Pylint
  `10.00/10` 通过；
- 按预注册门禁以 `VALIDATED-NO-GO` 关闭；typed mechanism 保留，不升级 tightness/property/
  performance/ASPLOS-ready；停止 static-influence same-total-cap 拆 pass 路线。

**记录**
- `gemini_doc/BOUNDFLOW_TYPED_MULTIPASS_REFINEMENT_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-04：Production Prepared Verifier v1

- 新增 production verifier Plan/Task/Schedule、production ReLU-split queue 与 complete-query 集成；
  每个动态 batch 执行 validate/optimizer/materialize/commit，跳过 audit tensor hash chain 和
  selected-native 双执行；旧 audit payload/hash/default behavior 条件兼容；
- MNISTFC、ResNet2B、OVAL21 各三组交替 fresh-process clause-0 audit/production 对照，semantic
  parity 全过，median internal speedup=`1.3663×/2.4723×/1.4511×`；
- full production median=`14.834/60.754/11.964 s`，三类 query 仍 unknown；竞品历史单次只作
  diagnostic，不形成 speedup claim；
- fresh replay、focused `19 passed`、全量 `800 passed, 37 skipped`、Black/Mypy/Pylint 与 diff
  gate 通过；evidence hash=
  `7b650dce529d47c54eeadb168b2311e83a4346b47ffc341d5293b6468c6ac08b`；
- 以 production runtime/internal CPU overhead `VALIDATED-REDUCED` 关闭。下一门禁为 parametric
  dynamic-batch PlanTemplate/PlanInstance 与 compile cache；GPU、complete property、公平
  competitor 和 ASPLOS-ready 仍 pending。

**记录**
- `gemini_doc/BOUNDFLOW_PRODUCTION_PREPARED_VERIFIER_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-04：Parametric Dynamic Batch Compiler v1

- 新增 optimizer PlanTemplate/PlanInstance、reusable Task/Schedule、query-scoped exact cache 和
  additive parametric production queue/query；动态 tensor content 全部绑定 instance，contract 或
  runtime tensor 漂移 fail closed；
- 三 workload 各三组交替 fresh-process full-query production-v1→v2 median：MNISTFC
  `14.807→3.456 s`（`4.2849×`）、ResNet2B `61.239→6.209 s`（`9.8630×`）、OVAL21
  `13.021→3.718 s`（`3.5024×`）；语义逐 clause/queue/state 对齐；
- 每 query 只编译一个 template，instances/miss/hit=`19/1/18`、`27/1/26`、`11/1/10`；
- artifact fresh replay、NRIR-27 historical replay、focused `22 passed`、全量
  `818 passed, 37 skipped`、Black/Mypy/Pylint 通过；evidence hash=
  `117fcecf8e089c16f4275abb97292039790bae75bc4b518ae699bc9ac432ce97`；
- 以 internal full-query CPU performance `VALIDATED-REDUCED` 关闭；property 仍 unknown，不升级
  CUDA、竞品 speedup、complete-property 或 ASPLOS-ready。下一门禁为 fixed-wall-clock typed BaB
  depth/node scaling。

**记录**
- `gemini_doc/BOUNDFLOW_PARAMETRIC_DYNAMIC_BATCH_COMPILER_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-04：Wall-Clock Parametric BaB Scaling v1

- 新增 first-class search budget/Plan/Task/Schedule，将 `7/2、31/4、127/6 × 3 workload × 3
  repeats` 编译为 27 个 source/budget/order hash 绑定的 fresh-process tasks；
- artifact 导出逐 clause logical domains、leaf verdict、compiler template/cache/instance 与 raw
  timing；fresh replay 重建 experiment IR 并重算 repeat、nesting 与 closure 门禁；
- 27/27 均完成 9 clauses，公共 domain lower 漂移 0；MNISTFC verified `6/9→8/9`，ResNet
  `0/9→0/9`，OVAL21 `8/9→8/9`；
- n127d6 median execution=`2.515/58.566/2.287 s`；不计算跨预算 speedup。evidence hash=
  `e01d35c0afa8501f3d02ffaaa4eeaf609c444ed497c1a2d2efff4e97b3520214`；
- 以 search-coverage `VALIDATED-REDUCED` 关闭；下一门禁是 unresolved-clause typed hard-clause
  escalation/stronger-bound integration，ASPLOS-ready 仍为 NO。

**记录**
- `gemini_doc/BOUNDFLOW_WALL_CLOCK_PARAMETRIC_BAB_SCALING_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-04：Typed Hard-Clause Escalation v1

- 新增 exact-query escalation Plan、baseline-derived Decision、8-task guarded TaskModule/Schedule，
  以及 baseline→shared native refinement→hard ordinal projection→parametric query→aggregate runtime；
- whole deadline 固定 60 秒，admission 只能是 baseline unresolved；verified/unsafe 不重跑，超时或
  child proof 篡改只能保留 baseline verdict；
- 三 workload × 三 fresh repeats 全部无 fallback：MNIST `6/9→8/9`，ResNet `0/9→0/9`，OVAL
  `8/9 unknown→9/9 verified`；
- artifact source-to-program、NRIR-29 baseline、refinement、compiler/cache/instance 与 aggregate fresh
  replay；evidence hash=`df096e70d6126d585132e14dc9796038855b37bf4d9ef76528b9feb6a1330205`；
- 以 property-coverage `VALIDATED-REDUCED` 关闭；下一门禁为相同预算/deadline 下的 per-clause
  objective-directed refinement，ASPLOS-ready 仍为 NO。

**记录**
- `gemini_doc/BOUNDFLOW_TYPED_HARD_CLAUSE_ESCALATION_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-04：Objective-Directed Hard-Clause Escalation v1

- 新增 additive objective escalation Plan/33-task TaskModule/Schedule；每个 original clause 静态拥有
  guarded compile/refine/query，并绑定 exact shared-source lineage、objective hash 与 ordinal；
- runtime 在一个 60 秒 deadline 内执行 baseline→admit→shared→per-clause objective refinement→
  scalar query→aggregate；deadline 后 proof 丢弃，NRIR-30 final coverage 不回退；
- 单次 pilot 通过后执行三 workload × 三 fresh repeats。MNIST/ResNet/OVAL final verified 保持
  8/9、0/9、9/9；ResNet 九条 root lower 三轮逐值一致改善 `+81.522583—+179.970459`；
- artifact fresh replay、focused `8 passed`、全量 `838 passed, 37 skipped`、Black/Mypy/Pylint
  `10.00/10` 与 diff gate 通过；evidence hash=
  `fb9e503bdf93cb9ce56f52915f1965f1f542e092945d4d7d77d8b8c4bd91764a`；
- 以 objective-root tightness `VALIDATED-REDUCED` 关闭，不升级 property/performance/ASPLOS-ready。
  下一门禁为 objective-ancestral dynamic-child propagation。

**记录**
- `gemini_doc/BOUNDFLOW_OBJECTIVE_DIRECTED_HARD_CLAUSE_ESCALATION_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-04：Objective-Ancestral Hard-Clause Escalation v1

- 新增 static Plan、committed dynamic Task IR、1:1 Schedule 与 native queue；typed root admission 和
  每个 child 的 parent final-bound/Plan/semantic lineage 均 fail closed，emit 依赖完整 committed proof；
- feasibility two-child gate 改善 `+59.367462/+59.253479` 后才进入正式实现；
- 固定 ResNet property 0 clause 0、31/depth4/60 s 三 fresh repeats：ancestral 均提交 7 nodes、24
  tasks，worst active lower `-104.7654114`，相对 31-node root-global `-200.4653931` 改善
  `+95.6999817`；无 property closure；
- artifact fresh replay、focused `8 passed`、全量 `846 passed, 37 skipped`、Black/mypy/Pylint/diff
  gate 通过；evidence hash=
  `8fba8deca18dcbf0b4b258aa390c1dd48d250c71ea1a48ddb991388765411bfc`；
- 以 typed lineage + committed-frontier tightness `VALIDATED-REDUCED` 关闭；下一门禁为 fixed-deadline
  child refinement cap/resource Pareto，ASPLOS-ready 仍为 NO。

**记录**
- `gemini_doc/BOUNDFLOW_OBJECTIVE_ANCESTRAL_HARD_CLAUSE_ESCALATION_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-04：Objective-Ancestral Child Budget Pareto v1

- 新增 five-cap Policy/Calibration/Decision/Plan IR 与 thin runtime wrapper；selected cap、90%
  retention rule 和 calibration evidence 全部 hash-bound，NRIR-32 frozen engine/artifact 未修改；
- ResNet clause 0 caps `8/16/32/64/128` fresh-process pilot 全部仅提交 7 nodes；worst lower 从
  `-173.078613` 改善到 `-104.765411`，预注册规则只能选择 cap128；
- pilot replay、5 个 focused tests、全量 `851 passed, 37 skipped` 与静态门禁通过；pilot hash=
  `db9b406eebebad0c1c4d6f39e8088667935f10e3d54f38cb848dce792dd757eb`；
- cap-only coverage 以 `VALIDATED-NO-GO` 关闭；下一门禁为 sibling packed refinement/evaluation +
  parametric evaluator，ASPLOS-ready 仍为 NO。

**记录**
- `gemini_doc/BOUNDFLOW_OBJECTIVE_ANCESTRAL_CHILD_BUDGET_PARETO_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-04：Sibling-Packed Objective-Ancestral Evaluator v1

- 新增 source/evaluator objective projection、same-parent SiblingGroup、child lineage、packed
  optimizer/native 与 atomic commit 的 Plan/Task/Schedule/Group IR；
- first-pair profiler 中 optimizer/native groups 均 `2→1`，child elapsed
  `13.291550→7.018038 s`，bounds exact；
- 31/depth4/60 s 三 fresh repeats 的 serial accepted nodes=`[7,7,7]`，packed=`[15,15,15]`，
  common lower/upper max diff=`7.6293945e-06`，formal hash=
  `9678f9624abd547b76326ad2a1b916c3944d14fc96b2fbe0e81cf61849a777b4`；
- 9-clause global-60s integration sound `unknown`：completed/unresolved=`[0]`，pending=`[1..8]`，
  evidence hash=`dcd0dc89fa7e4eb503e8a8b29438e16d215da10e66cd045cc76eb19a30037bf5`；
- 以 single-hard-clause same-algorithm deadline coverage `VALIDATED-REDUCED` 关闭；无 property/GPU/
  competitor/ASPLOS-ready 升级。下一门禁为 cross-clause shared evaluator + anytime global budget。

**记录**
- `gemini_doc/BOUNDFLOW_SIBLING_PACKED_OBJECTIVE_ANCESTRAL_EVALUATOR_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-04：Cross-Clause Anytime Objective Evaluator v1

- 新增 NRIR-31 floor + guarded NRIR-34 packed escalation 的 Plan/Decision/6-stage Task/Schedule 与
  native runtime；exact clause-0 source lineage、original ordinal、single global start 和 monotone
  aggregate 均 fail closed；
- 三 fresh repeats 的 floor elapsed=`[22.227251,21.622773,21.834220] s`，均完成 9/9 original
  ordinals；剩余预算内 packed accepted nodes=`[7,7,9]`，但 final 仍 9/9 unresolved；
- formal replay、六类同步重哈希 tamper、关联 `29 passed`、全量 `874 passed, 37 skipped` 与静态门禁
  通过；formal hash=`74533c9c211a3007bf5af43c08865febd95c3f9ccf1a268e56738793ec9d14d5`；
- 以 cross-clause control/original-ordinal preservation `VALIDATED-REDUCED` 关闭；whole cooperative
  elapsed 约 `62.09—67.72 s`，不声明硬实时、performance/property/GPU/competitor/ASPLOS-ready。
  下一门禁为 multi-clause anytime priority/time slicing。

**记录**
- `gemini_doc/BOUNDFLOW_CROSS_CLAUSE_ANYTIME_OBJECTIVE_EVALUATOR_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-05：Multi-Clause Anytime Priority v1

- 新增 root-lower priority、top-2 selection、dynamic equal-remaining slice 与 one-shot cutoff 的
  Policy/Plan/Candidate/Decision/8-task Task/Schedule/Slice/Outcome/Aggregate IR 和 native runtime；
- 三 fresh repeats 都复现 priority=`[2,3,4,5,0,8,6,7,1]`、selected=`[2,3]`；packed nodes=
  `[[3,3],[3,3],[3,1]]`，repeat 2 第二条未提交 atomic pair，final 仍 9/9 unresolved；
- formal replay、九类同步重哈希 tamper、NRIR-31/34/35 predecessor replay、16 focused tests 与
  全量 `890 passed, 37 skipped`、Black/mypy/Pylint `10.00/10` 通过；formal hash=
  `2a2081af4c38de3df7a23c62cfcecfeb74d4b15132390a069e04a28bb65bfbf0`；
- 预注册 multi-clause coverage gate 失败，以 `VALIDATED-NO-GO` 关闭，`performance_claimed=false`；
  IR/control 可保留，下一门禁为 shared parametric compiler/root/evaluator + stronger bound/candidate，
  ASPLOS-ready 仍为 NO。

**记录**
- `gemini_doc/BOUNDFLOW_MULTI_CLAUSE_ANYTIME_PRIORITY_V1_CHANGELOG_2026_08_04.md`

---

## 2026-08-05：Shared Parametric Objective Evaluator v1

- 新增 shared-parametric ancestral Plan/Batch/Task/Schedule 与 production queue；template/instance/cache
  边界显式化，root/完整 sibling pair 原子提交，生产路径不再做 selected-native audit re-execution；
- 新增 NRIR-36 control × NRIR-37 evaluator 的 multi-clause runtime，一个 cache owner 跨 batch/跨
  clauses 2/3 共享同一 template，仍保持 frozen floor、rank、slice、cap、node/depth 与 deadline；
- 真实 clause-2 root+pair parity 通过；单轮 pilot clauses 2/3 均 31 nodes，随后三 fresh processes
  继续稳定为 `[[31,31],[31,31],[31,31]]`，每轮 cache miss=1，whole=
  `[51.996191,52.251681,52.695640] s`；
- pilot/formal replay、11 类 artifact tamper、Task/Batch commit binding tamper、27 focused tests、全量
  `917 passed, 37 skipped`、mypy
  clean、Pylint `10.00/10` 通过；pilot/formal hash=
  `c96fff3fa2bc2563b4d46886d69b33f51ac985b19ad80d916309db57fe6cfefa` /
  `9234dcbe77803e0e7d7e62ca88c62e1b859c95af4ad8e3a19b85c0ab87294b83`；
- 以 shared compiler ownership + fixed-deadline coverage `VALIDATED-REDUCED` 关闭；final 仍 9/9
  unresolved，`performance_claimed=false`，下一门禁为 depth-4 frontier tightness attribution 与单变量
  stronger-bound/candidate。

**记录**
- `gemini_doc/BOUNDFLOW_SHARED_PARAMETRIC_OBJECTIVE_EVALUATOR_V1_CHANGELOG_2026_08_05.md`

---

## 2026-08-05：Full Frontier Tightness Attribution v1

- 新增 exact-frontier attribution Plan/七阶段 Task/Schedule 与 runtime，完整覆盖 selected clauses 2/3
  的 31-node source、16-node depth-4 active frontier、path/refinement/alpha-beta state；
- 按预注册只比较 optimizer `steps=5→15`，保持 split、ancestral refinement、parent warm、sibling batch、
  dtype/device 不变；baseline replay lower/upper max diff=0、refinement hashes exact；
- steps15 改善 32/32 nodes，但 clauses 2/3 worst-active lower 仅改善 `+0.055496/+0.028557`，未过
  `+1.0` gate，以 `VALIDATED-NO-GO` 冻结 optimizer-step 轴；
- replay、8 类同步 tamper、13 focused tests、全量 `930 passed, 37 skipped`、mypy clean、Pylint
  `10.00/10` 通过；pilot hash=
  `2719347a8e1c5c49c418b3a396ff405a004b0f4ace96af94d335e4026f7a24a2`；
- 下一单变量为把已有 objective-bound-impact branch IR 接入 shared ancestral evaluator；
  `performance_claimed=false`，ASPLOS-ready 仍为 NO。

**记录**
- `gemini_doc/BOUNDFLOW_FULL_FRONTIER_TIGHTNESS_ATTRIBUTION_V1_CHANGELOG_2026_08_05.md`

---

## 2026-08-05：Objective Branch Shared Evaluator v1

- 新增 composite Plan/6-task TaskModule/Schedule 与 objective-aware shared queue；每个仍有候选的 node
  evaluation 都绑定 exact objective branch program/score/selection，底层 NRIR-37 shared runtime 保持不动；
- 修复大尺度 float32 candidate width、child mean 与 materialized branch 的跨表示 1e-6 绝对误差假拒绝，
  统一为 `rel_tol=1e-6,abs_tol=1e-6`，同时保留 `+0.1` tamper fail closed；
- 真实 clauses 2/3 control/candidate 均 31 evaluations、16 depth-4 active nodes，root exact；worst-active
  improvement=`+2.043362/+5.641768`、median delta=`+2.537640/+5.885233`，两条通过 `+1.0` gate；
- artifact generate/replay 与 policy/coverage/selection/Task/claim/control tamper 通过；pilot hash=
  `dde1cc4076ea766e7b4859e75ec9ff214d61f3cf245385285274b47f541a72cc`；
- 16 focused、40 predecessor-inclusive tests、全量 `940 passed, 37 skipped`、mypy clean、Pylint
  `10.00/10` 通过；
- 以 fixed-budget branch selection `VALIDATED-REDUCED` 关闭，`performance_claimed=false`；下一门禁为
  three-repeat whole-query/global-deadline formal，ASPLOS-ready 仍为 NO。

**记录**
- `gemini_doc/BOUNDFLOW_OBJECTIVE_BRANCH_SHARED_EVALUATOR_V1_CHANGELOG_2026_08_05.md`

---

## 2026-08-05：Objective Branch Whole Query Formal v1

- 新增 raw objective-branch shared production queue 和 multi-clause anytime composition；objective
  scoring 被纳入 single-global-60s monotonic deadline，冻结其余 floor/rank/slice/refinement/cache/queue；
- 三 fresh processes 的 correctness、rank/selected、typed branch coverage、cache 与 nine-ordinal
  aggregate 全过；accepted nodes=`[[29,23],[29,21],[29,21]]`，branch counts exact；
- clauses 2/3 worst-active lower 为 `-48.315041` 与 `-43.299690/-44.731468`，既未达到 `31/15`
  coverage，也未达到相对 frozen widest `+1.0`；final 仍 9/9 unresolved；
- whole cooperative elapsed=`[63.357098,63.161128,62.485366] s`；artifact replay 与 formal+shard+
  manifest 同步重哈希 branch-coverage tamper fail closed；formal hash=
  `d69b56d4d82ad5bf8d30883258c15a39e5a45f1fac9dbc8eb35e91fda9f6a492`；
- focused `8 passed`、predecessor-inclusive `55 passed`、全量 `944 passed, 37 skipped`，静态门禁通过；
- 以 objective-branch global-budget `VALIDATED-NO-GO` 关闭，`performance_claimed=false`；NRIR-39
  fixed-budget 机制结论保留，但不得升级为 production 或 ASPLOS-ready claim。

**记录**
- `gemini_doc/BOUNDFLOW_OBJECTIVE_BRANCH_WHOLE_QUERY_FORMAL_V1_CHANGELOG_2026_08_05.md`

---

## 2026-08-05：Objective Branch Production Cost Attribution v1

- 新增 attribution Plan/6-task TaskModule/Schedule、prefix/wall/profile/Decision IR，不修改 NRIR-39/40
  frozen runtime 与 artifact；
- 从 frozen 31-node rows 重建 clauses 2/3 的 `21/23/29/31` active frontier，objective 相对 widest
  worst-active improvement 全部为正，frontier-order gate 成立；
- 三 fresh counterbalanced paired runs 的 objective/widest queue median ratio=
  `1.748660/1.750639`，cProfile branch-program share=`21.9371%/21.9139%`；31 次 branch program
  实际触发 341 次 candidate enumeration；
- formal replay、median/MAD 重算与同步重哈希 prefix tamper fail closed；formal hash=
  `fe67b77197905a8a4d7f92ad5eac686892243dfb0e7d7b7c7434861aaa794834`；
- focused `4 passed`、predecessor-inclusive `12 passed`、全量 `948 passed, 37 skipped`，静态门禁通过；
- 以 internal causal attribution `VALIDATED-REDUCED` 关闭，`performance_claimed=false`；下一单变量
  限定为 scorer ownership/validation reuse，不撤销 NRIR-40 production NO-GO。

**记录**
- `gemini_doc/BOUNDFLOW_OBJECTIVE_BRANCH_PRODUCTION_COST_ATTRIBUTION_V1_CHANGELOG_2026_08_05.md`

---

## 2026-08-05：Objective Branch Scorer Ownership v1

- 新增 typed validated capsule、Plan-owned candidate scorer Task/Schedule、prevalidated executor、additive
  production queue 与 global multi-clause composition；NRIR-39/40 frozen 文件保持不变；
- Phase A 六组 exact parity；31-node enumeration `341→31`，clauses 2/3 new/old median ratio=
  `0.706888/0.698486`，formal hash=`0d310c2f…25b58`；
- Phase B 三 fresh global-60s queries 均 selected `[2,3]`、accepted `[31,31]`，whole=
  `57.175184/57.697757/58.114412 s`，formal hash=`7274e834…7d759`；
- typed replay 与 synchronized capsule/score/call/deadline tamper 通过；targeted `10 passed`，全量
  `958 passed, 37 skipped`，Black/mypy/Pylint 通过；
- 以固定 ResNet2B property 0 CPU production admission `VALIDATED-REDUCED` 关闭；final unknown、
  `performance_claimed=false`、ASPLOS-ready=NO。

**记录**
- `gemini_doc/change_2026-08-05_objective_branch_scorer_ownership.md`
- `gemini_doc/BOUNDFLOW_OBJECTIVE_BRANCH_SCORER_OWNERSHIP_V1_CHANGELOG_2026_08_05.md`

---

## 2026-08-05：NRIR-42 发布

- 功能提交 `264365f` 已由 PR #53 合入 `main@8969064`；
- 当前状态、执行备忘与文档索引均已切换到该 integration base；
- 下一工程轴固定为 NRIR-43 cross-axis verification batch Schedule，尚未形成性能 claim。

**记录**
- `gemini_doc/change_2026-08-05_nrir42_publication.md`

---

## 2026-08-05：NRIR-43 Cross-Axis Verification Batch Schedule v1 预注册

- 从 `main@34ca6c6` 创建 `feat/cross-axis-verification-batch-schedule-v1`；
- 唯一变量冻结为 ready clause/node/candidate work 的 typed ragged batch Schedule；
- Phase A/B 的 exact semantics、launch count、paired timing、global deadline 与 NO-GO 条件已预注册；
- 当前尚无代码、artifact 或性能 claim。

**记录**
- `gemini_doc/change_2026-08-05_nrir43_preregistration.md`
- `gemini_doc/BOUNDFLOW_CROSS_AXIS_VERIFICATION_BATCH_SCHEDULE_V1_PLAN_2026_08_05.md`
- `gemini_doc/BOUNDFLOW_CROSS_AXIS_VERIFICATION_BATCH_SCHEDULE_V1_CHANGELOG_2026_08_05.md`

---

## 2026-08-05：NRIR-43 Cross-Axis Verification Batch Schedule Phase A NO-GO

- 新增 typed ragged Plan/Instance/Task/Schedule/Trace、联合 scorer runtime、additive queue 与 formal replay；
- 6/6 exact，per-clause scorer launch `31→16`；
- clauses 2/3 median ratio=`1.051134/1.044573`，预注册 timing gate 失败，Phase B gated off；
- formal hash=`692b9e273661fce9f12129e134550547afa4023361e2a79d751c437c92f30390`；
- targeted `10 passed`、全量 `968 passed, 37 skipped`、静态门禁通过；
- 下一变量为 NRIR-44 Root-Projection Floor Schedule。

**记录**
- `gemini_doc/change_2026-08-05_nrir43_cross_axis_batch_nogo.md`

---

## 2026-08-05：NRIR-43 发布

- 提交 `00b82c2` 已由 PR #54 合入 `main@2d245d6`；
- 发布不改变 NRIR-42 production 默认，也不升级性能 claim；
- 下一预注册路线为 NRIR-44 Root-Projection Floor Schedule。

**记录**
- `gemini_doc/change_2026-08-05_nrir43_publication.md`

---

## 2026-08-05：NRIR-44 Root-Projection Floor Schedule v1 预注册

- 从 `main@d9d76da` 创建 `feat/root-projection-floor-schedule-v1`；
- 唯一变量冻结为 ranking floor objective child query 的 `n31d4→n1d0` consumer projection；
- baseline/refinement/root/rank/top-2/production 不改，sound-but-less-complete 边界显式；
- Phase A/B correctness、work reduction、timing 与 NO-GO 条件已冻结；当前无代码或正式 claim。

**记录**
- `gemini_doc/change_2026-08-05_nrir44_preregistration.md`
- `gemini_doc/BOUNDFLOW_ROOT_PROJECTION_FLOOR_SCHEDULE_V1_PLAN_2026_08_05.md`
- `gemini_doc/BOUNDFLOW_ROOT_PROJECTION_FLOOR_SCHEDULE_V1_CHANGELOG_2026_08_05.md`

---

## 2026-08-05：NRIR-44 Root-Projection Floor Schedule v1 关闭

- 新增 typed consumer/liveness Plan/Instance/Task/Schedule/Trace 与 additive projected floor/global
  composition，frozen NRIR-31/42/43 文件不改；
- Phase A 三轮 exact，objective evaluations=`279→9`，old/projected floor median ratio=`0.407530`，
  formal hash=`ecb553d8…ff0fe`；
- Phase B floor=`8.538814/8.622447/8.648849 s`，whole=
  `43.571040/44.144990/44.095736 s`，相对 NRIR-42 median ratio=`0.764254`；每轮 production
  nodes=`[31,31]`、worst lower exact；
- Phase B formal payload hash=`2f22d44f…7272d9`；replay 与同步外层重哈希 tamper fail closed；
- 以 fixed ResNet2B property 0 CPU8 `VALIDATED-REDUCED` 关闭，final unknown、
  `performance_claimed=false`、ASPLOS-ready=NO；全量 `979 passed, 37 skipped`。

**记录**
- `gemini_doc/change_2026-08-05_nrir44_root_projection_floor.md`
- `gemini_doc/BOUNDFLOW_ROOT_PROJECTION_FLOOR_SCHEDULE_V1_CHANGELOG_2026_08_05.md`

---

## 2026-08-05：NRIR-44 发布

- 功能/证据提交 `437680e` 已由 PR #55 合入 `main@f194034`；
- 发布不扩大 fixed ResNet2B property 0 CPU8 admission 边界，final unknown 与 ASPLOS-ready=NO 不变；
- 下一动作是先归因剩余 top-2 production queue 成本，再冻结 NRIR-45 单变量。

**记录**
- `gemini_doc/change_2026-08-05_nrir44_publication.md`

---

## 2026-08-05：NRIR-45 Prepared Intermediate Refinement Capsule v1 预注册

- 从 `main@b6eb697` 创建 `feat/prepared-intermediate-refinement-capsule-v1`；
- cProfile 定位单 queue 246 次 `_select_targets` 中 186 次来自重复 Program validation；
- 唯一变量冻结为 prepare-once refinement validation ownership，不改算法、policy、预算或 deadline；
- Phase A/B 的 exact semantics、ownership、fresh timing 与 NO-GO 条件已冻结；当前无正式 claim。

**记录**
- `gemini_doc/change_2026-08-05_nrir45_preregistration.md`
- `gemini_doc/BOUNDFLOW_PREPARED_INTERMEDIATE_REFINEMENT_CAPSULE_V1_PLAN_2026_08_05.md`
- `gemini_doc/BOUNDFLOW_PREPARED_INTERMEDIATE_REFINEMENT_CAPSULE_V1_CHANGELOG_2026_08_05.md`

---

## 2026-08-05：NRIR-45 Prepared Intermediate Refinement Capsule v1 关闭

- 新增 typed capsule/receipt、5-stage Task/Schedule/Trace、prepared Program/Execution、additive
  per-child/shared queue 与 projected-floor global composition；
- Phase A 六组 exact，target selection=`246→98`、full validation=`186→38`、full hash=`217→39`；
  clauses 2/3 median ratio=`0.727519/0.736603`，formal hash=`be1ccb42…05d439`；
- Phase B whole trace=`31.262521/31.319772/31.470078 s`，measured=
  `36.396631/36.513683/36.611709 s`，相对 NRIR-44 median ratio=`0.710268/0.615738`；
- 每轮 `[31,31]` nodes、60/60 capsule full replay；payload hash=`4ae71919…1a01f8`；
- 两阶段 replay/tamper、全量 `984 passed, 37 skipped`、Pylint `10.00/10` 通过；以 fixed
  ResNet2B property 0 CPU8 `VALIDATED-REDUCED` 关闭，final unknown、ASPLOS-ready=NO。

**记录**
- `gemini_doc/change_2026-08-05_nrir45_prepared_refinement.md`
- `gemini_doc/BOUNDFLOW_PREPARED_INTERMEDIATE_REFINEMENT_CAPSULE_V1_CHANGELOG_2026_08_05.md`

---

## 2026-08-05：NRIR-46 Template/Instance Phase 0 NO-GO

- 新增 three-fresh-process compiler ownership attribution runner/test/artifact；
- compile total median=`5.366369 s`，strict static topology median=`1.071197 s`，低于预注册
  `1.5 s` gate；ownership-convertible ceiling median=`2.102134 s`；
- 每轮 60 个 target ledger 全部互异，target selection observed/semantic=`124/60`；
- formal hash=`712ce359…cf846`，replay/tamper 通过；
- targeted `2 passed`、全量 `986 passed, 37 skipped`、静态门禁通过；
- NRIR46 以 `VALIDATED-NO-GO` 关闭，未实现 Template/Instance，Phase A/B gated off；下一候选路线为
  独立预注册的 single-pass exact target admission receipt。

**记录**
- `gemini_doc/change_2026-08-05_nrir46_phase0_nogo.md`
- `gemini_doc/BOUNDFLOW_INTERMEDIATE_REFINEMENT_TEMPLATE_INSTANCE_V1_CHANGELOG_2026_08_05.md`

---

## 2026-08-05：NRIR-46 发布与 NRIR-47 预注册

- NRIR46 功能/证据提交 `765be8f` 已由 PR #57 合入 `main@ca0bcf3`；
- 下一分支 `feat/single-pass-target-admission-receipt-v1` 只消除 production compile/validate 的
  target reselection，不共享 60 个动态 target ledger；
- explicit full replay 仍从 exact source 重选；
- Phase A compiler/queue ratio 门槛=`0.85/0.97`；Phase B trace/measured ratio=`0.98/0.98`；
- 当前无 NRIR47 实现、artifact 或新性能 claim。

**记录**
- `gemini_doc/change_2026-08-05_nrir46_publication.md`
- `gemini_doc/change_2026-08-05_nrir47_preregistration.md`
- `gemini_doc/BOUNDFLOW_SINGLE_PASS_TARGET_ADMISSION_RECEIPT_V1_PLAN_2026_08_05.md`
- `gemini_doc/BOUNDFLOW_SINGLE_PASS_TARGET_ADMISSION_RECEIPT_V1_CHANGELOG_2026_08_05.md`

---

## 2026-08-05：NRIR-47 Single-Pass Target Admission Receipt Phase A NO-GO

- 新增 typed receipt/Task/Schedule、additive single-pass compiler、prepared binding、candidate production
  route 与 explicit full replay；legacy compiler 文件保持冻结；
- 每条 candidate queue compile selector/reselection=`30/0`、runtime selector=`30`、receipt/full
  replay=`31/31`；correctness/ownership 与 186 receipt replay/tamper 门禁通过；
- compiler ratio=`0.936003 > 0.85`，clauses 2/3 queue ratio=
  `1.011205/1.019338 > 0.97`；Phase A timing 失败，Phase B gated off；
- formal hash=`a7561e51…042ce`；全量 `992 passed, 37 skipped`，Pylint `10.00/10`；
- 以 `VALIDATED-NO-GO` 关闭，candidate 不默认启用；下一门禁转 top-2 production execution
  math/queue phase attribution。

**记录**
- `gemini_doc/change_2026-08-05_nrir47_phase_a_nogo.md`
- `gemini_doc/BOUNDFLOW_SINGLE_PASS_TARGET_ADMISSION_RECEIPT_V1_CHANGELOG_2026_08_05.md`

---

## 2026-08-05：NRIR-47 发布与 NRIR-48 Execution Cost Attribution 预注册

- NRIR47 `351f5ce` 已由 PR #58 合入 `main@1e44949`；NO-GO、Phase B gated off 与 candidate disabled
  边界不变；
- 新分支 `feat/top2-production-execution-cost-attribution-v1` 只测 frozen NRIR45 default route；
- clauses 2/3 three-fresh paired control/profile、七个互斥类别、child-execute 子类、closure、
  perturbation、dominance 与稳定性门禁已冻结；
- 当前没有 NRIR48 runner、artifact、dominant category 或性能 claim。

**记录**
- `gemini_doc/change_2026-08-05_nrir47_publication.md`
- `gemini_doc/change_2026-08-05_nrir48_preregistration.md`
- `gemini_doc/BOUNDFLOW_TOP2_PRODUCTION_EXECUTION_COST_ATTRIBUTION_V1_PLAN_2026_08_05.md`
- `gemini_doc/BOUNDFLOW_TOP2_PRODUCTION_EXECUTION_COST_ATTRIBUTION_V1_CHANGELOG_2026_08_05.md`

---

## 2026-08-05：NRIR-48 Top-2 Production Execution Cost Attribution 关闭

- 新增 additive paired attribution runner 与七类顶层/五类 child-execute 互斥闭合；
- 6/6 semantics exact，profile/control ratio=`1.023199/1.020221`；
- child execute 两条 3/3 dominant，queue share=`32.1966%/31.1640%`；selected-CROWN 占 parent=
  `71.7725%/72.7291%`，按预注册规则成为唯一 NRIR-49 来源；
- formal hash=`571c2e47…d177a4`；replay/tamper、focused `4 passed`、全量
  `996 passed, 37 skipped`、静态门禁通过；
- 状态为 attribution `VALIDATED-REDUCED`，不是 speedup；下一路线只做 selected-CROWN execution。

**记录**
- `gemini_doc/change_2026-08-05_nrir48_execution_cost_attribution.md`
- `gemini_doc/BOUNDFLOW_TOP2_PRODUCTION_EXECUTION_COST_ATTRIBUTION_V1_CHANGELOG_2026_08_05.md`

---

## 2026-08-06：NRIR49 G0 GPU Opportunity Admission pre-reboot

- 新增 fail-closed G0 admission runner/test/artifact，GPU 不可用时禁止生成 memory/Amdahl/performance
  数值；
- 当前 CUDA/PyTorch/TVM/FFI 软件栈存在，根因收敛为 ASUS `dgpu_disable=1`；enable 已 queued，需重启；
- 独立 αβ-CROWN 官方锁定环境已建成，commit=`e5c7e17`、auto_LiRPA=`5a098e8`，import smoke PASS；
- VNN-COMP 2021 `mnistfc:2` 在双方 30 秒 CPU qualification 中均整题 `verified`，solveability 门禁通过；
- 正式 v7 artifact replay PASS，当前唯一 blocker=`gpu_infrastructure_ready`；`g1_ready=false`；
- targeted `21 passed`、全量 `1014 passed, 37 skipped`、mypy clean、Pylint `10.00/10`；
- 未修改 bound math/TIR/kernel/default policy，未产生性能 claim。
- 新增 post-reboot 六门 CUDA 功能 smoke/replay；当前 pre-reboot dry-run 六项均 blocked、exit `2`，
  不会把软件 build capability 误当作真实 GPU 可执行。

**记录**
- `gemini_doc/BOUNDFLOW_NRIR49_G0_GPU_OPPORTUNITY_ADMISSION_V1_PLAN_2026_08_06.md`
- `gemini_doc/BOUNDFLOW_NRIR49_G0_GPU_OPPORTUNITY_ADMISSION_V1_CHANGELOG_2026_08_06.md`

---

## 2026-08-06：NRIR49A G1 GPU Selected-CROWN Attribution NO-GO

- RTX 4060 Laptop上完成clauses 2/3五fresh workers、五chunk Latin sweep与paired default32 control；
- selected-CROWN queue/complete share中位=`7.0986%/7.0523%`，profile/control perturbation中位=
  `0.999304/1.006747`；
- 20%机会门槛失败，queue 1.20x与complete 1.15x目标均超过Amdahl无限区域加速上限；
- 最大allocated/reserved仅占物理显存`0.996%/1.353%`，合法batch上限1、无OOM，memory path=`N/A`；
- 60组结构exact，数值差异在预注册`2e-4`内；summary/manifest hash=`7eefe6a7…ab50`/
  `d0272fe4…c81f`，独立replay与digest重算通过；
- G1以`VALIDATED-NO-GO`关闭，selected-CROWN G2/G3 gated off；下一步重新归因GPU winner，
  `performance_claimed=false`。

**记录**
- `gemini_doc/BOUNDFLOW_NRIR49A_G1_GPU_ATTRIBUTION_V1_PLAN_2026_08_06.md`
- `gemini_doc/BOUNDFLOW_NRIR49A_G1_GPU_ATTRIBUTION_V1_CHANGELOG_2026_08_06.md`
- `gemini_doc/change_2026-08-06_nrir49a_g1_gpu_attribution_nogo.md`
