# 2026-07-09：BoundFlow 分支盘点与 Linux TVM 算子交接

## 摘要

当前远程仓库实际只有三条工作线：

- `origin/main`：较旧但稳定的主线，停在 DAG backward operator preservation。
- `origin/codex/phase7a-structured-crown-docs`：比 `main` 新，承载 Phase 7A PR-10 到 PR-14、shared CROWN 结构化路径和 TVM-FFI 安装加固。
- `origin/feat/macos-arm64-dev-env`：当前最新远程分支，在 phase7a 分支基础上增加 macOS arm64 开发环境 bootstrap。

当前本机分支是 `feat/macos-arm64-dev-env`，已经跟踪 `origin/feat/macos-arm64-dev-env`。但工作区还有一批未提交改动，主要是 Phase 7A PR-15 到 Phase 7B PR-28 的 benchmark、planner、MPS/Metal 探索和文档。这些内容目前不在任何远程分支提交里。

## 已提交分支现状

### `main` / `origin/main`

当前提交：

```text
ce36a51 feat: preserve operator path in dag backward
```

定位：

- 这是远程默认分支，也是 `origin/HEAD`。
- 包含早期 BoundFlow 主线：Task/Planner、TVM backend skeleton、Phase 5 bench/artifact pipeline、Phase 6/7A 早期 CROWN/alpha-CROWN/BaB/general DAG 工作。
- 相比后续 phase7a 分支，它缺少 PR-10 之后的 shared CROWN structured ReLU/layout-only 改造、PR-11 到 PR-14 benchmark 文档同步、以及 TVM-FFI install hardening。

用途建议：

- 不建议在这里直接继续 TVM 算子开发；它比当前实际研发前沿旧。
- 只适合作为默认主线参考或回看旧基线。

### `origin/codex/phase7a-structured-crown-docs`

当前提交：

```text
f3ef979 chore: harden TVM-FFI install and add CLI updater
```

相对 `origin/main` 新增提交：

```text
14be342 runtime: structure shared CROWN backward and layout ops
62cc086 docs: consolidate repo docs and sync post-PR10 state
be00067 docs: sync phase7a PR11-PR14 work docs
f436853 feat: benchmark shared CROWN relu pullback path
f3ef979 chore: harden TVM-FFI install and add CLI updater
```

定位：

- 这是 Phase 7A 的真实后继线。
- 主要完成：
  - ReLU backward structured path。
  - layout-only `reshape/permute/transpose` shared CROWN 支持。
  - PR-11 到 PR-14 的 shared CROWN benchmark 与 ReLU pullback 相关文档同步。
  - TVM-FFI 安装脚本加固，包括 Cython core module 和 `USE_GTEST=OFF` 等安装稳定性修复。

用途建议：

- 适合作为 Linux/CUDA 或 TVM 算子继续开发的干净 phase7a 基线。
- 如果不需要 macOS 环境 commit，可以从这个分支新开 Linux 工作分支。

### `feat/macos-arm64-dev-env` / `origin/feat/macos-arm64-dev-env`

当前提交：

```text
018fb8b feat: support macos arm64 dev bootstrap
```

相对 `origin/codex/phase7a-structured-crown-docs` 新增提交：

```text
018fb8b feat: support macos arm64 dev bootstrap
```

定位：

- 当前本机所在分支。
- 在 phase7a 后继线上增加 Apple Silicon / macOS arm64 开发环境支持。
- 主要包括：
  - 新增 `environment-macos-arm64.yaml`。
  - `scripts/install_dev.sh` 自动识别 Darwin arm64，选择 macOS 环境文件。
  - macOS 上关闭 CUDA，默认 LLVM CPU backend，不默认启用 Metal。
  - 用 `sysctl -n hw.ncpu` 兼容 macOS 核心数检测。
  - 用 CMake override 取代 Linux/GNU `sed -i` 改 TVM 配置。
  - TVM-FFI 与 TVM Python 包从 vendored 根目录 editable install。
  - `scripts/rebuild_tvm.sh` 在 build 目录缺失时给出明确提示。

用途建议：

- 这是目前远程上的最新完整开发线。
- 对 Linux 开发没有破坏性：Linux 仍走原 `environment.yaml`，CUDA 默认仍按原路径启用。
- 如果要换 Linux 继续推 TVM 算子，推荐从 `origin/feat/macos-arm64-dev-env` 新开分支，例如：

```bash
git fetch --all --prune
git switch -c feat/linux-tvm-operators origin/feat/macos-arm64-dev-env
```

## 当前本机未提交工作区

当前工作区包含大量未提交改动。它们不是 `origin/feat/macos-arm64-dev-env` 的一部分。

已修改文件集中在：

- `boundflow/runtime/linear_operator.py`
- `boundflow/runtime/crown_ibp.py`
- `boundflow/runtime/perturbation.py`
- `scripts/bench_phase7a_shared_crown_path_attribution.py`
- `tests/test_phase7a_linear_operator_concretize.py`
- `tests/test_phase7a_pr11_shared_crown_bench.py`
- `docs/change_log.md`
- `gemini_doc/next_plan_after_phase7a_pr14.md`

新增文件包括：

- `boundflow/runtime/bound_planner.py`
- `environment-macos-arm64-mps-aggressive.yaml`
- `environment-macos-arm64-mps-nightly.yaml`
- Phase 7B benchmark/report 脚本：
  - `scripts/bench_phase7b_crossover_matrix.py`
  - `scripts/postprocess_phase7b_cost_model.py`
  - `scripts/report_phase7b_planner_v2_candidates.py`
  - `scripts/report_mps_*`
- Phase 7B / MPS / Metal 相关测试：
  - `tests/test_phase7b_*`
  - `tests/test_mps_*`
- 多份 PR-15 到 PR-28 的 `gemini_doc/change_*.md` 记录。

按意图归类如下：

- PR-15：operator attribution。给 shared CROWN ReLU pullback 增加 opt-in attribution，记录 materialization、fallback、operator depth 和 wrapper 成本。
- PR-16：run-local dense cache。给 `LinearOperator.to_dense()` 增加 CROWN run 内 cache，降低重复 materialization。
- PR-17：final concretization policy。增加 `structured` / `dense_barrier` 策略对比，支持 benchmark 显式选择。
- PR-18：hybrid planner / capability table。新增 `bound_planner.py`，把 operator capability、cache 策略和 final policy 选择显式化。
- PR-19：Phase 7B crossover matrix。建立 workload × scale × policy 的 benchmark matrix。
- PR-20：cost model v1 postprocess。把 matrix 输出转成可审计的 policy recommendation。
- PR-21：CPU matrix evidence。记录 CPU 上的正式 matrix 结果和 guardrails。
- PR-22：planner v2。只提升 high-confidence CPU rule，其他 evidence 保留不进入默认 planner。
- PR-23 到 PR-28：Mac MPS aggressive lane、MPS env var sweep、prefer-metal guardrails、MPS dispatch profile、custom Metal kernel feasibility。

这些 WIP 更偏向“解释和选择 shared CROWN / MPS 路径”的 benchmark/planner 层，不是直接 TVM 算子 lowering。切到 Linux 前需要决定：

- 如果这些 PR15-PR28 内容仍要保留，应先整理成一个或多个 WIP commit 并推送。
- 如果 Linux 只继续 TVM 算子，不依赖这些 MPS/Phase7B 改动，应先不要把它们混入 TVM 算子分支。

## Linux TVM 算子开发建议

推荐基线：

```bash
origin/feat/macos-arm64-dev-env
```

推荐新分支：

```bash
feat/linux-tvm-operators
```

理由：

- 它包含 `origin/codex/phase7a-structured-crown-docs` 的 Phase 7A 后继工作。
- 它额外包含 macOS bootstrap，但 Linux 路径仍使用原 `environment.yaml`，不影响 Linux/CUDA 安装。
- 比 `origin/main` 更接近当前真实研发前沿。

Linux 机器上建议先跑：

```bash
git fetch --all --prune
git switch -c feat/linux-tvm-operators origin/feat/macos-arm64-dev-env
bash scripts/install_dev.sh
conda run -n boundflow python tests/test_env.py
conda run -n boundflow python -m pytest -q tests/test_phase4c_tvmexecutor_matches_python.py tests/test_phase4c_tvmexecutor_matches_python_cnn.py tests/test_phase4d_onnx_frontend_matches_torch.py
```

TVM 算子推进建议从以下边界开始：

- 不要从 MPS/Metal WIP 直接切入 TVM lowering；那批内容主要是 Mac 侧性能探索。
- 优先确认 Linux 上 TVM/LLVM/CUDA 路径能稳定重建。
- 继续沿 `boundflow/backends/tvm/` 和 Phase 4C/4D 测试推进真实算子 lowering。
- 如果要利用 PR18/PR22 的 planner 思路，先把当前 WIP 整理成独立分支提交，再明确它和 TVM 算子分支的依赖关系。

## 当前风险与操作建议

- 当前本机工作区不干净，不能直接无脑切分支；否则容易把 PR15-PR28 WIP 混入 Linux TVM 算子分支。
- 本文档可以独立提交；但提交本文档不会保存当前 WIP 代码。
- 若要完整保存当前 WIP，建议另开分支提交：

```bash
git switch -c wip/phase7b-planner-mps-evidence
git add boundflow runtime scripts tests docs gemini_doc environment-macos-arm64-mps-aggressive.yaml environment-macos-arm64-mps-nightly.yaml
git commit -m "wip: capture phase7b planner and mps evidence"
git push -u origin wip/phase7b-planner-mps-evidence
```

上面命令只是保存 WIP 的思路，实际执行前应重新检查 `git status --short`，避免把无关本地文件带入提交。

