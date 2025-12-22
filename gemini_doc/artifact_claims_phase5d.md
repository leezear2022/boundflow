# BoundFlow Phase 5D Artifact Claims（口径/证据映射）

本文用于把“论文/AE 口径”钉死：每条 claim 对应**可复现命令**与**产物文件/字段**。  
注意：本文只基于仓库当前实现（Phase 5D 产线）给出可验证的最小 claims；Phase 5E/6 的扩展 claims 以 TODO 形式预留。

---

## 运行入口（workflow script）

推荐使用一键 runner：

- Quick（CI/冒烟）：`python scripts/run_phase5d_artifact.py --mode quick --workload all --run-id quick_test`
- Full（论文级矩阵）：`python scripts/run_phase5d_artifact.py --mode full --workload all --run-id full_run`
- 降级模式（无 TVM 环境）：`python scripts/run_phase5d_artifact.py --mode quick --allow-no-tvm --run-id quick_no_tvm`

输出目录：

- `artifacts/phase5d/<run_id>/`
  - `results.jsonl`：bench 原始 JSONL（1 行 = 1 配置点）
  - `results_flat.csv`：扁平化逐点记录（画图/透视用）
  - `tables/table_ablation.csv`：分组汇总表（当前为最小版本）
  - `figures/*.png`：示例图（matplotlib 可选依赖；缺失不影响主流程）
  - `MANIFEST.txt`：运行环境、命令、产物清单（自动生成）
    - 包含关键输出的 `sha256`，用于确认传输/拷贝后未漂移。

---

## Claim 1：产线输出可机器解析且 schema 稳定（JSONL contract）

**Claim**
- BoundFlow 的消融评测输出使用 JSONL（stdout payload），并通过契约测试保证关键字段存在且类型/范围稳定。

**Evidence（文件/字段）**
- Schema 文档：`docs/bench_jsonl_schema.md`
- Contract test：`tests/test_phase5d_pr13d_bench_jsonl_schema_contract.py`
- Artifact 输出：`artifacts/phase5d/<run_id>/results.jsonl`
  - 必含：`schema_version/meta/workload/config/planner/tvm/runtime/baseline/correctness`
  - 关键字段：`meta.time_utc`、`runtime.compile_first_run_ms`、`runtime.run_ms_p50/p95`

**Command**
- `python scripts/bench_ablation_matrix.py --matrix small --warmup 1 --iters 1 --output /tmp/out.jsonl --no-auto-lirpa`
- 或：`python scripts/run_phase5d_artifact.py --mode quick`

**Threats**
- 若本地 TVM/依赖未正确构建，bench 可能失败（非 schema 语义问题）。

---

## Claim 2：cold vs hot 计时口径分离（解释编译/缓存冷启动）

**Claim**
- 将首次运行（常含编译触发）与稳态运行分离计时，避免把 cold-start 噪声混入稳态性能比较。

**Evidence（文件/字段）**
- 实现：`scripts/bench_ablation_matrix.py`
  - `compile_first_run_ms`：`scripts/bench_ablation_matrix.py:218`
  - `run_ms_p50/p95`：`scripts/bench_ablation_matrix.py:230`
- Schema 解释：`docs/bench_jsonl_schema.md:39`
- Artifact 表/图：
  - `artifacts/phase5d/<run_id>/results_flat.csv`：列 `compile_first_run_ms`、`run_ms_p50`
  - `artifacts/phase5d/<run_id>/figures/fig_runtime_cold_vs_hot.png`（若 matplotlib 可用）
  - `artifacts/phase5d/<run_id>/figures/fig_runtime_breakdown.png`（若 matplotlib 可用；展示 plan/cold/hot 的堆叠）

**Command**
- `python scripts/run_phase5d_artifact.py --mode full`

**Threats**
- `compile_first_run_ms` 是“第一次 run 的 wall time”，包含编译触发 + 执行，不能被当成“纯编译时间”。

---

## Claim 3：stdout 不被环境提示污染（可用于管道与重定向）

**Claim**
- `env.sh` 默认不会污染 stdout，保证 `python ... > results.jsonl` 的 payload 纯净。

**Evidence**
- 实现：`env.sh:33`
- 回归：`tests/test_env_sh_quiet_stdout.py`

**Command**
- `bash -c "source ./env.sh" >/tmp/stdout.txt 2>/tmp/stderr.txt`

**Threats**
- 其它用户自定义脚本若写 stdout，仍可能污染；因此建议统一用 runner。

---

## Claim 4（最小版本）：postprocess 可从 JSONL 稳定产出 CSV/表/图

**Claim**
- 给定 schema 合法的 JSONL，后处理脚本可稳定生成逐点 CSV、分组汇总表，以及示例图与 MANIFEST。

**Evidence**
- 实现：`scripts/postprocess_ablation_jsonl.py`
- 回归：`tests/test_phase5d_pr13e_postprocess_jsonl.py`
- Artifact 输出：
  - `results_flat.csv`
  - `tables/table_ablation.csv`
  - `tables/table_main.csv`（主表最小版本：核心分组键 + plan/cold/hot + bytes_est + call_tir）
  - `MANIFEST.txt`

**Command**
- `python scripts/postprocess_ablation_jsonl.py artifacts/phase5d/<run_id>/results.jsonl --out-dir /tmp/out`
- 或：`python scripts/run_phase5d_artifact.py --mode quick`

**Threats**
- 图依赖 matplotlib，缺失时会自动跳过，不影响 CSV/表/manifest。

---

## Claim 5（最小版本）：memory/reuse/静态内存规划的“象限视图”可复现

**Claim**
- 基于同一套 JSONL/CSV，给出 memory 与 runtime 的象限式可视化（reuse_on × memory_plan_mode），用于解释“内存规划/复用”对 steady-state 的影响方向。

**Evidence（文件/字段）**
- 源数据：`artifacts/phase5d/<run_id>/results_flat.csv`
  - `physical_bytes_est`（planner 估算）
  - `run_ms_p50`（稳态）
  - `reuse_on`、`memory_plan_mode`
- 图：`artifacts/phase5d/<run_id>/figures/fig_mem_quadrants.png`（若 matplotlib 可用）

**Command**
- `python scripts/run_phase5d_artifact.py --mode full`

**Threats**
- `physical_bytes_est` 是 planner/StoragePlan 估算，不等价于真实峰值显存/内存；用于趋势解释而非硬件峰值证明。

---

## Claim 6（最小版本）：对齐 auto_LiRPA（IBP）并给出 speedup 证据链

**Claim**
- 在相同 workload/eps 下，BoundFlow（reference semantics）与 auto_LiRPA（IBP）bounds 对齐（correctness gate），并输出 hot 口径的相对耗时（speedup）用于论文对比。

**Evidence（文件/字段）**
- 源数据：`artifacts/phase5d/<run_id>/results.jsonl`
  - `baseline.auto_lirpa.*`：`init_ms/run_ms_cold/run_ms_p50/p95`、`baseline_key/cache_hit/spec_hash`
  - `correctness.python_vs_auto_lirpa_gate.ok`、`python_vs_auto_lirpa_max_abs_diff_*`
- 主表：`artifacts/phase5d/<run_id>/tables/table_main.csv`
  - `auto_lirpa_run_ms_p50`、`speedup_hot_vs_auto_lirpa`、`python_vs_auto_lirpa_ok_rate`
- 图：`artifacts/phase5d/<run_id>/figures/fig_speedup_hot_vs_auto_lirpa_by_workload.png`（若 matplotlib 可用）

**Command**
- `python scripts/run_phase5d_artifact.py --mode full --workload all --run-id full_run`

**Threats**
- auto_LiRPA 依赖/算子覆盖可能导致 baseline 不可用（`available=false` + `reason`）；该场景应在表中显式呈现而非静默过滤。

---

## TODO（Phase 5E/6 扩展 claims）

- TODO：加入更真实 workload（CNN/conv2d 等）后，给出端到端速度/内存主图主表与 AE 对应口径。
- TODO：加入更完整的论文 claims→图表编号映射（如 “Figure 4/5, Table 2”），并在 runner 中固定输出命名与筛选规则（失败点/缺失字段/阈值过滤）。
