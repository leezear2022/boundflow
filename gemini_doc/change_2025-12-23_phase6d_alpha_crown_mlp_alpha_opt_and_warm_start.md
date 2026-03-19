# 变更记录：Phase 6D（α-CROWN MLP）起步——ReLU 下界 α 参数 + K-step 优化 + warm-start

## 动机

Phase 6B/6C 已经把 CROWN-IBP（MLP: Linear+ReLU）的 correctness 与 multi-spec 真 batch 的系统收益（吞吐/forward 复用）钉牢；Phase 6D 进入“bound optimization”阶段：

- 将不稳定 ReLU 的 lower relaxation 从固定 DeepPoly baseline（取 0）扩展为参数化形式 `y >= α x`（`α∈[0,1]`）；
- 用最小 K-step 的 autograd 优化循环收紧 bounds，并提供 warm-start 以便后续复用（面向 αβ/β-CROWN 与 BaB 的工程接口形态）。

## 本次改动

- 更新：`boundflow/runtime/crown_ibp.py`
  - `run_crown_ibp_mlp(..., relu_alpha=...)`：为每个 ReLU 节点提供可选的 `α` 输入（按 relu 输入 value_name 索引）。
  - 对不稳定区间（`l<0<u`）：lower slope 使用 `α`（并在执行期 clamp 到 `[0,1]`），其余区间保持原语义不变。

- 新增：`boundflow/runtime/alpha_crown.py`
  - `AlphaState`：存储每个 ReLU 节点的 `alpha_by_relu_input`（按 neuron 维度，跨 batch/spec 共享）。
  - `run_alpha_crown_mlp(...)`：最小 α-CROWN 优化循环：
    - `steps/lr/alpha_init/objective` 可配置；
    - 支持 `warm_start`（复用上一轮 alpha 作为初值）；
    - 以 “best-of（包含 step=0）” 的方式返回最优 bounds，保证不会因为后续 step 变差而回退。

- 新增测试：`tests/test_phase6d_alpha_crown_mlp.py`
  - `alpha_init=0.5` 的 toy case（不稳定 ReLU）：优化后 lower bound 应不劣于 step=0（并趋近于 0）。
  - soundness：对 `L∞` 扰动采样点检查 `f(x)` 落在 `[lb,ub]` 内。
  - warm-start：同样 step 数下 warm-start 的结果不劣于 cold-start。

- 新增 bench：`scripts/bench_phase6d_alpha_opt_convergence.py`
  - 输出每一步 `lb_mean/ub_mean/alpha_mean` 的轨迹（stdout JSON），便于观察收敛与口径回归。

## 如何验证

```bash
python -m pytest -q tests/test_phase6d_alpha_crown_mlp.py

# 观察优化轨迹（CPU）
python scripts/bench_phase6d_alpha_opt_convergence.py --device cpu --steps 20 --specs 32 --objective lower
```

## 备注

- 当前 α 参数化仅作用于不稳定 ReLU 的 lower relaxation；upper 仍使用既有 secant 形式。
- 后续扩到 αβ/β-CROWN 时，可在此基础上引入 split 约束的编码与更强的复用策略（subproblem warm-start / cache key 细化等）。

