# Phase 6F（β / αβ-CROWN MLP）落地计划（PR 级）

## 目标（为什么做）

Phase 6E 已经把 “split state + α-CROWN 作为 bound oracle + BaB driver” 的工程骨架跑通，但其对一般情形仍是 **sound 但不保证 complete**（split 约束没有被通用地编码进 bound propagation）。Phase 6F 的目标是：

- 把 ReLU split 约束（active/inactive）以 **β 参数化形式**编码进 bound propagation（MLP 链式子集先行）；
- 在 BaB 中用 αβ-CROWN 作为节点 bound oracle，使 “split 约束生效、infeasible split 可剪枝、证明/反证更稳定”，为后续性能化（batching/caching）打语义基础。

## 范围（本阶段只做什么）

- 仅支持链式 MLP（`linear`+`relu`）与你当前 `run_crown_ibp_mlp` 的约束（single-task、last output）。
- β 只对 **已 split 的 ReLU neuron** 生效（稀疏存储优先），先追求 correctness，再追求速度。
- BaB 仍在 Python 控制流；不引入 TVM。

## 数据结构（建议）

### 1) SplitState（沿用现状）

- `ReluSplitState.by_relu_input: Dict[str, Tensor[int8][H]]`，值域 `{-1,0,+1}`。

### 2) BetaState（建议稀疏）

### Phase 6F 建议：先 dense（或 dense+mask），稀疏优化延后

理由：Phase 6F 的首要目标是 correctness + 可微优化闭环。稀疏结构（嵌套 dict / 动态索引）会让 autograd、batching、cache key 与后续 GPU 并行更复杂。
折中策略：

- Phase 6F：用 `Dict[str, Tensor[H]]` 存 dense β，并从 `split_state` 派生 mask（仅对已 split neuron 生效）；保持张量图连续与梯度稳定。
- Phase 6G：在确认 split neuron 稀疏且 batching/caching 方案稳定后，再做存储与算子的稀疏化（优化内存/吞吐），不影响 6F 的语义闭环。

建议接口（Phase 6F）：

- `BetaState.beta_by_relu_input: Dict[str, Tensor]`，每个 Tensor 形状为 `[H]`（与该 ReLU 输入的 neuron 维度一致）
- 生效 mask：`mask = (split_state != 0)`（同样形状 `[H]`），仅对 mask==True 的位置注入 split encoding/参与优化

最小默认：β 初始化为 0，并在 update 后做投影/约束（范围按你的后续推导确定；Phase 6F 先保守以 correctness 为先）。

## API（建议签名）

### 1) bound oracle：αβ-CROWN

新增文件：`boundflow/runtime/beta_crown.py`（或 `alpha_beta_crown.py`）

建议入口：

```py
def run_alpha_beta_crown_mlp(
    module: BFTaskModule,
    input_spec: InputSpecLike,
    *,
    linear_spec_C: Optional[torch.Tensor] = None,
    relu_split_state: Optional[Dict[str, torch.Tensor]] = None,
    steps: int = 20,
    lr: float = 0.2,
    alpha_init: float = 0.5,
    beta_init: float = 0.0,
    warm_start_alpha: Optional[AlphaState] = None,
    warm_start_beta: Optional[BetaState] = None,
    objective: Literal["lower","upper","gap","both"] = "lower",
    lb_weight: float = 1.0,
    ub_weight: float = 1.0,
) -> Tuple[IntervalState, AlphaState, BetaState, Stats]
```

实现策略（最小闭环）：

- forward：复用你现有 `_forward_ibp_trace_mlp(...)` 拿到 `relu_pre`（IBP pre-activation bounds）
- backward：沿用 `run_crown_ibp_mlp` 的符号选择逻辑，但在 ReLU 处：
  - 对 split neuron：除了将其变为 stable（pos/neg），还要把 split 约束以 β 形式注入到 affine form / bias 更新里（**Phase 6F 的核心**）
- optimization：用 autograd 同时更新 α 与 β；并提供 warm-start（子节点继承父节点 best α/β）。

### 2) BaB：把 oracle 换成 αβ

更新：`boundflow/runtime/bab.py`

新增配置项：

- `oracle: Literal["alpha","alpha_beta"] = "alpha_beta"`
- `beta_steps/beta_lr/beta_init`

节点结构：

- `(split_state, alpha_state, beta_state)`

## 必须有的测试（DoD）

### DoD-1：β 参数有梯度且 finite（硬钉子）

新增：`tests/test_phase6f_beta_crown_grad.py`

- 构造一个存在 split state 的 case
- `loss = -lb.mean()` 反传
- 断言：`beta.grad is not None` 且 finite 且非零

### DoD-2：infeasible split 能被识别并剪枝（必须用 β 才能过）

新增：`tests/test_phase6f_infeasible_split_prune.py`

构造：一个 **非平凡空域**，其不可行性来自多条 split 约束在输入空间中的联合（而不是“同一个 neuron 同时 active/inactive”这种 O(1) merge check 就能判掉的冲突）。

一个可操作的 toy 思路（MLP first layer）：

- 让两个 ReLU pre-activation 在输入上呈确定的仿射关系（例如同权重、bias 相差常数：`h2 = h1 + 1`）；
- split 选择 `h1` active（`h1>=0`）且 `h2` inactive（`h2<=0`），该子域在输入空间上为空；
- 仅靠“对每个 neuron 单独做区间收缩”通常无法立即识别该组合为空域，而 β-style 的 split constraint encoding 应能把该空域更早标记 infeasible 并 prune（或让 bound 立即达到可剪枝条件）。

期望：

- bound oracle 抛出明确的 infeasible（或返回标记），BaB 侧将其 prune；
- 与 Phase 6E 的 “1D Linf patch” 无关（不能靠输入域收缩补丁兜底）。

### DoD-3：BaB + αβ 能在更一般 toy 上稳定证明（对比 α-only）

新增：`tests/test_phase6f_bab_alpha_beta_vs_alpha.py`

构造一个小 MLP（2~3 ReLU），设置 `alpha_steps=0`（或很小），并限制节点数：

- α-only：在 `max_nodes` 内返回 `unknown`（或证明失败）
- αβ：在同 budget 内返回 `proven`（或 prune 更快）

注：这里不追求论文级案例，只要能稳定复现 “β 编码使 split 约束进入 bound propagation，减少卡住/误剪枝” 的工程效应。

## 产物与验收

- 代码：新增 `beta_crown` 模块；`bab.py` 可切换 oracle；测试全绿。
- 文档：新增 `gemini_doc/change_YYYY-MM-DD_phase6f_beta_crown_mlp_*.md`（动机/改动/验证/限制），并在 Phase 6E 文档中强调 “complete 需 β/LP” 的口径（已做）。

## 不做（避免返工/膨胀）

- 不在 6F 引入 TVM/编译后端。
- 不在 6F 做 multi-node batching（留到 6G）。
- 不在 6F 做 CNN/conv2d 的 β 编码（先把 MLP 语义闭环钉死）。
