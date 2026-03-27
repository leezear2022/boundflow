# 下一步计划：Phase 7A PR-10——ReLU barrier 结构化

## 1. 当前状态

PR-9 已经把 general DAG backward 里最明显的两个 dense barrier 去掉了：

- DAG 汇合点的 adjoint merge 不再依赖 `to_dense() + sum`
- `concat` backward 不再通过 dense slice 回退到 `DenseLinearOperator`

这意味着 `{add, concat}` 的 general DAG 子集终于和 PR-2 / PR-3 / PR-4 的 operator 化主线对齐：图的结构性不再在 merge/slice 热路径被提前打平。

但 backward 热路径里仍保留一个更核心、也更广泛的 dense barrier：

- `ReLU` backward 仍然把线性界状态压回显式 dense 形式后再做 relax/alpha/beta 处理

因此 PR-9 之后最自然的下一步，不是继续扩新的图语义，而是继续沿“扩表达”方向推进，把 `ReLU barrier` 作为下一个明确的系统瓶颈来处理。

---

## 2. 为什么 PR-10 应该先做 ReLU barrier

原因有三条：

### 2.1 它是当前 backward 主路径里剩下的最大 dense barrier

PR-9 之后，`add/concat` 已经不再强制 materialize；但只要经过不稳定 ReLU，线性界状态仍会在 backward 中回落到 dense。  
这会让：

- plain CROWN
- alpha-CROWN
- alpha-beta-CROWN
- BaB oracle

都继续共享这一处显式大张量开销。

### 2.2 它比继续扩图更符合当前主线

PR-8 已经把 solver 栈扩到了最小 residual/general DAG；PR-9 又把 DAG merge/slice 的 operator 化补上。  
此时再继续扩新 merge/op，收益会再次被 ReLU barrier 吃掉。

换句话说，当前最需要补的不是“图能不能表达”，而是“表达之后能不能不被 dense barrier 抵消”。

### 2.3 它比先做更强 lazy row-norm 更接近 solver 主路径

`RightMatmulLinearOperator` 和更一般 operator 组合的 lazy row-norm 当然值得做，但那一层主要影响的是 concretize / 范数归约效率。  
而 `ReLU barrier` 是直接卡在 backward 主路径里的公共瓶颈，优先级更高。

---

## 3. PR-10 建议目标

PR-10 的目标建议收敛为：

- 不扩新图语义
- 不扩新 merge/op
- 不改 public API
- 只把 `ReLU backward` 从“显式 dense barrier”推进到“尽量 operator-preserving 的 barrier”

更具体地说：

1. 保持现有 ReLU relax/alpha/beta 语义不变  
2. 尽量让进入 ReLU 前后的线性界状态保持 `LinearOperator` 形式  
3. 只在确实无法避免时，才在 barrier 内部局部 materialize，而不是把整个后续状态永久回退成 `DenseLinearOperator`

---

## 4. PR-10 非目标

为了避免 scope 失控，建议明确不做：

- 不扩 `mul/gate`、新 merge、新 DAG 语义
- 不扩 deeper infeasible detector
- 不把所有 operator 的 row-norm 一次性全部结构化
- 不在同一 PR 里同时重写 alpha / alpha-beta / BaB 顶层算法接口

alpha / alpha-beta / BaB 应继续通过共享的 CROWN backward 路径自动继承 PR-10 的收益，而不是再开第二套 solver 入口。

---

## 5. PR-10 预期收益

如果 PR-10 做成，预期收益主要在三层：

1. general DAG + chain CNN 的 backward 热路径里，dense materialize 点进一步后移  
2. alpha / alpha-beta / BaB 的 oracle 路径会自动吃到这部分表达收益  
3. 后续如果再做更强 lazy row-norm、或者更一般 operator 组合，就会建立在更干净的主路径上，而不是继续被 ReLU barrier 提前打平

---

## 6. 建议测试入口

PR-10 至少应覆盖下面几类回归：

- `tests/test_phase7a_pr9_operator_preserving_dag_backward.py`
  - 确保 PR-9 的 DAG operator 路径不被回退
- `tests/test_phase7a_pr5_alpha_crown_cnn.py`
  - conv + structured alpha 路径继续可运行
- `tests/test_phase7a_pr6_alpha_beta_crown_cnn.py`
  - conv split/beta 路径继续可运行
- `tests/test_phase7a_pr7_bab_chain_cnn.py`
  - BaB 继续通过共享 oracle 路径工作
- 新增 PR-10 专项测试
  - 锁定 ReLU backward 在 operator barrier 前后的状态形式
  - 锁定数值 soundness 与现有 dense 参考一致

---

## 7. 为什么现在不把 lazy row-norm 作为并列主线

不是说它不重要，而是当前排序应该是：

1. 先把 backward 主路径里的最大 dense barrier（ReLU）继续结构化
2. 再补更一般 operator 组合上的 lazy row-norm

否则会出现一个不划算的情况：

- 你把 row-norm 做得更 lazy 了
- 但主路径仍在 ReLU 处把状态打平

那样收益会被主路径更早的位置吃掉。

---

## 8. 一句话结论

PR-9 之后，下一步最自然的主线是：

**PR-10：ReLU barrier 结构化。**

它比继续扩图、也比先补更一般 lazy row-norm，更符合当前 BoundFlow 从“能跑 general DAG”走向“general DAG 真正保持 operator 表达能力”的节奏。
