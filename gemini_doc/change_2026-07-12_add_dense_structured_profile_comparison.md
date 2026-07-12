# 变更记录：增加 dense/structured ReLU profile 对照

## 修改

- profile runner 新增 `--relu-modes dense,structured`，每条 query 与 CSV 都记录 mode。
- dense 与 structured 使用相同 workload、method、spec、domain、seed、warmup/repeats 和
  correctness gate。
- verification profile 显式冻结模型权重并记录 `weights_require_grad=false`；α/β 是唯一需要
  autograd 的优化状态，避免把 certified-training 权重梯度混入 verifier memory 口径。
- dense trace 必须包含 persistent `relu_sign_split`；structured trace 不允许 persistent
  materialization，只允许有 reason/site 的 ephemeral bias/concretization reduction。
- 增加 αβ fixed-split 与真实 `solve_bab_mlp` 搜索的 dense/structured 结果、节点数和界对照。

## 测量纪律

两种 mode 的 latency/peak 都来自 trace-off；trace-on 只解释 mechanism。结构化 Python 路径若
稳定明显变慢或 peak 更高，将保留 dense feature flag，不用 logical bytes 单独证明收益。

## 待验证

- clean GPU 双模式矩阵；
- persistent/ephemeral bytes 对照；
- peak allocated/reserved 与 latency guardrail；
- 全量 pytest 与 artifact summary。
