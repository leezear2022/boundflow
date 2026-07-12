# BoundFlow ASPLOS Claims Map

> 本表是动态证据账本。`planned` 不代表已经实现；只有代码、测试和工件均存在时才能改为
> `validated`。当前执行基线为 Gate 0。

| Claim | 当前状态 | 代码/设计落点 | 必需测试 | 必需工件 |
|---|---|---|---|---|
| C1：显式物化语义的 Structured Bound-Operator IR | partial（PR-9 基础） | `boundflow/runtime/linear_operator.py`、`crown_ibp.py` | dense/operator 数值与 gradient 对齐；materialization trace | count/bytes/reason/lifetime JSONL |
| C2：Query/Memory-Aware Materialization Planner | planned（PR-11） | `boundflow/planner/`，具体接口待 PR-11 定稿 | fixed/local/global/oracle；预算约束 | latency–memory Pareto、plan dump |
| C3：BaB-Oriented Repeated-Query Runtime | partial（现有 node batch/cache） | `boundflow/runtime/bab.py` 等；QueryState 接口待 PR-13 | same-solver executor 对齐、state validity | TTV、solved/timeout、p90/p99、batch fill |
| TVM 后端执行 Planner 结果而非定义核心抽象 | partial | `boundflow/backends/tvm/`、`runtime/tvm_executor.py` | Python/TVM/unfused/fused 对齐 | compile/cold/warm、launch、bytes |
| 相同浮点语义下保持 reference bound computation | partial | dense reference + planned paths | allclose、gradient、auto_LiRPA、replay | correctness fields 与失败记录 |

## PR-10 子阶段

| 子阶段 | 状态 | 完成证据 |
|---|---|---|
| PR-10A Materialization instrumentation | validated | `25225e5`；ReLU barrier opt-in trace |
| PR-10A.1 Trace schema v1 | validated | `boundflow.materialization/v1`、schema contract tests、164 passed |
| PR-10B 真实 workload profile | in progress | trace-on/off runner；mechanism + mini-ResNet、method/spec/domain scan |
| PR-10C Dense/gradient reference oracle | planned | local step、full backward、solver、α gradient |
| PR-10D Structured ReLU transform | planned | exact SignSplit operator；不得错误穿过 matmul/conv |
| PR-10E 全路径回归与 benchmark | planned | correctness/structural/opportunity gates |

## 当前 Gate 0 证据

- PyTorch 2.12.1+cu132、CUDA 13.2、LLVM 20.1.8、TVM 与单一内嵌 tvm-ffi 已完成现场验证；
- MLP/CNN reduced artifact 已生成：small matrix、warmup 3、iters 10，2 行均通过 correctness；
  它是 Gate 0 回归，不替代论文要求的至少 5 次独立重复；
- Gate 0 已冻结在本地提交 `4e0e059`，全量验证为 162 passed、1 个预期 skip；
- PR-10 已从 opt-in materialization instrumentation 启动，尚未修改 ReLU operator 路径。
