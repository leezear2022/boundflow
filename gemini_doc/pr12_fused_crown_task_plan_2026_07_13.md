# PR-12：Fused CROWN-Task Lowering for Memory-Efficient Plain CROWN

> 状态：执行中；PR-12D correctness 已关闭，PR-12E 正式 runtime/Pareto 证据链与 PR-12F
> frozen held-out 已完成，但性能门禁失败、Planner quality 仅 guarded/partial。不得并行扩展
> α/αβ autograd、BaB scheduler 或 training，PR-13 继续阻塞。

当前分支从只读 tag `pr11-validated-reduced` 创建。起点工件位于
`artifacts/phase7a-pr12/baseline/`，候选 schema 见
`gemini_doc/backend_candidate_schema_v1.md`。Linear 与 Conv 使用不同 capability id；Conv
signature 额外过滤 kernel/stride/padding/dilation/groups/layout 和 input-shape contract。

## 假设

对静态 shape、FP32、CUDA、`requires_grad=False` 的 plain CROWN 查询，将 ReLU sign
selection、relaxation-bias reduction 与相邻 affine backward 合并为一个 TIR task，可以保留
structured 路径的显存优势，并降低 eager Python/operator dispatch 与中间算子开销。

PR-11 的 regret attribution 显示 9 个高 regret case 全部首先来自有限候选集未包含 measured
oracle；7 个 case 同时带 backend-gap 假设标记。因此 PR-12 只验证后端 Pareto 改善，不能包装
成 PR-11 cost-model 修复。

## 范围

支持：plain CROWN、无梯度、inference/final-bound、static shape、FP32、CUDA；先做
ReLU+Linear backward correctness foundation，再做 headline 所需的 ReLU+Conv2d backward。

明确排除：α/αβ optimization、differentiable structured/custom autograd、training、新 BaB
scheduling、dynamic shape、attention、multi-GPU 和新 branch heuristic。

## Planner/backend 合同

计划必须分别表达 placement（dense/structured）和 backend variant（PyTorch eager、TVM
unfused、TVM fused TIR），并通过 capability 过滤 grad/α/β/conv/dtype/layout/static-shape，
不能把 fused path 隐藏在原有 `STRUCTURED` 标签中。

编译 cache key 至少包含 semantic/operator DAG hash、fused pattern、shape、spec/domain bucket、
dtype/layout、target/compute capability、TVM commit、Planner/backend schema、schedule/tile 参数。
weights、bounds 和 α/β 默认是 runtime tensor，不烘焙进 key。

## 实现顺序

1. 冻结 `backend_profile_v1`：dense eager 与 structured eager；
2. 增加 fused task IR/schema、capability 与 cache key contract；
3. ReLU+Linear TIR：不写回完整 scaled-A；
4. ReLU+Conv2d TIR：融合 sign/slope/bias/transpose-conv composition；
5. thin Relax `call_tir` orchestration；
6. Python dense、structured eager、TVM unfused/default、BoundFlow fused 三方以上对照；
7. 端到端 Planner auto + fused 以及 compile amortization。

## 门禁

- Correctness：Linear/Conv/end-to-end 0 failure；finite、lower<=upper、dense reference allclose；
  α/αβ 不会误选 fused；cache key 无错误命中；PR-10/PR-11 全回归通过。
- Mechanism：不写回完整 ReLU-scaled A；至少 sign-split+bias reduce+affine composition 形成
  粗粒度 task；launch 与 logical materialization bytes 下降。
- Performance（内部目标，不是论文既成结果）：相对 structured eager 代表 workload 几何平均
  >=2×；保留相对 dense 至少 20% peak-memory 优势；dense 可运行点尽量 <=1.5× dense latency；
  至少一个 dense OOM 点成功；break-even 落入真实 repeated-query 规模。

## 证据版本

PR-11 profile/schema/tag 只读。PR-12 新建 `backend_profile_v2`，显式包含 dense eager、
structured eager、TVM unfused/default 与 structured fused TIR；compile/cold/warm、launch、memory、
cache 与 correctness 全部写入新 JSONL 和 manifest。

## 2026-07-13 PR-12E/F 实际判定

- calibration 12/12、held-out 24/24 candidate rows correctness 通过；default/custom stream 均测；
- fused 在 5 个 held-out 上均降低 peak，其中 64 MiB memory-sensitive Linear 从 68.599 MiB
  降到 29.282 MiB，但 latency 从 2.025 ms 增至 8.516 ms；
- 另外两个 Linear warm speedup 为 1.088×/1.079×；Conv 与三 block mini-ResNet 分别为
  0.792×/0.968×；
- calibration-only Planner 5/5 预算可行、0 unsafe，median/p90/max regret 为
  1.000×/1.262×/1.262×，3/5 选择更快或为预算唯一可行；fanout fallback 1/1；
- 因未达到性能目标且缺 structured eager/TVM-unfused 正式对照，PR-12 overall 保持
  IN PROGRESS。详见 `gemini_doc/change_2026-07-13_pr12ef_runtime_pareto_heldout.md`。
