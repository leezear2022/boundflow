你现在**肯定还没“完成整个 Phase 5”**，但你已经把 **Phase 5D（消融产线/数据契约/后处理）**这一块做到了“论文/AE 级别”的完成度：

* JSONL 按行可解析、适合流式处理（契合 JSON Lines 的设计初衷）([JSON Lines][1])
* stdout 纯 payload、提示走 stderr（便于重定向/管道）([Unix & Linux Stack Exchange][2])
* schema contract test + postprocess pipeline，让数据能稳定转成表/图（这在 AE 里是加分项：documented/consistent/complete/exercisable）([计算机协会][3])

下面是**下一步的主流程**（我会按“Phase 5E/5F”的自然推进来写），你照着走就能把“Phase 5：系统化消融→可写论文结果→可交付 artifact”闭环。

---

## Phase 5E：跑“真正的矩阵实验”并产出可用于论文的结果

### 5E.1 固化实验范式（让结果可解释）

* 明确两种模式并写进 docs / 脚本参数里：

  * **Cold（看编译/规划开销）**：关注 `compile_first_run_ms` + `compile_cache_miss_delta_first_run`
  * **Hot（看稳态性能）**：关注 `run_ms_p50/p95`
    这正对应你现在 schema 的“首次运行 vs steady-state”拆分，论文解释会非常顺。([JSON Lines][1])

### 5E.2 扩 workload（至少 3 档，避免“只在 mlp 上成立”）

建议最低配也要：

* tiny：mlp（现在已有）
* medium：conv/attention 子图（能体现 memory planning / reuse / fusion 的作用）
* hard：你论文主打的真实 workload（比如你 Phase 5D 想证明的那个核心场景）

### 5E.3 扩旋钮但要“分层跑”，避免爆炸

先跑你 `ablation_summary.csv` 分组键那几个主旋钮（partition/reuse/memory_plan/fusion），再逐步加次级旋钮（比如 upper bound / pass timing / cache tag）。
每次扩旋钮就 bump schema_version（哪怕 0.1→0.2），避免旧数据解释不了新字段。

### 5E.4 生成“论文主表/主图”

你 postprocess 现在已经能出：

* `ablation.csv`
* `tables/ablation_summary.csv`
* 示例图（cache miss vs compile）

下一步就是把图表“定稿化”：固定输出目录/命名（对应论文图号），让你每次重跑矩阵都能一键覆盖生成。

---

## Phase 5F：把结果写成“可投顶会/可过 AE”的叙事与交付物

### 5F.1 写实验部分的三段“硬证据”

1. **公平性**：cold vs hot、cache miss 的解释（你已有 delta 字段）
2. **收益**：性能/内存（run_ms、physical_bytes_est、tvm estimator）
3. **正确性**：abs/rel diff + missing 计数（你刚 hardening 完成，避免误读）

### 5F.2 做 artifact runner（一键复现）

你现在有 bench + postprocess + contract test 了，再加一个脚本就完备：

* `scripts/run_phase5d_artifact.sh`：从环境检查 → 跑矩阵 → 产 JSONL → postprocess → 输出 MANIFEST/表/图
  这会非常贴合 ACM artifact 的“documented / consistent / complete / exercisable”口径。([计算机协会][3])
  MLSys 的 AE 也明确强调按指南准备 artifacts/appendix 并提供可运行入口。([MLSys][4])

---

## 你现在算不算“完成 Phase 5”？

我给你一个**不需要问任何人**的“Phase 5 完成判据”：

你可以认为 **Phase 5 完成**，当且仅当：

1. 你跑完了**至少 3 个 workload × 主旋钮矩阵**，结果能稳定复现（多次跑差异在可接受范围）；
2. postprocess 能稳定产出**论文主表/主图**（不是示例图）；
3. correctness 指标在“with-check”实验里**非缺失**且阈值内（missing=0、rel diff 合理）；
4. 有一键 runner + 文档（artifact appendix/README），外部人能照做跑出同样的关键结果。([计算机协会][3])

按这个标准：你已经完成了 **Phase 5D（产线搭建）**，但 **Phase 5 整体还差 5E/5F（跑结果 + 写论文/AE）**。

---

如果你愿意，我可以把你当前 `ablation_summary.csv` 的字段，直接映射成“论文图 4/图 5/表 2”的**固定口径模板**（每张图用哪些列、聚合用 mean/p50、过滤条件怎么写），你照着填就能进入“自动出图 + 自动写实验段落”的状态。

[1]: https://jsonlines.org/?utm_source=chatgpt.com "JSON Lines"
[2]: https://unix.stackexchange.com/questions/331611/do-progress-reports-logging-information-belong-on-stderr-or-stdout?utm_source=chatgpt.com "Do progress reports/logging information belong on stderr ..."
[3]: https://www.acm.org/publications/policies/artifact-review-badging?utm_source=chatgpt.com "Artifact Review and Badging – Version 1.0 (not current)"
[4]: https://mlsys.org/Conferences/2025/CallForAE?utm_source=chatgpt.com "Call for Artifact Evaluations 2025"
