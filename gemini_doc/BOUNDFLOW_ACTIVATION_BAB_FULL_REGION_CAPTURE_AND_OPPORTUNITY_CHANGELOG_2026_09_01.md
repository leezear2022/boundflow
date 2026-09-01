# BoundFlow activation-BaB 全段事务捕获与剩余机会记录

status: implemented-and-locally-validated
date: 2026-09-01
branch: feat/rvir-v4-production-state-ownership-v1
external-audit: not-requested
performance-claimed: false

## 1. 为什么不继续沿 `/input-8 → /input-4` 做

重新逐节点核对当前累计 owner 后确认：

```text
/49
  → /48 → /input-28                    terminal
  → /45 → /44 → /43 → /input-24
       → /input-20 → /input-16         residual
  → /39 → /37 → /input-12 → /input-8
       + /38 → /input-4                projection residual
  → /input → /input-1                  input-domain concretization
```

所以 current candidate 已覆盖 initial/root CROWN 从 `/49` 到 `/input-1` 的完整图；此前变更记录中
把 `/input-8 → /input-4` 写成下一段是错误的，已在原文更正。

complete query 中真正未接管的是后半段 activation-BaB `stage_solve.update_bounds_core`。

## 2. 可重算的剩余机会

新增 `scripts/analyze_root_crown_remaining_opportunity.py`，只读取已冻结的三对 same-solver raw，
不采信手写摘要。独立重算结果：

- 当前 control/candidate complete-query geomean：`1.1202264917767388x`；
- candidate 中 activation-BaB solve share：`20.8229%–21.6392%`；
- 若该 BaB region 加速 `2x`，candidate 自身 query 预计再快 `1.1187632x`，相对原 control
  预计 `1.2532682x`；
- 若 region 加速 `4x`，两者分别为 `1.1893931x` 与 `1.3323897x`；
- 若 region 加速 `10x`，两者分别为 `1.2362211x` 与 `1.3848476x`；
- 即使该 region 在当前“一轮 BaB”工作负载上耗时归零，相对原 control 的理论 geomean 也只有
  `1.4221767x`。

以上均是从真实 share 得到的 Amdahl 投影，`projection_is_claim=false`，不是实测性能。它说明这
一刀值得做，但也说明单轮 query 不可能产生 10x headline；最终 10x 必须依赖更深 BaB 中该事务的
重复次数、其他 solver transaction 融合以及内存/调度收益共同传播。

## 3. production exact-call 全段捕获

新增 `scripts/probe_bab_full_region_capture.py`，在一个真实 αβ-CROWN CUDA query 中同时安装四段
只读观察器；没有执行 candidate，也没有计时。捕获成功：

- 四段均为 1 个 outer optimizer transaction；
- 每段均为 `10 forward / 9 backward`；
- input：incoming A `[1,6,8,16,16]`，输出系数 `[1,6,3,32,32]`；
- projection：incoming `[1,6,16,8,8]`，输出 `[1,6,8,16,16]`；
- residual：incoming/output 均为 `[1,6,16,8,8]`；
- terminal：incoming `[1,6,100]`，输出 `[1,6,1024]`；
- active β value/location/sign/gradient 均为 `[6,1]`；
- sparse α 分别覆盖 feature count `27/178/86/121/132/164`；
- solver 离散结果保持 `status=verified`，本轮 candidate 未执行。

本地 diagnostic artifact：

- `artifacts/bab-full-region-capture/resnet2b-prop0-v1/capture.pt`；
- capture SHA256：`58c566728f50a5be2b464bf76f8d8b446478a7b8e4f37c03301b62e60ab7c65f`；
- receipt SHA256：`5a61a6be0500ec20b5c417bc42fe2c0115d59219ab3e3695ed7aba97a69d6a1a`。

artifact 仍是本地诊断输入，不形成性能或 formal claim。

## 4. 捕获过程中识别出的四个必要语义差异

探针不是一次“绿灯脚本”；前三次 fail-closed 分别暴露了真实设计差异：

1. β 容器既可能是 start-node keyed mapping，也可能是单元素 sequence，不能硬编码一种布局；
2. active β 由 preactivation Linear `/input-28` 持有，不由 ReLU `/48` 持有；
3. BaB 的 β 在 ReLU coefficient 产生后、Linear 消费前注入，因此必须区分
   `relu_output_lower_a` 与 `linear_incoming_lower_a`；
4. BaB 使用冻结 intermediate bounds，这些 lower/upper 不进入 VJP；root α-CROWN 中相同 bounds
   则参与 VJP。

第 3、4 条证明不能把当前 root template 只改 shape 后直接调用。新 full owner 必须显式表达
`beta injection effect` 与 `frozen-bound gradient ownership`。

## 5. 代码变化

- 四个历史 root capture 源文件保持逐字节不变，旧 formal artifact 的 code-revision replay 继续成立；
- 新增独立、版本化的 BaB capture contract，固定 `10/9` 生命周期；
- BaB terminal capture 增加 active β value/location/sign/gradient，以及 β 注入前后的两份
  coefficient；
- BaB residual/projection/terminal capture 显式区分可微状态与 frozen bounds；
- root 历史合同现场重跑仍为 `5 forward / 4 backward`，bounds gradients 全存在；
- 新增剩余机会分析器和 4 个确定性单测。

## 6. 下一实现刀

下一步不再外审，也不再做独立单算子旁路。直接实现 activation-BaB full-owner correctness：

1. 从本轮 capture 构造 `(spec=1, domain=6)` 的 terminal/residual/projection/input templates；
2. 在 terminal TIR 中加入 sparse β scatter/add，位置固定在 ReLU 输出与 Linear 输入之间；
3. custom backward 只返回 α、β 和 incoming coefficient 的 VJP，不为 frozen bounds 生成梯度；
4. 10 evaluation / 9 mutation 全轨迹与 PyTorch oracle 闭合后，再接 production live bridge；
5. correctness 通过前不计时；之后才跑三对 fresh same-solver，目标先验证 region `>=4x` 是否可达。

这一步的研究价值不是“又加一个 kernel”，而是把 verification-specific split state、compressed
α/β、frozen-bound ownership 和整图 generate-and-consume 放进同一个 TVM/TIR transaction。

后续状态：上述 full-owner correctness 已在
`BOUNDFLOW_ACTIVATION_BAB_FULL_OWNER_CORRECTNESS_CHANGELOG_2026_09_01.md` 实现并本地验证；当前
下一刀已推进为版本化 TVM/TIR lowering，仍未开放 timing 或 performance claim。

## 7. 本地验证

- 真实 external αβ-CROWN CUDA probe：PASS，四段 `10/9`、active β `[6,1]`；
- root terminal 默认 `5/4` 现场回归：PASS；
- 历史 terminal/residual formal artifact code-revision replay：PASS；
- 相关专项：`80 passed`；
- 全量：`2188 passed, 3 skipped`；
- mypy：4 个本轮源文件 clean；
- pylint：`10.00/10`；
- `git diff --check`：PASS。

3 个 skip 均为既有 TVM 重复编译或冻结 VNN-COMP checkout 边界。本轮未请求、未执行新外审。
