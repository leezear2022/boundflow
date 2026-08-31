# Root CROWN 末端 ReLU/Linear TIR 全 VJP 接入与实测记录

status: implemented-and-measured-reduced
date: 2026-09-01
external-audit: not-requested
performance-claimed: false

## 1. 本轮解决了什么

本轮没有继续做审计流程，而是在真实 αβ-CROWN 的 root incomplete query 中，把末端
`/48 ReLU → /input-28 Linear` 的 lower-CROWN 事务替换为一个 TVM/TIR owner：

```text
incoming lower A + preactivation lower/upper + compressed α + Linear weight/bias
  → fused TIR forward
  → output A + accumulated bias
  → custom backward
  → compressed dα + preactivation dl/du
```

它接管真实优化器的一次五步轨迹：5 次 forward、前 4 次 backward，最后一次 forward 作为
terminal evaluation。host solver 仍拥有 optimizer、branch、queue、termination 和最终提交。

## 2. 真实生产形状与语义

从 ResNet2B property 0 的真实 root transaction 捕获到：

| 对象 | 形状/值 |
|---|---|
| incoming lower A | `[3, 1, 100]` |
| preactivation lower/upper | `[1, 100]` |
| compressed raw α | `[2, 3, 1, 27]` |
| Linear weight/bias | `[100, 1024]` / `[100]` |
| output A/bias | `[3, 1, 1024]` / `[3, 1]` |
| α feature ordinals | `27` 个严格递增索引 |
| optimizer trajectory | `5 forward / 4 backward` |

forward 在同一 compiled transaction 中完成：

1. 按 incoming A 符号选择 lower/upper ReLU slope；
2. 直接形成 Linear 输出 A，不把 ReLU 后 dense A 保存为跨层状态；
3. 同时累计 ReLU intercept 与 Linear bias。

backward 返回完整的一阶状态梯度：`dα`、`d lower`、`d upper`。内部允许有 bounded transient
adjoint/reduction workspace，但它不跨层保存，也不从 runtime owner 逃逸。

## 3. 实现中发现并修正的问题

### 3.1 27 路条件链不适合作为 GPU lookup

第一版把 compressed α 的 27 个位置展开成嵌套条件，单次 candidate 一度约 `5.07 ms`。改成
GPU 上的 `feature→ordinal` 静态映射后，同一事务下降到约 `0.50 ms` 量级。说明这里真正需要的是
verification representation lowering，不是把 Python 特判原样翻译进 TIR。

### 3.2 backward reduction 必须并行化

第一版 dα reduction 串行约 `0.345 ms`。将 reduction 做 32-lane rfactor 后，alpha-only
backward 曾下降到约 `0.0675 ms`；当前 full-VJP backward 中位数为 `0.072864 ms`。

### 3.3 只返回 dα 会破坏真实 optimizer 轨迹

alpha-only custom backward 在 isolated 对照中正确，却在 live solver 中造成最终 lower 约
`0.005–0.006` 的漂移。原因不是 α 公式，而是原生 autograd 同时把梯度传播到 preactivation
lower/upper；漏掉这两个 VJP 后，上游可优化 bounds 的 mutation 轨迹改变。

补齐 `d lower/d upper` 后，五步 α、bounds、output A 和 gradient 重新与 native 对齐。这个失败明确
证明：局部算子数值相同不足以证明 solver transaction 等价，custom backward 必须覆盖所有可变状态。

### 3.4 环境入口

αβ-CROWN venv 直接启动时看不到本仓库 TVM；运行 live worker 前必须先加载 `env.sh`。本轮第一次
刷新数据因此在 compile 前 fail closed，加载环境后按相同协议重跑，未把它计作性能样本。

## 4. 当前数值与局部性能

独立 capture 上进行 5 个 evaluation correctness、100 次计时：

| 指标 | 当前结果 |
|---|---:|
| native full forward+VJP | `0.549376 ms` |
| TIR full forward+VJP | `0.290816 ms` |
| isolated speedup | `1.889084x` |
| TIR forward | `0.110592 ms` |
| TIR backward | `0.072864 ms` |
| 最大数值误差 | `1.117587e-7` |
| lower/gradient sign | 全一致 |
| fallback | `0` |
| DLPack pointer | 全部 exact |

compile 为 `1.239642278 s`，明确排除在 warm query 计时之外。因为 complete-query 尚无稳定收益，本轮
不计算或宣称 compile break-even。

## 5. 十进程 same-solver 传播结果

正式数据是 5 对、10 个 fresh process，偶数 pair 为 control→candidate，奇数 pair 反向，全部使用
同一个 αβ-CROWN、model/property、seed、branch/queue/termination 协议。

| scope | geomean | worst | best |
|---|---:|---:|---:|
| optimized-bounds transaction | `1.024849x` | `0.994968x` | `1.055805x` |
| root incomplete | `1.016168x` | `0.990699x` | `1.041130x` |
| complete query | `1.007267x` | `0.994676x` | `1.018231x` |

五对最大 final lower 差为 `2.264977e-6 <= 3e-6`；branch decision、queue、depth、visited domains、
verdict 等离散语义全部一致。5 个 candidate 的 template/TIR/device-source hash 一致，均为
5 forward、4 backward、0 fallback、DLPack pointer 全 exact。

## 6. 结论边界

本轮成立的是：

- 真实 root CROWN 的末端 ReLU/Linear 可以由 TVM/TIR + custom backward 正确接管；
- compiled transaction isolated 约快 `1.89x`；
- 完整 optimizer trajectory 没有因遗漏 bounds VJP 而漂移；
- 代码路径具备真实 activation receipt 和可复放五对数据。

本轮不成立的是：

- 不声称 complete-query 有稳定加速；最差 pair 仍慢约 `0.53%`；
- 不把 `1.007x` 写成论文性能成果；
- 不声称整个 CROWN、IBP、BaB 或 BoundFlow compiler 已被一个 TIR region 接管；
- 不把内部 transient adjoint workspace 表述为“零中间张量”。

决策是 `mechanism-correct-no-stable-query-speedup`。继续调这个单一 `100→1024` site 的微秒收益，
不能把完整 query 推到目标；下一步必须扩大 compiled region 的事务占比。

## 7. 下一性能切片

下一刀按真实 topology 向前扩，不再做新的审计门：

1. 捕获 `/45 ReLU → /44 Add → /43 Conv` 及 residual 输入的完整 forward/VJP/effect；
2. 复用已经完成的 compiled Conv、staged residual、compressed α 与 bounded arena 资产；
3. 设计一个跨 `ReLU + Add + Conv` 的 structured TIR region，局部卷积项生成后直接消费，避免发布
   dense 中间 A；
4. 先在真实五步轨迹做 correctness，再计 isolated region；只有事务占比与 Amdahl 可达时才重跑
   complete-query paired protocol；
5. 若该 region 仍不足，再扩 `/input-24 ReLU → /input-20 Conv`，而不是继续微调末端 Linear。

## 8. 证据入口

- 正式 artifact：`artifacts/root-crown-terminal-tir/resnet2b-prop0-v1/`
- replay：`scripts/package_root_crown_terminal_five_pair.py`
- production capture：`boundflow/runtime/root_crown_terminal_capture.py`
- TIR：`boundflow/backends/tvm/root_crown_terminal_linear.py`
- custom autograd/runtime：`boundflow/runtime/root_crown_terminal_tir.py`
- live bridge：`boundflow/runtime/root_crown_terminal_live.py`
- same-solver worker：`scripts/run_root_crown_terminal_live_worker.py`
- tests：`tests/test_root_crown_terminal_tir.py`、
  `tests/test_root_crown_terminal_five_pair_artifact.py`

本轮按用户要求不发起外部审计；后续也只在用户明确要求时准备外审材料。

## 9. 本地验证

- production capture：`5 forward / 4 backward`，真实 solver 完成；
- isolated GPU probe：5 evaluation、100 repeats，数值/sign/launch/pointer 全通过；
- five-pair replay：PASS，10 个 fresh process、全部离散语义 exact；
- artifact summary tamper：专项测试拒绝；
- root-terminal 专项：`15 passed`；
- 全量回归：`2122 passed, 4 skipped`；4 个 skip 均为既有 TVM duplicate-cost、冻结
  VNN-COMP checkout 和 cuDNN 环境边界；
- mypy：10 个本轮文件 clean；
- pylint：`10.00/10`；
- black 与 `git diff --check`：PASS。
