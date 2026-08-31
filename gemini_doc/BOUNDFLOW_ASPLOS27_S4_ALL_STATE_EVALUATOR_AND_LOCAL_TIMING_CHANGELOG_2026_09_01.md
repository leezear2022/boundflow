# S4 all-state evaluator、10/9 optimizer 与本地计时变更记录

date: 2026-09-01
status: implemented-local-observation
performance-claimed: false
complete-query-claimed: false

## 1. 这轮实际解决了什么

此前 S4-1C 只能在冻结参数下做一次梯度计算：系数传播仍读取旧 R3 的完整
`[2,1,D,W]` α 和旧 β，因此不能证明第二轮以后读取的是优化器刚更新的 S4 状态。

本轮完成了下面这条可重复路径：

```text
六份 compact α + 一份 active β
  → compact-state coefficient TIR
  → selector Pass A + lower
  → six-site value graph Pass B
  → compact dα/dβ + terminal lA Pass C
  → Adam / clamp / scheduler
  → 10 evaluations / 9 mutations
```

关键变化：

- 新增 compact α/β coefficient TIR ABI：4 个 ReLU coefficient、4 个 ReLU bias、2 个 residual stage2；
- 新 compact 模板全部位于 S4 独立模块；旧 R3 dense 模板逐字不改，历史 artifact 的源码哈希链保持有效；
- selector、value VM、gradient emitter 增加 generation re-arm，复用原 storage、DLPack view 和 compiled module；
- 新 evaluator 将 Pass A/B/C 固定到同一非默认 CUDA stream；
- 修复新 compact kernel 未显式继承 PyTorch stream 导致的跨进程轨迹不确定性；修复后两个 fresh 进程的 terminal lower 逐位一致；
- 新 optimizer driver 在同一 stream 上执行六 α + 一 β 的 10/9 Adam、投影和 scheduler；
- terminal evaluation 发放六份 lA lease，不产生第 10 次参数 mutation。

## 2. Correctness

独立 dense PyTorch/autograd 与 compact candidate 逐轮比较 10 次 lower、7 份 gradient，并在前 9 轮比较
mutation 后参数：

- terminal lower 最大绝对误差：`2.1457672119140625e-06`；
- 六组 compact α 最大绝对误差：`9.7751617431640625e-06`；
- active β 最大绝对误差：`1.7136335372924805e-06`；
- lower、gradient、parameter 符号一致；
- 计数：10 evaluations、9 Adam mutations、10 scheduler calls、10 value graph submissions、180 compact coefficient launches、0 fallback；
- 新专项：`2 passed`；
- S4/R3 联合：`202 passed`。
- 全量回归：`2095 passed, 3 skipped`；三项 skip 均为既有环境边界；
- 9 个本轮 Python 文件：mypy clean、pylint `10.00/10`、black clean；
- 5 个旧 R3 formal artifact replay 均通过，证明独立 S4 模板没有破坏历史源码哈希链。

## 3. 本地五对计时观察

复现入口：

```bash
conda activate boundflow
python scripts/run_asplos27_s4_all_state_timing.py \
  --pairs 5 \
  --output artifacts/asplos27-s4-all-state-timing/local-v2/summary.json
```

固定 scope 是 prepare 完成后的单个 10-evaluation/9-mutation region wrapper；control 为完整 dense
PyTorch/autograd 六 α + 一 β，candidate 为 S4 compact evaluator。两条路径各丢弃一次 first-use warmup，5 对按
`native-candidate / candidate-native` 交替。

结果：

- native 中位：`107.18515014648438 ms`；
- candidate 中位：`35.22560119628906 ms`；
- paired speedup 几何平均：`3.0505977129204114x`；
- 最差 pair：`3.0099765022983114x`；
- 五对 lower 最大误差：`2.1457672119140625e-06`；
- 五对 parameter 最大误差：`9.7751617431640625e-06`。

prepare 后执行期显存增量峰值：

- allocated：native `20,454,400 B`，candidate `294,912 B`，约 `69.36x` 更小；
- reserved：native `25,165,824 B`，candidate `2,097,152 B`，`12x` 更小。

artifact SHA256：`41ff10a368769e7979e4cb4d88dfc675eb71ac714408baeae99fd5341f378fcb`。

## 4. 口径边界

这些数字是本地、可复现的工程观察，不是 formal performance claim：

- 不含 compile 和 prepare；
- 是 ResNet2B 固定 production region，不是 complete query 或 queue；
- CUDA event 覆盖 wrapper 的 GPU stream 时间和提交间隙，但没有冻结独立 raw/replay/tamper artifact；
- 旧 D2B 只优化 P-anchor，和本轮 all-state scope 不同，不能把旧 `P/D2B` 数字直接并列成同语义三方结论；
- 尚未通过 RVIR exact-call 接回 αβ-CROWN host solver；
- 尚未证明 held-out model family、solve/TTV 或 ASPLOS 最终门槛。

因此当前可以说：**all-state compiled region 已从 single-evaluation correctness 推进到 10/9 轨迹，并在本机
观察到约 3.05x region-wrapper 加速和显著执行期内存下降**。不能说 complete query 已快 3.05x，更不能说
项目总体达到 10x。

## 5. 下一步

1. 把 evaluator 通过 RVIR exact-call adapter 接回同一个 αβ-CROWN host solver；
2. 做 B0 / 旧 D2B / S4 三方同 scope wrapper 与 complete-query 归因；
3. 若 query 传播成立，再冻结 fresh-process raw artifact、replay 和外审交接；
4. 若 region 收益被 adapter/host 吞掉，优先压缩 Python action dispatch 与 10 次提交，而不是继续增加局部 kernel。
