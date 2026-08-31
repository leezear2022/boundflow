# Root CROWN 残差块完整 VJP 生产捕获修改记录

status: captured-ready-for-tir-parameterization
date: 2026-09-01
external-audit: not-requested
performance-claimed: false

## 1. 目的

末端 `/48 ReLU → /input-28 Linear` 虽然 isolated 约快 `1.889x`，但 fresh same-solver query
只有 `1.007267x`。因此本轮不继续微调末端 Linear，而是向前扩大真实 compiled region，目标拓扑为：

```text
/45 ReLU
  → /44 Add
    ├─ /43 Conv → /input-24 ReLU → /input-20 Conv
    └─ skip
  → /input-16 boundary
```

这就是已有 R3 Residual11 两阶段 TIR 的生产语义来源。但旧实现冻结为 `6 domains × 1 spec`，
当前 root optimizer 是 `1 domain × 3 specs`，且真实 optimizer 需要完整 bounds VJP。开工前必须先捕获
当前事务，不能直接套用旧 ABI。

## 2. 新增内容

- `boundflow/runtime/root_crown_residual_capture.py`
  - 只观察一次真实 iteration=5 root optimizer；
  - 同时 patch 六个 production `bound_backward` 节点；
  - 捕获 5 forward / 4 backward 的完整事务；
  - backward 后立即复制全部 VJP，避免 optimizer 下一步覆盖 `.grad`；
  - 退出时恢复所有方法并核对 device/stream。
- `scripts/probe_root_crown_residual_capture.py`
  - 使用同一个 ResNet2B/property 0、seed、max-iteration 和 α/β 步数；
  - 保存本地 tensor capture 与 value-free receipt。
- `scripts/package_root_crown_residual_capture.py`
  - 验证五次 evaluation、22 个 tensor 字段、两套三轴 sparse coordinates；
  - 校验 finite、interval、α range、末次无 backward；
  - 绑定代码和模型/property hash；
  - 支持从 artifact 重新推导 summary。
- `tests/test_root_crown_residual_capture_artifact.py`
  - 正常 replay；
  - receipt 篡改拒绝。

## 3. 真实捕获结果

生产运行一次通过：solver status=`verified`，六个节点各出现 5 次，autograd backward 4 次，
无 partial transaction，device/stream 未漂移。

| 边界/状态 | 真实形状 |
|---|---|
| region incoming A | `[3,1,16,8,8]` |
| `/45` lower/upper | `[1,16,8,8]` |
| `/45` compressed α | `[2,3,1,178]` |
| `/43` weight/bias | `[16,16,3,3]` / `[16]` |
| `/input-24` lower/upper | `[1,16,8,8]` |
| `/input-24` compressed α | `[2,3,1,86]` |
| `/input-20` weight/bias | `[16,16,3,3]` / `[16]` |
| region output A/bias | `[3,1,16,8,8]` / `[3,1]` |

两个 sparse α 都不是一维 flat index：它们各由 `(channel,height,width)` 三个坐标张量表达，长度分别
为 `178` 和 `86`。打包器因此按三轴范围与坐标唯一性验证，不能错误地要求每个轴单独递增。

## 4. 完整 VJP 边界

前四个 evaluation 都捕获了以下七路梯度：

1. region incoming A；
2. `/45` lower；
3. `/45` upper；
4. `/45` compressed α；
5. `/input-24` lower；
6. `/input-24` upper；
7. `/input-24` compressed α。

同时捕获 region output A/bias adjoint 作为 custom backward 输入。第五次 terminal evaluation 按生产
轨迹没有 backward，artifact 强制这七路 gradient 均为 `None`。

这修正了旧路线的两个错误假设：

- `spec` 和 `domain` 不能互换或拍平成同一个“batch”轴；
- 只返回 dα 不足以保持 root optimizer 轨迹，lower/upper VJP 必须由新 TIR owner 返回。

## 5. 复用与重新设计边界

可以直接复用：

- Residual11 的两次 ConvTranspose 索引和 skip-add 数学；
- staged/tile reduction、caller-owned arena、current-stream DLPack；
- compressed α 的 feature-map lowering；
- 已验证的 residual producer→ReLU 局部消费思路。

必须重新参数化：

- 所有 TIR loop 显式保留 `spec_count=3` 与 `domain_count=1`；
- 两层 α lookup 使用三轴 coordinates，而不是写死 flat map；
- custom backward 同时生成 7 路 VJP；
- forward 不发布两层 ReLU 后 dense A；backward 允许 bounded transient adjoint/recompute，但不能跨层
  保存 dense A；
- live bridge 必须在 `/44 Add` 一次性提交 candidate 的 combined main+skip 输出，避免继续执行 native
  main branch。

## 6. 当前验证与边界

- production GPU capture：PASS；
- artifact replay：PASS；
- artifact tests：`2 passed`；
- mypy：4 个本轮文件 clean；
- pylint：`10.00/10`；
- artifact 无 `/home`、`/tmp` 路径；
- 本轮不计时、不形成性能 claim、不发起外部审计。

artifact：`artifacts/root-crown-residual-capture/resnet2b-prop0-v1`。

## 7. 下一工程动作

唯一下一动作是实现 `RootResidualTemplateV1` 与 shape-parameterized TIR：

```text
3×1 incoming A
  → /45 slope/intercept
  → /43 ConvTranspose
  → /input-24 slope/intercept
  → /input-20 ConvTranspose
  + skip
  → combined output A/bias
```

随后实现七路 custom VJP，并先对本 artifact 的 5 evaluation 做 native parity。只有该完整 residual region
isolated 明显胜出，才接 live `/44 Add` 替换并重跑 same-solver；不再复活旧的三 site per-op bridge。
