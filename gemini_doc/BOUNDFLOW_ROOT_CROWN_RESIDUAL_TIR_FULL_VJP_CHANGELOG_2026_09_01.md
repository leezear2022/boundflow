# Root CROWN residual TIR full-VJP 实现与性能记录

date: 2026-09-01
status: implemented-and-measured
external-audit: not-requested
performance-claimed: false

## 1. 本轮目标

把 ResNet2B root α-CROWN 中的真实生产区域

```text
/45 ReLU
  → /44 Add
    → /43 Conv
      → /input-24 ReLU
        → /input-20 Conv
    → residual skip
  → /input-16 boundary
```

实现为 spec/domain 双轴的 TVM/TIR forward + full VJP，并通过 live bridge 替换同一个
αβ-CROWN exact-call。该区域包含两个 ReLU、两个 ConvTranspose backward-bound、残差合并、四段
bias 累加以及七路梯度输出。

## 2. 实现内容

新增：

- `boundflow/backends/tvm/root_crown_residual.py`
  - `RootCrownResidualTemplateV1`；
  - 独立 `spec_count/domain_count`；
  - 稀疏 α 三维坐标映射；
  - residual forward TIR；
  - incoming A、两组 lower/upper、两组 compressed α 的七路 full VJP；
  - deterministic TIR/module/source hash 与 workspace inventory。
- `boundflow/runtime/root_crown_residual_tir.py`
  - current-stream exact DLPack launch；
  - persistent output/gradient arena；
  - custom autograd owner；
  - CUDA module admission warmup；
  - host-required `main_a` state output。
- `boundflow/runtime/root_crown_residual_live.py`
  - 只替换 `/49` root exact-call；
  - 其他 start-node 保留原 auto_LiRPA 路径；
  - Add 处将主分支旁路、把 compiled region 结果送入 skip boundary；
  - solver 事务结束后才回填最终 `lA/d`，避免污染 backward queue。
- `scripts/probe_root_crown_residual_tir.py`
  - 5 个生产 evaluation 的独立 PyTorch 闭合 oracle；
  - 7 路 VJP、符号、DLPack 和隔离 timing。
- `scripts/run_root_crown_residual_live_worker.py`
  - fresh control/candidate same-solver worker；
  - compile/warmup 排除在 query timing 外。
- `tests/test_root_crown_residual_tir.py`
  - template、非法 ABI 与 runtime fail-closed 单测。

## 3. 数学与状态边界

前向为：

```text
A1 = relu_bound(A0, L0, U0, α0)
A2 = conv_transpose(A1, W0)
A3 = relu_bound(A2, L1, U1, α1)
Y  = A1 + conv_transpose(A3, W1)
```

bias 同时累加两个 ReLU intercept 和两个 Conv bias。反向不保存 autograd dense history，而在 TIR
中重算必要的 `A1/A2`，输出：

```text
dA0,
dL0, dU0, dα0,
dL1, dU1, dα1
```

`main_a=A2` 是 αβ-CROWN 分支状态需要的输出。当前 schedule 原本就物化该 tensor；把它变成 caller-owned
state output 没有新增同形中间计算。

一个重要诊断：capture artifact 中四个 lower/upper `.grad` 还包含 selected region 之外对共享 bound tensor
的使用，因此不能直接作为局部 VJP oracle。独立闭合 oracle 与 TIR 的局部 VJP一致；incoming A 和两组 α
只由该局部路径拥有，仍与 capture gradient 直接一致。最终 live solver 的总梯度与轨迹由 same-solver
lower/branch/queue 共同验证。

## 4. Correctness 结果

生产 capture 的 5 forward / 4 backward：

- forward A 最大误差：`2.9802322387695312e-08`；
- forward bias 最大误差：`7.152557373046875e-07`；
- 七路 VJP 对独立 oracle 的全局最大误差：`9.5367431640625e-07`；
- forward 与全部 VJP 符号一致；
- DLPack pointer：全部 exact；
- fallback：`0`。

三组 fresh same-solver control/candidate：

- lower 最大差：`1.6689300537109375e-06`；
- final decision、depth、queue、split、visited domains、upper mask 全部一致；
- candidate 每进程恰为 5 forward / 4 backward，主路径旁路调用为 0。

## 5. 性能结果

### 5.1 隔离区域（50 repeats）

| 实现 | 中位耗时 |
|---|---:|
| PyTorch 独立 oracle | `1.210880 ms` |
| residual TIR forward + full VJP | `0.575472 ms` |
| native / candidate | `2.104151x` |

这是局部机制测量，不是 query claim。

### 5.2 首次 live 接入的 warmup 诊断

未做 Prepared Runtime warmup 时，19 个 CUDA kernel 的首次 module materialization 落入真实 autograd：

- query：`0.74865x`；
- root：`0.64505x`；
- autograd backward：control `245.75 ms`，candidate `429.95 ms`。

加入 admission warmup 并同步完成后，该一次性成本移出 query，未隐藏到异步尾部。

### 5.3 三组 fresh same-solver（warm）

| pair | query | root | optimizer transaction | autograd backward |
|---:|---:|---:|---:|---:|
| 0 | `0.997674x` | `1.000445x` | `1.000931x` | `1.034134x` |
| 1 | `1.008325x` | `1.012793x` | `1.011850x` | `1.029657x` |
| 2 | `0.987963x` | `0.998666x` | `1.000893x` | `0.985297x` |
| geomean | **`0.997953x`** | **`1.003948x`** | **`1.004545x`** | **`1.016121x`** |

结论：局部 TIR 快约 2.1 倍，但该 region 在完整 query 中占比太小，complete query 仍为持平，不能形成
performance claim。

## 6. 当前限制

1. correctness schedule 仍有 12 个 dense workspace buffer；尚未执行 tile-local streaming、shared memory、
   tensorization 或 schedule autotuning。
2. forward/backward 分别包含多个内部 kernel；Prepared Runtime 只合并 host submission ownership，不把
   “一次调用”伪称为单 kernel。
3. 当前 template 固定当前生产几何的 3×3、padding=1、stride=1；schema 保留 spec/domain 双轴，但还未
   扩到其他 shape family。
4. complete-query geomean `0.997953x` 明确不是加速。

## 7. 决策与下一步

本阶段关闭为 `MECHANISM-CORRECT / QUERY-NEUTRAL`：正确性与 live ownership 成立，性能 claim 关闭。

下一主动作不是外审，也不是继续微调这个小区域，而是建立更大的 cumulative suffix owner：优先把
terminal Linear/ReLU 与本 residual 合并为一个 custom-autograd submission，随后扩到 `/input-16 → /39`
的前一残差块。每次扩大都沿用：

1. production capture；
2. 局部 closed oracle/full VJP；
3. Prepared Runtime warmup；
4. same-solver trajectory；
5. 只有 query 收益稳定传播后才升级性能结论。

## 8. 工程验证

- targeted：`19 passed`；
- full suite：`2141 passed, 4 skipped`；
- touched-file mypy（`--follow-imports=skip`）：clean；
- touched-file pylint：`10.00/10`；
- `git diff --check`：PASS。

仓库级 mypy 仍包含其他历史文件的既有错误，本轮没有据此宣称 repository-wide mypy clean。
