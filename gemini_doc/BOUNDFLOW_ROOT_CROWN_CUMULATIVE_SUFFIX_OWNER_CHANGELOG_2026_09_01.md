# Root CROWN terminal + residual 累计 suffix owner 实现与性能记录

date: 2026-09-01
status: implemented-and-measured
external-audit: not-requested
performance-claimed: false

## 1. 本轮目标

上一阶段已经分别证明：

- terminal `ReLU → Linear` TIR 的局部 forward/full-VJP 正确；
- 紧邻的两卷积 residual TIR 的局部 forward/full-VJP 正确；
- 单独替换 residual 时局部约 `2.10x`，但 complete query 约 `0.998x`，收益没有传播。

本轮不做外审，也不再继续孤立优化单个算子，而是把两个相邻区域收束为同一个累计
custom-autograd owner：

```text
/48 ReLU → /input-28 Linear → /46 Flatten(view)
  → /45 ReLU → /44 Add → /43 Conv → /input-24 ReLU → /input-20 Conv
```

目标是让 terminal 输出不再形成独立 autograd owner，并把 residual VJP 产生的 terminal-output
adjoint 直接交给 terminal VJP。

## 2. 实现内容

### 2.1 Prepared Runtime

`boundflow/runtime/root_crown_suffix_tir.py` 在 cumulative executor 内新增 terminal warmup adapter：

- 预先 materialize CUDA module；
- 建立 output、compressed-alpha-gradient 和 bound-gradient 持久 arena；
- 清除 synthetic source 的 DLPack view，只保留 persistent view；
- warm query 前把 launch/pointer counter 归零。

已有 `root_crown_terminal_tir.py` 被历史 replay artifact 绑定源码 hash，因此本轮不修改其公共 ABI 或文件
内容；warmup adapter 只在新 suffix owner 内调用既有 forward/backward 并管理其 persistent arena。第一次
全量回归曾准确检出该 hash 边界，移出改动后旧 artifact replay 恢复通过。

### 2.2 累计 owner

新增 `boundflow/runtime/root_crown_suffix_tir.py`：

- 对 terminal/residual template 做 spec、domain、flattened feature、GPU capability 的 fail-closed
  边界检查；
- terminal forward 直接写 persistent arena，不创建中间 autograd node；
- host 的 `/46 Flatten` 只产生同一 storage 的 view，并以 data pointer 绑定；
- residual Add 处创建唯一 cumulative custom-autograd owner；
- forward 返回 residual output A 与 `terminal_bias + residual_bias`；
- backward 顺序固定为：

```text
residual full VJP
  → d(terminal output A)
  → zero-copy reshape
  → terminal full VJP
  → terminal lower/upper/α + residual entry/inner lower/upper/α
```

最终对 optimizer 暴露九路梯度，不把 terminal/residual 边界的 dense A 保存为外部 autograd history。

### 2.3 真实 solver 接入

新增 `boundflow/runtime/root_crown_suffix_live.py`：

- 只替换 `/49` root start-node 的一次五 evaluation optimizer transaction；
- 其他 start-node 继续走原 auto_LiRPA；
- terminal Linear 返回 persistent A 和零 bias，避免 host 重复累计 terminal bias；
- residual Add 一次性返回完整 suffix A/bias；
- 主 residual 分支在 host traversal 中被旁路；
- transaction 完成后再回填 `/45`、`/input-24` 的 `lA/d` 状态，避免重新激活已旁路队列；
- 必须满足 5 stage、5 consume、5+5 forward launch、4+4 backward launch、0 fallback。

新增：

- `scripts/run_root_crown_suffix_live_worker.py`：fresh same-solver control/candidate worker；
- `scripts/probe_root_crown_suffix_tir.py`：组合 PyTorch 闭式 oracle、九路梯度和隔离计时；
- `tests/test_root_crown_suffix_tir.py`：zero-copy template boundary 正/负合同。

## 3. 当前所有权边界

本轮实现的是：

```text
一个 custom-autograd owner
  + 两个 Prepared TVM module
  + terminal/residual 之间一个 persistent zero-copy view
```

它不是：

- 一个 CUDA kernel；
- 一个合并后的 TVM PrimFunc；
- 已完成跨 module producer-consumer schedule fusion。

因此本轮证明的是“跨 BoundOp 的累计梯度所有权可以降低 solver wrapper 成本”，而不是已经完成最终
codegen fusion。receipt 显式分别绑定 terminal/residual scheduled-TIR 和 device-source hash，不把两个 module
伪写成一个。

## 4. 独立组合 oracle

输入是同一 deterministic production transaction 的两份 capture。五个 evaluation 的 terminal 输出与
residual 输入 boundary 最大差为 `2.9802322387695312e-08`。

组合 oracle 独立执行：

1. PyTorch terminal ReLU bound transform + matmul/bias；
2. zero-copy reshape；
3. PyTorch residual 两个 ReLU bound transform、两个 conv-transpose、skip 和四段 bias；
4. 对最终 A/bias 做 autograd，输出九路梯度。

100-repeat probe 结果：

| 项目 | 结果 |
|---|---:|
| evaluation | `5` |
| backward evaluation | `4` |
| forward/VJP 全局最大误差 | `3.5762786865234375e-07` |
| forward/VJP sign | 全部 exact |
| candidate 中位 | `0.663968 ms` |
| PyTorch 闭式 oracle 中位 | `1.827360 ms` |
| local native/candidate | `2.752181x` |
| fallback | `0` |

该局部数字排除了 compile/prepare，但包含两段 prepared executor、唯一 custom backward 以及九路 VJP；
它不是 complete-query claim。

## 5. Same-solver 结果

协议保持同一个 αβ-CROWN host solver、模型/property、seed、branch、termination、timeout 和 optimizer
配置；candidate 的 compile/prepare 在 query 计时前完成。三对 fresh 采用交替顺序：

```text
pair 0: control → candidate
pair 1: candidate → control
pair 2: control → candidate
```

control/candidate 比值：

| pair | complete query | root incomplete | optimizer transaction | autograd backward |
|---:|---:|---:|---:|---:|
| 0 | `1.025907x` | `1.084295x` | `1.132071x` | `1.244472x` |
| 1 | `1.078329x` | `1.123990x` | `1.170069x` | `1.292163x` |
| 2 | `1.069936x` | `1.140965x` | `1.209559x` | `1.271660x` |
| geomean | **`1.057805x`** | **`1.116162x`** | **`1.170139x`** | **`1.269281x`** |

补充：

- complete-query GPU event geomean：`1.057797x`；
- 最差 complete-query pair：`1.025907x`；
- 三对 lower 最大差：`1.1920928955078125e-06`；
- status、depth、final decision、split、queue、visited domains、upper mask 全部一致；
- 每进程 receipt 为 5 cumulative owner、0 intermediate owner、0 fallback。

在三对冻结测量前有一组接线 smoke 得到 query `0.951556x`；它用于发现单样本波动，不纳入上述三对
交替结果，也没有从记录中删除。三对测量全部高于 1，说明当前累计收益稳定，但样本数仍不足以升级正式
论文性能 claim。

## 6. 结论

相对上轮单 residual 的 query `0.997953x`，本轮累计 terminal+residual owner 达到 `1.057805x`。
这是当前路线第一次同时出现：

1. 局部组合 forward/full-VJP 明显加速；
2. same-solver optimizer transaction 明显加速；
3. 收益传播到 complete query；
4. 最差 fresh pair 仍大于 1。

本阶段可记为：

```text
MECHANISM-CORRECT / QUERY-POSITIVE-PILOT / PERFORMANCE-CLAIM-CLOSED
```

## 7. 下一步

下一主动作继续扩大累计 suffix，而不是外审：捕获并编译 `/input-16 → /39` 的前一 residual block。
该块包含 main 与 projection-skip 两支，需要新的 multi-branch state/VJP owner。执行顺序：

1. 只读 capture 前一 residual 的真实 topology、α 坐标、bounds、weights、bias 和 output VJP；
2. 独立 PyTorch 闭式公式确认 branch merge 和 projection skip；
3. 扩为 terminal + residual-2 + residual-1 的 cumulative owner；
4. 先复用两个 Prepared module 边界验证收益传播；
5. 再按 profile 决定是否把相邻 module 合并为一个 TVM PrimFunc/schedule；
6. 重跑 same-solver 三对，目标是继续提高 complete-query，而不是只提高局部 kernel 数字。

外审仍不自动启动；只有用户明确要求或准备升级正式论文 claim 时才生成 exchange。

## 8. 工程验证

- suffix/terminal/residual targeted：`37 passed`；
- final full suite：`2147 passed, 3 skipped`；
- touched-file mypy：clean；
- touched-file pylint：`10.00/10`；
- `git diff --check`：PASS；
- frozen terminal five-pair replay：PASS。

第一次全量为 `2146 passed, 3 skipped, 1 failed`；唯一失败是旧 terminal artifact 检测到其绑定源码
hash 被 warmup 方法改变。warmup 移到新 suffix executor、旧文件恢复逐字节一致后，定向 replay 与第二次
全量均通过。该失败属于防篡改门禁正常工作，不被省略。
