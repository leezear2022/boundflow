---
status: implementation-started
date: 2026-08-28
type: changelog
topic: boundflow
slug: asplos27-s2-coarse-crown-custom-vjp
external-audit: deferred-by-user
performance-claimed: false
---

# ASPLOS’27 S2 coarse CROWN + custom VJP changelog

## 2026-08-28 开工

- 用户要求继续实现，下一轮再统一外审；
- S1 external-audit exchange保留为已交付历史边界，不把后续S2写入该round；
- 只读单evaluation归因确认native约`9.002 ms`、旧D2B约`6.483 ms`；
- 旧D2B内部effective-value约`3.693 ms`为最大瓶颈，forward约`1.745 ms`、coefficient/sign约
  `0.869 ms`；
- S2第一刀改为standard Relax + TVM cuDNN重建selected-value Conv chain，不包裹旧serial
  `effective_pre23`；
- 本文后续连续记录实现、失败、correctness、formal与closure；`performance_claimed=false`。

## 2026-08-28 cuDNN build/runtime准入

- `env.sh`按当前解释器的purelib精确发现PyTorch wheel携带的cuDNN，导出
  `BOUNDFLOW_CUDNN_ROOT`并把其`lib`加入`LD_LIBRARY_PATH`；不再用可能误选Python版本的glob；
- `install_dev.sh`优先从真实`nvcc`解析CUDA root，向TVM CMake显式传递
  `USE_CUDNN`、`CUDA_CUDNN_LIBRARY`，并只在`import tvm_ffi`失败时重建tvm-ffi；
- 增加TVM cuDNN runtime global-function smoke；本机增量TVM build通过，
  `relax.ext.cudnn`与`cudnn.conv2d.forward`可见，`libcudnn.so.9`可解析；
- 最初一次TVM core/cuDNN已经成功后，冗余editable tvm-ffi build失败；该问题通过条件化
  tvm-ffi步骤关闭，不能误记为TVM/cuDNN core build失败。

## 2026-08-28 S2-A selected-value compiler

- 新增一个standard Relax function，按真实依赖表达input select、ReLU17/19/23 select以及Conv0/2/4/
  shortcut5/8；不再逐输出重算前层Conv；
- 四个verification-specific select进入scheduled TIR；五个Conv call全部进入TVM cuDNN codegen；
  Conv4与Conv8因签名相同共享一个partition function，所以证据分别记录`4 functions / 5 calls`；
- source Relax、partitioned Relax、RunCodegen后IR、device sources、target与compile time进入compiled
  identity；cuDNN不是五个call时直接拒绝；
- 本机首次编译约`0.33 s`，独立对比旧D2B `pre25`最大差`1.90735e-6`且sign exact；该数是实现
  correctness诊断，不是formal性能claim。

## 2026-08-28 S2-B prepared direct custom VJP

- `PreparedS2CrownProgramV1`复用R3-D2B的plan、lifetime trace、two-slot arena、active-β coefficient
  wavefront、recompute-A26和compressed-gradient kernel，只替换错误的effective-value owner；
- 28个输入/参数/state view在prepare阶段一次DLPack绑定；cuDNN/Relax chain预热后由一个CUDA Graph
  submission执行，输出直接成为`pre25`的graph-stable view；
- `run_vjp(dynamic_alpha, upstream)`是直接custom VJP API；不创建`autograd.Function`上下文，不保存
  dense A或autograd history；外部dynamic α/upstream在准入后复制进固定owner；
- execution receipt绑定plan、trace、B1/B2/D1C、三层Relax identity、device sources、4 partition
  functions/5 Conv calls、4 selected TIR、arena、active β与零fallback；
- 与独立native PyTorch比较，lower最大差`3.09944e-6`、compressed dα最大差`4.37722e-8`、sign
  exact；短测native/D2B/S2中位约`8.920/6.332/2.122 ms`，即S2约`4.20x`，仅作为formal前
  feasibility，不形成性能claim。

## 2026-08-28 correctness与fail-closed测试

- 增加native PyTorch、旧D2B direct、S2 canonical三方lower/dα对照，reference不经过S2 compiler；
- 检查5个cuDNN call、单graph replay、active β、零saved dense A/history、零warm DLPack与零fallback；
- receipt对claim、call count、replay count、β owner、dense-A、fallback和hash篡改全部拒绝；
- default stream与immutable state version漂移均在selected graph launch前拒绝；
- 专属测试`5 passed`；formal artifact与全量回归尚未在此节点运行。

## 2026-08-28 wrapper成本回收

- 第一版host-boundary短测只有约`3.41x`，低于冻结的worst `3.50x`；归因显示旧B1 forward的17次
  launch仍在每次调用走Python dispatch，约`1.76 ms`；
- 在不改变数学、arena或kernel的前提下，把已固定pointer/schedule的整个forward wavefront捕获为第二个
  CUDA Graph；运行时显式记录一次forward replay以及其17个logical launch、D1C 4-stage ownership；
- 修改后三方worker预跑N/D/P约`8.751/4.835/1.921 ms`，P/N约`4.56x`；仍仅为formal前
  feasibility；六fresh协议继续执行。

## 2026-08-28 formal runner与replay实现

- fresh worker固定`NDP/NPD/DNP/DPN/PND/PDN`六全排列，每个进程5 warmup groups、30 measured
  groups，三方使用独立tensor owner、同一non-default stream及调用前后device boundary sync；
- raw JSONL保存全部540个latency样本、三方lower/dα数值、receipt、cold prepare、warm peak与环境；
- artifact replay只从raw重算correctness、sign、geomean/worst、memory和所有门禁，并核对git blob、source/
  model、plan/trace、Relax/device identities及manifest；
- tamper probe定义10类攻击；receipt攻击会重签inner receipt，latency攻击会重算median/summary，所有case再重签
  outer manifest，避免只测SHA256表层。
