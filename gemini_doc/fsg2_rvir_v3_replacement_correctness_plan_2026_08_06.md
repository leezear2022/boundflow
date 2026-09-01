# FSG2 RVIR-v3 Replacement Correctness 计划与变更记录

## 目标

把RVIR-v2的`external_abcrown_exact_call/v1` passthrough与真正replacement分开。v3执行入口在
类型/API层面不接收original callable，只接收owned executable tensor/state payload和明确的
BoundFlow backend。

## 必须实现

- owned input lower/upper、linear spec、intermediate bounds与program tensors；
- initial/alpha/beta/split phase、lower/upper polarity、domain/spec/ragged slices；
- α/β/split mutable state只允许预声明copy-in/copy-out，执行后生成pre/post digest receipt；
- backend identity必须属于BoundFlow，external exact/provider backend fail closed；
- result shape/dtype/device、finite、lower<=upper、requested polarity逐项校验后才能copy-out；
- payload或state创建后被修改、漏state、多余mutation、ragged gap/overlap、fallback/original callback
  均fail closed；
- typed execution receipt显式记录`original_callback_count=0`、replacement dispatch=1。

## 验证矩阵

- phase：initial-CROWN、alpha optimization、beta/split；
- polarity：lower-only、upper-only、both；
- batch：single、dense batch、ragged contiguous slices；
- mutation：read-only、copy-in-only、copy-out、shape/dtype/undeclared mutation rejection；
- backend：正向独立Torch reference replacement；external provider/unknown backend拒绝；
- compatibility：RVIR-v2测试继续通过，但v2不能升级为replacement claim。

## Claim边界

合同/合成reference通过只证明replacement transport与fail-closed语义，不自动证明official
αβ-CROWN production call已被替换。只有真实model/property的same-solver observer证明逐call、
branch/parent/node/verdict门禁后，FSG2才可完整关闭并进入FSG3 B2 timing；否则以明确coverage
`VALIDATED-REDUCED`或`NO-GO`关闭，不伪造全量replacement。

## 当前实现进度

- v3 contract与独立Torch affine backend已覆盖initial/alpha/beta phase、三种polarity、dense/ragged、
  mutation receipt和负向拒绝；
- native plain-CROWN backend复用现有Bound/Plan/Task/Schedule语义执行栈，但v3入口不接收original
  callable；
- fresh frozen ResNet diagnostic：lower max diff=`7.152557373046875e-7`、sign=`9/9`、五层IR
  hash齐全、original/fallback count=`0/0`；
- 新增real initial-CROWN artifact generator/replay；正式artifact须从提交后的clean revision生成；
- α/β/split external state到native state的生产映射仍未准入，当前必须fail closed。

## 生产状态inventory与artifact加固

- 新增`scripts/run_fsg2_abcrown_state_inventory.py`，在冻结的αβ-CROWN、auto_LiRPA、
  VNN-COMP revision上运行ResNet2B property 0、CUDA、`max_iterations=1`，逐个记录真实
  `compute_bounds`的phase、parent/depth、module α/β state、split相关kwargs及tensor
  shape/dtype/device/content digest；property使用临时副本，避免污染benchmark仓库；
- inventory的admission结论由原始call rows确定性派生，不允许“看到alpha phase”自动升级成
  replacement：当前native RVIR-v3 backend只实现initial-CROWN，真实alpha为start-node keyed嵌套
  state，beta/split调用前又没有可直接own的完整module beta state；
- `run_fsg2_rvir_v3_initial_artifact.py replay`新增source commit下逐文件code provenance复核，
  防止artifact在不同实现上仅凭payload digest通过；
- 新增纯逻辑测试，保证真实形态和空inventory都不能意外把B2 timing标为admitted。

正式inventory必须在上述runner提交、code path clean后生成；正式结果再写入本节并据此关闭
FSG2 gate。

首轮正式抓取发现需要进一步区分`intermediate_constr`键存在与其中实际存在tensor leaf：
真实beta-split嵌套调用具有该键，但本workload中其tensor leaf为0；`interm_bounds`则稳定为12个
tensor。runner据此升级为同时记录pre/post beta state、key presence、intermediate bounds与
aux-reference bounds，避免把provider上下文误称为已被RVIR own的split state。

## 最终关闭

- initial-CROWN artifact：lower max diff=`7.152557373046875e-7`、sign=`9/9`、
  original/fallback=`0/0`，summary/manifest=`fd6dbd43…e6c4`/`94f06ef…10a`；
- production inventory v2：24 calls=`12 initial + 1 alpha + 11 beta/split`；alpha pre-state=
  21 tensors；11个beta/split calls的pre/post显式beta state均为0；summary/manifest=
  `37f6dbcd…6544`/`e8548a25…ff06`，semantic replay通过；
- targeted=`18 passed`；全量（`source env.sh`）=`1107 passed, 3 skipped`；Black、mypy与
  Pylint 10.00/10通过；
- FSG2状态：`VALIDATED-REDUCED initial-only`；完整production alpha/beta/split replacement与
  B2 timing=`NO-GO/not admitted`；
- FSG3—FSG5因依赖B2而gated，未执行，也不能被解释为各优化层已分别失败。

完整审计入口：`change_2026-08-06_fsg2_replacement_boundary_and_downstream_gate.md`。
