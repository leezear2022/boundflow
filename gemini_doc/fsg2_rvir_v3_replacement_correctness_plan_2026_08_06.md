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
