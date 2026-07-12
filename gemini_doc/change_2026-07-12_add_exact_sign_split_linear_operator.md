# 变更记录：增加精确 SignSplitLinearOperator

## 语义

```text
T(A) = A⁺ ⊙ positive_scale + A⁻ ⊙ negative_scale
```

operator 只延迟 sign-split 本身。`matmul_right`、`conv2d_right`、reshape、slice 和 add 只能
包裹/组合该 operator，不能把 `sign(AW)` 错误改写成 `sign(A)` 与 `sign(W)` 的局部规则。

## 修改

- 新增精确 `SignSplitLinearOperator`，支持 flat/NCHW scale、batch scale 与 autograd。
- 支持 LinearOperator contract、composition 和 dense reference materialization。
- 当前 center/norm/contract reduction 使用显式 ephemeral materialization，并写入 trace reason；
  不缓存 autograd graph，不把临时 dense 声称为零 materialization。

## 边界

本批只增加 operator 及单元测试，尚未切换 CROWN ReLU 主路径。下一提交才实现 ephemeral bias
reduction 与 structured coefficient 返回。

## 验证结果

- SignSplit、已有 DAG operator、concretization 与 conv lazy norms：26 passed；
- `py_compile` 与 `git diff --check` 通过；主 CROWN 路径尚未切换。
