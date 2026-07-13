# Backend Candidate Schema v1.0

> Schema：`boundflow.backend_candidate/v1.0`
> Profile schema：`boundflow.backend_profile/v2.0`

## 目的

PR-11 的 action 只表达 dense/structured/reduce-batch。PR-12 起将“系数表示”和“执行后端”
拆成两维，避免把 fused 后端收益回填成 PR-11 Planner 收益。

`ExecutionCandidate` 的稳定字段为：

```text
placement
backend
domain_batch_size
spec_batch_size
materialization_points
capability_id
schedule_id
reason
schema_version
```

placement 当前为 `dense | structured`；backend 当前保留 `pytorch_eager`、`torch_compile`、
`tvm_relax_unfused`、`tvm_tir_default`、`tvm_fused_tir`。

## Capability filtering

排序前必须检查 bound method、requires-grad、α、β、split state、operator family、dtype、layout、
device、optimization stage 与 static shape。所有不支持项返回稳定 rejection reason，不允许 silent
fallback 或 silent wrong。

当前注册两个独立 capability id：

```text
capability_id = tvm_fused_tir_linear_plain_crown_fp32_static_v1
method        = plain CROWN
stage         = inference | final_bound
grad/alpha/beta/split = false
operator      = Linear
device/dtype  = CUDA / float32
layout/shape  = contiguous / static

capability_id = tvm_fused_tir_conv_plain_crown_fp32_static_v1
method/stage  = 同上
operator      = Conv2d
layout/shape  = NCHW / static
```

Conv signature 进一步固定 DSCOHW/OIHW/DSCIHW layout、input/output spatial shape、kernel、
stride、padding、dilation、groups、bias presence、compute capability 与 schedule id。当前只接受
1×1/3×3、stride 1/2、padding 0/1、groups=1、dilation=1；其余属性在 lowering 前显式拒绝。

## 版本隔离

- PR-11：`planner_model=pr11-v1-frozen`，只读；
- PR-12：新 candidate/profile schema；
- 新候选的 profile 写入 `artifacts/phase7a-pr12/`，不得覆盖 PR-11 profile；
- schema 字段或 oracle/regret 定义变化时必须升级版本。
