# 变更记录：固化 RVIR-2 fail-closed 负向测试

> 日期：2026-08-03
> 分支：`test/rvir2-fail-closed-hardening`
> 来源：RVIR 外部审计 minor F1

## 变更内容

为 `tests/test_real_verifier_ir_integration.py` 增加四条专用负向回归测试：

- 将 external task backend implementation 篡改为未声明的 local fused backend；
- 从 external schedule 中删除唯一 emit；
- 为同一 external task/region 添加第二次 launch；
- 将 Bound IR 的 `semantics_owner` 从 `external_verifier` 改为 `boundflow`。

四条路径都必须在调用 provider `exact_call` 前 fail closed。实现、artifact 和既有
correctness claim 均不改变。

## 验证结果

- `pytest -q tests/test_real_verifier_ir_integration.py`：`6 passed`；
- `pytest -q tests`：`456 passed, 37 skipped`；
- Black：目标测试文件 unchanged；
- Pylint：目标测试文件 `10.00/10`；
- `git diff --check`：PASS；
- `dol lint --soft`：PASS。
