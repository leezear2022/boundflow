# FSG4/B3-A Prepared Core 实现候选变更记录

日期：2026-08-14
状态：历史实现候选；已由`change_2026-08-14_fsg4_b3a_prepared_core_closure.md`取代

## 起因与目标

B3-0正式artifact确认B2在一次真实core内执行一次module binding move、两次完整scope构造，且template
只有compile没有core hit。B3-A的单变量目标是把静态graph/device/dtype/shape/policy-structure/mutable
inventory冻结为可复用模板，把动态query/state绑定成独立实例；不混入terminal-only schedule、atomic
commit、TIR/JIT或runtime优化。

## 本轮修改

- 新增`boundflow/runtime/fsg4_b3_prepared_core.py`：
  - `PreparedCoreTemplateV1`冻结primal graph和参数内容、topology、placement、shape bucket、policy契约、
    12条mutable path及binding inventory；
  - `PreparedCoreTemplateCache`提供exact insert/resolve、compile/hit cardinality；
  - `CorePlanInstanceV1`绑定snapshot、mapping、input/objective、intermediate bounds、split、α/β及mutation
    policy，构造一次native scope并生成完整receipt；
  - topology/device/dtype/mutable inventory/module parameter/state drift全部fail closed。
- `execute_rvir_v4_native_optimizer_trace`新增typed prevalidated plan入口；默认`None`仍重建scope，显式
  B3-A receipt必须与initial state精确一致后才复用scope。
- `_LiveExecutor`新增opt-in cache/hash pair；prepared路径从模板取得program/module，不再在core移动bindings；
  默认B2路径不变。
- 显式counter schema加入B3-A冻结值，diagnostic runner加入`--configuration B3-A`和query/core外模板准备；
  B3-A只允许module move、scope、template hit三项发生预注册变化。

## 验证

- targeted：`31 passed in 4.58s`；
- Black：touched files clean；
- mypy：三个touched runtime source在`--follow-imports=skip`下clean；
- Pylint：touched runtime/script/tests `10.00/10`；
- `git diff --check`：PASS。

## 限制与下一步

当前尚未从提交后的source生成fresh GPU artifact，因此不能把预期的module move `1→0`、scope `2→1`、
template hit `0→1`写成实测事实，也没有性能主张。下一步先提交本实现候选，再运行同一ResNet2B/property、
同一solver/protocol的B3-A fresh GPU counter/correctness artifact；失败必须保留证据并停在B3-A，不得开启
B3-B。
