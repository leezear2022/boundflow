# 2026-08-16：FSG4/B4 cumulative fusion 预注册

## 结论

Round 2外审已批准B3并由executor关闭exchange。本轮据此只启动B4预注册，没有实现或运行TIR，
没有新增performance claim。B4以B3为直接基线，B0为累计对照；B5—B7继续关闭。

## 证据驱动的路线修正

B3 raw显示whole core只占query约17.736%，optimizer只占query约7.933%。optimizer-only即使无限快，
query上限也只有1.0862x，低于追回B0所需的1.0989x。production中固定存在10次optimizer、1次
terminal export、3次KFSB child，共14次lower-only CROWN backward；三段合计占query约12.010%，
需要约3.9897x持续加速才可单独追回B0 parity。

因此B4不再定义为“接一个旧fused Conv kernel”，而是：先做raw kernel/materialization归因；再消除
terminal export重复CROWN；随后建立具有显式custom backward的lower-only α/β CUDA/TIR region，并覆盖
optimizer、terminal与KFSB全部14次调用。所有增量均相对B3测量，同时保留B0累计结果。

## 文档

- 计划：`gemini_doc/BOUNDFLOW_FSG4_B4_CUMULATIVE_CUDA_TIR_FUSION_PLAN_2026_08_16.md`；
- changelog：
  `gemini_doc/BOUNDFLOW_FSG4_B4_CUMULATIVE_CUDA_TIR_FUSION_CHANGELOG_2026_08_16.md`。

## 当前状态

`PREREGISTERED-NOT-IMPLEMENTED`。下一唯一动作是B4-0 read-only profiler schema与fresh attribution
artifact；B4-0关闭前不得实现TIR或启动B4-A/B/C/D。
