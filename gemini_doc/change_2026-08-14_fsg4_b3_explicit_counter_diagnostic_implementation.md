# FSG4/B3-0 显式 Counter 诊断实现记录

日期：2026-08-14
状态：`IMPLEMENTED-RERUN-PENDING`
分支：`feat/rvir-v4-production-state-ownership-v1`

## 目标

实现不依赖`sys.setprofile`或通用profiler的B2结构计数器，在真实same-solver B2 control worker中，逐事件
统计B3预注册要求消除或保留的执行动作。本文只记录实现与静态验证，不形成真实counter或性能结论。

## 改动

- 新增`boundflow/runtime/fsg4_b3_explicit_counters.py`：
  - 固定26项counter inventory；
  - 保存带ordinal/detail的显式event journal；
  - 从journal独立聚合counter；
  - 对B2的compile/module move/scope/10/9 optimizer/5 forward/KFSB/12-path atomic结构执行
    fail-closed验证；
  - snapshot canonical hash绑定semantic hash、raw worker SHA256、provider/fallback与环境准入。
- 新增`scripts/run_fsg4_b3_counter_diagnostic.py`：
  - `run`只允许B2 control，复用FSG3真实worker与环境门禁；
  - 只patch明确命名的函数/方法seam，不安装Python/CUDA profiler；
  - 计数template compile、module binding move、scope、optimizer evaluation/update/snapshot、forward
    trace、KFSB child batch、candidate D2H、typed validate/hash和atomic commit；
  - 原始worker、event journal、report、code revision与manifest逐层hash绑定；
  - `replay`从event journal重新聚合counter，并复验FSG3语义/环境/provider/fallback；
  - 使用同目录临时staging，只有完整replay通过后才原子发布artifact。
- 新增`tests/test_fsg4_b3_explicit_counters.py`，覆盖journal重算、snapshot hash、五类固定counter
  tamper、布尔伪装拒绝以及instrumentation完整恢复。

## 已完成验证

```text
pytest -q tests/test_fsg4_b3_explicit_counters.py tests/test_fsg3_same_solver_worker.py
17 passed

mypy boundflow/runtime/fsg4_b3_explicit_counters.py \
  scripts/run_fsg4_b3_counter_diagnostic.py
Success: no issues found in 2 source files

pylint boundflow/runtime/fsg4_b3_explicit_counters.py \
  scripts/run_fsg4_b3_counter_diagnostic.py \
  tests/test_fsg4_b3_explicit_counters.py
10.00/10
```

```text
pytest -q tests
1243 passed, 3 skipped, 6 warnings in 440.91s
```

3个skip分别是TVM已存在时避免重复编译，以及两项冻结VNN-COMP checkout在测试默认路径不可用；均非
B3-0回归。6项warning为既有Torch JIT/profiler与treespec弃用提示。

## Claim 边界

- 没有运行真实B2 GPU worker；
- 没有观察到真实counter；
- raw worker虽然保留诊断运行时的计时字段，但本阶段明确
  `diagnostic_timing_claimed=false/performance_claimed=false`；
- 不实现B3-A/B/C，不启动B4 TIR。

## 首次真实诊断失败与修正

source=`f6004a6`的首次`run`完成了真实B2 solver、语义与环境门禁，但snapshot counter gate拒绝，因此临时
staging未发布为有效artifact。随后以同一source执行只读debug journal聚合，观察到唯一不一致：

```text
timed_candidate_d2h_copy_count: expected=12, observed=6
其余全部固定counter与预注册一致
```

根因是计数seam覆盖不完整，而不是B2物理路径少了6次copy：

- 6个β candidate在`_replacement`执行GPU→CPU，已被原实现统计；
- 6个α candidate先在`_project_alpha`把GPU dense值copy进CPU-owned sparse layout，进入
  `_replacement`时已经是CPU，原实现漏计。

修正保持预注册门槛`12`不变，在`_project_alpha`增加显式设备转换计数，并让snapshot gate报告逐字段
expected/observed。debug raw worker保留在`/tmp/fsg4-b3-counter-debug.XOJtGg/worker.json`，不属于正式
artifact且不形成性能结论。

## 下一步

1. 全量回归并提交实现，冻结source commit；
2. 从该commit运行一次fresh B2 GPU control diagnostic；
3. replay通过后，将实际counter与预注册表逐项比对；
4. 若预期成立，关闭B3-0并启动B3-A PreparedCoreTemplate；否则先修正预注册事实，不改门槛迁就结果。
