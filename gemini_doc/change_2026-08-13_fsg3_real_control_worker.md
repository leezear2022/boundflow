# 2026-08-13 — FSG3 Real B0/B1/B2 Control Worker

## 改动

- live executor支持显式precompiled Program/Module与关闭timed-scope correctness payload复制；
- 新增同一official αβ-CROWN host中的B0 original、B1 typed passthrough、B2 whole-call control worker；
- B2在query前独立测量ONNX→BoundIR/reference plan cold compile，query内使用process-hit plan；cold total
  固定为`compile + query`组合，隔离property copy/hook setup并另存raw cold outer诊断；
- query/core wall和CUDA event、compile、post-validation、peak memory分别记录；
- B1构造并validate production typed snapshot后exactly-once调用provider，B2保持zero-provider/fallback；
- post-query提取queue、lower-only upper mask、branch与termination语义；
- 保存NVSMI XML环境、compute process、AC、thermal counter与设备identity原始证据。

## 边界

本切片只开放control worker。profile请求在分层span与closure实现前显式fail closed；尚未生成正式36-run
artifact，也不形成speedup claim。

## 验证

- 真实GPU smoke分别以fresh进程执行B0/B1/B2，三路worker均退出0并通过typed raw schema：
  - B0：provider `core/compute/update=1/14/3`；
  - B1：typed validation一次，provider `core/compute/update=1/14/3`；
  - B2：whole-call replacement，provider/fallback=`0/0/0/0`；
- 三路离散语义exact；B1/B2相对B0的lower最大绝对差分别为
  `7.152557373046875e-07`与`8.344650268554688e-07`，低于预注册`2e-4`门禁；
- 三次smoke开始时驱动thermal evidence不满足环境准入，因此只证明worker物理路径与语义，不进入
  performance统计，也不保存为正式artifact；
- helper/unit：`20 passed`；全量回归：`1218 passed, 3 skipped`；
- 两个改动脚本在激活`boundflow`环境后mypy clean，pylint `10.00/10`，`git diff --check`通过；
- 一次未激活Conda hook的mypy/pylint调用因TVM/PYTHONPATH缺失失败，随后按仓库规定激活环境重跑
  通过；该失败属于验证命令环境错误，不是代码回归。
