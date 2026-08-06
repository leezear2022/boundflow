# RVIR-v4 Corrected Production-State Capture 修改记录

日期：2026-08-06

## 背景

FSG2把beta探针指向singular/旧字段，漏掉了当前auto_LiRPA实际拥有的plural
`node.sparse_betas`。因此“显式beta tensor为0”不能支撑solver-core replacement。本轮只修正生产态
ownership与可重放捕获，不做性能比较，也不宣称BoundFlow已经替换provider计算。

## 修改

- 扩展`boundflow/runtime/rvir_v4_production_state.py`：
  - provider名称做可逆semantic-path编码；
  - alpha按activation/start-node捕获；
  - SparseBeta按layer/collection捕获`val/loc/sign[/bias]`；
  - split history保存location/sign/bias/score/depth；
  - ReLU隐式零bias与非零一般split bias分开验证；
  - typed snapshot支持安全payload roundtrip和pre/post mutation receipt。
- 新增`scripts/run_rvir_v4_production_state_capture.py`：
  - 在同一αβ-CROWN GPU run记录24-call tree与真实`update_bounds_core`边界；
  - 固定三仓库commit、model/property digest、seed、batch与迭代数；
  - 生成digest-bound capture、calls/core projections、summary、manifest和replay结果；
  - replay逐项重建语义，禁止仅凭文件digest空转。
- 扩展`tests/test_rvir_v4_production_state.py`，覆盖plural/singular字段、history mismatch、
  隐式零与非零bias、mutation closure、tamper、payload roundtrip以及score/depth。

## 诊断验证

- 定向测试：`11 passed`；
- mypy：2个source文件clean；
- Pylint：`10.00/10`；
- 真实RTX 4060 Laptop GPU诊断run：
  - calls=`24`，phase=`12 initial / 1 alpha / 11 beta / 0 unclassified`；
  - core=`1`，history entries=`36`，beta value tensors=`6`；
  - mutation receipts=`12`，changed=`7`；
  - solver status=`verified`，artifact semantic projection=`validated_corrected_capture`；
  - `performance_claimed=false`。

## 状态边界

当前实现和诊断已通过，但正式artifact必须等代码提交、source provenance可冻结后再生成。因此本修改提交
本身不关闭V4-0，不准入V4-1、V4-2或B2 timing。下一动作是从clean committed code生成正式artifact，
执行原样replay和篡改拒绝探针，再更新closure文档。
