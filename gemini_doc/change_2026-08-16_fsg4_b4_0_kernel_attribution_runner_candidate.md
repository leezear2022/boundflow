# FSG4/B4-0 Kernel/Materialization Attribution Runner 实现记录

日期：2026-08-16  
状态：`IMPLEMENTED-PENDING-CLEAN-SOURCE-FORMAL-ARTIFACT`  
性能声明：`performance_claimed=false`

## 1. 目标

在不修改 B3 默认求解路径的前提下，为一个 fresh B3 control worker 和一个独立 profiler worker建立
可审计的 B4-0 artifact。该 artifact只回答“14次 production lower-only CROWN及其周边究竟由哪些
operator、CUDA kernel与materialization组成”，不回答B4是否加速。

## 2. 本轮实现

- 新增`boundflow/runtime/fsg4_b4_kernel_attribution.py`：
  - typed profiler event与canonical parser；
  - correlation-parent优先、显式temporal-marker fallback的phase归属；
  - CPU operator、CUDA kernel、stream、shape、memory delta与materialization ledger；
  - exact/root phase aggregation及B3 raw Amdahl/required-region-speedup重算。
- 新增`scripts/run_fsg4_b4_kernel_attribution.py`：
  - `worker/generate/replay`三种模式；
  - control/profile进程分离，profile时间永不进入speedup；
  - optimizer 10 CROWN、terminal export 1 CROWN、KFSB 3 CROWN以及4次forward的显式marker；
  - 全量raw event以确定性gzip JSONL保存，manifest绑定压缩文件SHA256，worker同时绑定解压内容
    SHA256、canonical raw hash、行数与worker hash；
  - source/code/protocol、B3正式manifest、外部仓库commit、模型/性质digest与零本机路径门禁。
- 新增11项schema/aggregation/runner测试，覆盖CUDA correlation、user annotation与真实kernel区分、
  temporal fallback、unattributed保留、canonical roundtrip、gzip确定性、worker hash与日志脱敏。

## 3. 测量纠错

真实RTX 4060 profiler smoke揭示：PyTorch会为带CUDA时间的复合CPU annotation生成
`device_type=CUDA,is_user_annotation=true`事件，例如`Optimizer.step#Adam.step`。它不是kernel。
实现现按`is_user_annotation`将其记为`phase_device_total`，只有`is_user_annotation=false`的CUDA
device event才计作kernel。无correlation parent的kernel不被删除，而是以最小时间包含marker进行显式
`temporal_marker`归属；仍无法归属时保留为`unattributed`。

smoke同时确认14-call marker为`10/1/3`、forward marker为`1/3`，gzip raw管线可承载约27万级事件。
这些是实现诊断，不是正式B4-0 artifact数字，也不是性能结果。

## 4. 验证

- B4 targeted：`11 passed`；
- B3 frozen replay：PASS，summary hash保持
  `4c19afd43c18e0409932b86506efdaf6bfc3e07baabcc222dbe79c8149f99bac`；
- B3/B4相关回归：`50 passed`；
- 全量：`1325 passed, 3 skipped`（需`conda activate boundflow`加载TVM hook）；
- Black：PASS；Mypy touched files：clean；Pylint：`10.00/10`；`git diff --check`：PASS。

首次直接调用Conda Python但未激活环境时，三项PR-12测试在collection阶段因`import tvm`失败；激活
`boundflow`后全量通过，故该项属于环境hook边界，不是代码回归。

## 5. Claim 边界与下一步

当前只支持“B4-0 runner/schema已实现并通过本地验证”。尚未从clean source生成正式artifact，未关闭
phase/opportunity门禁，未准入B4-A或B4-B，也没有B4 speedup、B0 parity或ASPLOS-ready claim。

下一唯一动作：提交clean source，再从该commit运行fresh control/profile、root replay与tamper；只有
正式raw证明至少一个candidate覆盖`>=5%` B3 core或能消除一个完整重复CROWN call，才准入B4-A/B。
