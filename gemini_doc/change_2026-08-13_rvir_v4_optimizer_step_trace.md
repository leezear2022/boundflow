# RVIR-v4 V4-2B Optimizer Step Trace 修改记录

日期：2026-08-13

## 结论

V4-2B的typed step trace、production observer与正式artifact runner已经实现，状态为
`IMPLEMENTED-CAPTURE-READY / FORMAL-ARTIFACT-BLOCKED`。它证明代码能够在作用域内观察真实
PyTorch Adam执行并形成可语义重放的10 evaluation/9 update raw trace；它尚未证明真实ResNet GPU
production trace，不能关闭V4-2B、V4-2或准入B2。

## 代码修改

- `boundflow/runtime/rvir_v4_optimizer_mutation.py`
  - 新增严格mutation policy payload parser；
  - 新增`ProductionOptimizerStepV4`与`ProductionOptimizerStepTraceV4`；
  - raw payload保存每step的24个α/SparseBeta tensors与lower，metadata绑定content digest、schema、
    ownership、lineage、policy和trace hash；
  - 固定10 evaluation/9 update、9个observed Adam step ordinal、双LR `0.01/0.05`及逐步`0.98` decay；
  - copy-in漂移、raw tensor篡改、step/policy/LR schedule/mutation count漂移均fail closed。
- `scripts/run_rvir_v4_production_state_capture.py`
  - 可选optimizer-step模式只观察active core内outer optimized call的depth-1 beta calls；
  - hook真实`torch.optim.Adam.__init__/step`，拒绝非两个parameter groups、无preceding evaluation、
    step错序、重复Adam实例；context退出恢复原methods；
  - 从live `BoundedModule.bound_opts`捕获完整controls并与core policy绑定。
- `scripts/run_rvir_v4_optimizer_step_artifact.py`
  - 新增独立generate/replay CLI；formal generation固定CUDA、三仓commit、模型/property digest和clean
    code provenance；
  - replay同时检查manifest/file/code digest、24-call phase tree、core/trace policy、raw trace语义和
    冻结JSON projection；
  - 明确`optimizer_replacement_admitted=false`、`b2_same_solver_timing_admitted=false`、
    `performance_claimed=false`。
- 测试新增真实CPU Adam + ExponentialLR的嵌套执行，不以手工计数替代step观察；另覆盖parameter-group、
  source、严格CUDA device语法、lower、copy-in、ordinal、LR schedule和mutation-count负向门禁；
- README、current status、execution memo、master plan与claims map已同步当前V4-2B范围，防止把
  implementation-ready误读为formal closure。

## Production 配置事实修正

沿`LiRPANet.set_crown_bound_opts()`和activation-split core的实际赋值顺序复核后：

- beta core消费已attach的pre-state α，因此live `init_alpha=false`，不是`true`；
- 本固定协议先设置alpha-CROWN `max_time = 1.0 × bab timeout`，再进入beta设置；`bab timeout=60`，
  所以live `max_time=60.0 s`，不是default `1e9`；
- 相反配置现在会被mutation policy gate拒绝。这是对真实production call的收紧，不是为了让测试通过
  而放宽合同。

## 验证

- focused V4-2/V4-0：`31 passed`；
- Black：通过；
- mypy（runtime、两个runner、两个test文件）：clean；
- Pylint同五文件：`10.00/10`；
- full suite：首轮与最终文件冻结复跑均为`1108 passed, 39 skipped`；最终复跑耗时`381.88 s`；
- CUDA probe：当前运行内核=`7.1.5-arch1-2`，已安装内核包=`7.1.8.arch1-3`；已加载NVIDIA kernel
  module=`610.43.03`，用户态`nvidia-utils/libnvidia-ml=610.57.04`。因此`nvidia-smi`报告NVML
  driver/library mismatch；αβ-CROWN Torch=`2.11.0+cu130`，`torch.cuda.is_available=false`，
  `cudaGetDeviceCount error 803`。这是一致性明确的“升级后尚未重启”状态；
- formal worker probe按预注册在`RVIR-v4 production capture requires CUDA`处fail closed，没有生成
  `/tmp/rvir-v4-optimizer-step-probe.pt`。
- DocOps：`dol validate`首次发现历史hook留下两组重复event ID；按原时间顺序补为缺失的
  `ev006471/ev006635`并连续化本轮尾部ID，未删除事件语义。修复后`dol validate`与
  `dol lint --soft`均PASS；当前blocker精确记录为loaded NVIDIA `610.43` vs userspace `610.57`。

## 下一步

1. 重启进入已安装的`7.1.8`内核，使NVIDIA kernel module与`610.57.04`用户态库一致；重启后先过
   `nvidia-smi`与两套Torch CUDA probe；
2. 从本轮clean commit运行V4-2B formal generate；
3. 运行original replay及外层同步重签后的state/step/result/policy tamper probes；
4. 通过后关闭V4-2B，进入V4-2C pre-state native initializer；B2仍不得计时。
