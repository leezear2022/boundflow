# RVIR-v4 V4-2B Production Optimizer Step Trace 正式关闭记录

日期：2026-08-13

## 结论

系统重启后GPU门禁恢复，V4-2B正式ResNet2B production optimizer-step artifact已从
`af8db0832360fc3ef029686a81e73f337286d8ae`的4个runner-enforced clean code paths生成，并通过原始
语义重放和五类同步重签名篡改探针。生成时工作树另有DocOps自动事件，不把它误写成全工作树clean。
V4-2B以`VALIDATED-PRODUCTION-TRACE`关闭。

这个结论只证明provider真实10-evaluation/9-Adam-update轨迹已被完整捕获、typed化和可审计重放；它不
证明BoundFlow已经独立执行optimizer mutation。V4-2总体、B2计时与任何性能claim仍未准入。

## 环境恢复证据

- 内核：`7.1.8-arch1-3`；loaded NVIDIA module与NVML/KMD均为`610.57.04`；
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB；`nvidia-smi` exit 0；
- BoundFlow环境PyTorch `2.12.1+cu132`与external αβ-CROWN环境PyTorch `2.11.0+cu130`均报告
  `torch.cuda.is_available()=true`；两个环境均执行真实CUDA tensor probe；
- external固定身份：αβ-CROWN=`e5c7e17...49f4`、auto_LiRPA=`5a098e8...f2d`、
  VNN-COMP=`90419aa...6cf`；模型/property SHA256分别为`791aa24d...a6d`与`89edf066...9ff`。

## 正式工件

- 路径：`artifacts/rvir-v4-optimizer-step/resnet2b-core-step-trace-v1/`；
- manifest SHA256：`7d7745e40a901c4f3a420188c42efa2f487248d2da992ca0c08730beb612fbe6`；
- source git head：`af8db0832360fc3ef029686a81e73f337286d8ae`；
- trace hash：`fa070bb09265337b297e18fd38ff342ffa657fad75c447f340789a1cb5cc31f4`；
- summary hash：`8ae8be3fc53f1571109f265ea96a9ddacf806c3c5596459574517df2783505b7`；
- 结构：1 core、24 calls；phase=`initial 12 / alpha 1 / beta 11 / unclassified 0`；
- optimizer trace：10 evaluations、9 observed Adam updates、每步24个raw state tensors；
- 每个相邻step恰有7个mutable tensor改变，9个transition均为7；所有state source均为CUDA；
- `optimizer_replacement_admitted=false`、`b2_same_solver_timing_admitted=false`、
  `performance_claimed=false`。

原始CLI replay exit 0，并从raw `production_capture.pt`和`optimizer_step_trace.pt`分别重建typed trace，
再逐字段比较JSON projection、summary、stdout、code revision和manifest/file digests。

### 与冻结V4-0 capture-v2的source parity

新GPU重跑不会伪装成raw digest exact：pre/post snapshot hash与历史capture-v2不同。独立parity verifier
先分别完整replay两个工件，再逐字段比较，确认差异边界为：

- source/protocol/solver、24-call topology/schema、1-core结构、62 tensor path/schema、36 history、policy、
  branch decision及12个mutation path/changed flag全部exact；finite mask与sign全部exact；
- pre/post中10/17个tensor content digest因有限浮点重跑发生变化，最大absolute diff均为
  `6.079673767089844e-06`；最终lower max diff=`3.5762786865234375e-07`；均通过预注册
  `atol=rtol=2e-4`，upper非有限模式exact；
- 报告：`artifacts/rvir-v4-optimizer-step/resnet2b-core-step-trace-v1-source-parity.json`；report hash=
  `c2b48275314223b19cddbbc1c36b591590c40a7a6f9f1c0af9cc662c19198aec`，文件SHA256=
  `a84daeed1719ee48d1d8ebeb23c945da96d9a1e03088555fe8de073dc7cc3476`，verifier code SHA256=
  `a08b295efd374482c9077f966f941d2ecb3e092813d3142ea591146c973004ad`；重复生成逐字节一致。

因此这是预注册数值容差内的GPU浮点漂移，不是source、topology、policy或离散state漂移。专用负向测试
对内部有效、保持符号但超出容差的α值确认会由numeric tolerance门禁拒绝。

## 同步重签名篡改门禁

新增`probe_rvir_v4_optimizer_step_artifact_tamper.py`。每个probe复制正式工件、修改payload，重写
`production_capture.pt`，更新其file digest并重新计算manifest hash，然后运行完整replay：

| probe | 同步修复的攻击者视图 | semantic拒绝原因 |
|---|---|---|
| state internal rehash | tensor content、tensor digest、step state hash、trace hash、manifest | call/state交叉绑定不同 |
| lower internal rehash | lower content、lower digest、trace hash、manifest | call/lower交叉绑定不同 |
| call-result resign | 独立call result digest、manifest | call/lower交叉绑定不同 |
| step-lineage internal rehash | step call id、trace hash、manifest | trace/call lineage不同 |
| policy internal rehash | decay、policy hash、trace hash、manifest | fixed production policy不准入 |

五项全部为`rejected-as-expected`。正式报告：
`artifacts/rvir-v4-optimizer-step/resnet2b-core-step-trace-v1-tamper-report.json`，report hash=
`b11ca1da812df8ab81a1763d55241f87b58db0d6e5c468528df520ddd9bcab79`；报告直接绑定probe code
SHA256=`d4110e455d7fb06bbb473dbc1db489d98485ab53a7733a5f228a642b5995fa3a`。独立重复运行的报告逐字节
一致，文件SHA256均为`9fed4f97883016c3fedddc2af1b72c028c7b5bd38fc756f29d59ee926cda6b16`。

## 验证

- formal generate：exit 0；original replay：exit 0；tamper probes：5/5拒绝；
- focused optimizer/artifact/source-parity tests：`29 passed`；
- Black：2 files unchanged；mypy（explicit package bases）：2 files clean；Pylint=`10.00/10`；
- GPU恢复后的full suite=`1157 passed, 3 skipped`；3项skip为1项TVM重复编译规避与2项测试自身未配置
  frozen VNN-COMP checkout，不包含CUDA skip；
- DocOps validate/lint在提交前复核。

## 下一步

只进入V4-2C pre-state native initializer：把V4-1 topology/layout映射抽为共享mapper，恢复dense α与
SparseBeta，并对compressed-coordinate round trip、external intermediate bounds及split/history hash
做exact/`2e-4`门禁。在V4-2D逐step mutation parity和V4-2E atomic copy-out都通过以前，V4-2/B2继续
关闭，不得开始性能计时。
