# S4-1B 六站点 production correctness 实现变更记录

date: 2026-08-31
status: implemented-correctness-candidate
external-audit: pending
timing-recorded: false
performance-claimed: false
source-commit: 760fa0d

## 1. 本轮目标

在不进入S4-1C梯度、optimizer或性能计时的前提下，把已经关闭的S4-1B0 ternary端点和此前R3/RVIR
资产接成一条真实production correctness纵向切片：

```text
R31B2 coefficient propagation
  -> ordered Pass A / six compact selectors
  -> coefficient-arena generation rebind
  -> one 42-read + 7-write TVM Pass B
  -> V17/V19/V23/V25/V28/V31 persistent arena
  -> coefficient recompute handoff boundary
```

## 2. 代码变更

实现提交=`760fa0d`，严格落在施工包冻结的五个文件，总计2,485行：

- `boundflow/backends/tvm/asplos27_s4_six_site_value.py`
  - 六个selector pack TIR：一个ternary endpoint、五个binary sign；nonfinite统一写`-128`；
  - 49参数Relax ABI：42 read + selected-input target + 6 V target；
  - 6 cuDNN Conv、1 Gemm、1 input select、5 selected-ReLU、6 persistent copy；
  - source/partition/lowered/device source均保留canonical content并由hash反向复核，不只检查64字符格式。
- `boundflow/runtime/asplos27_s4_coefficient_selector_pass.py`
  - 冻结19-action单次状态机，漏项/重复/换序/stream/generation/identity均fail closed；
  - 在真实R31B2、D1C/D2B staged residual边界截取A29/A26/A24/A20/A18/Ainput；
  - 六个source/output DLPack view在Pass A前绑定，production capture不走PyTorch eager pack。
- `boundflow/runtime/asplos27_s4_six_site_value.py`
  - `PREPARED -> PASS_A_RUNNING -> SELECTORS_READY -> ARENA_REBOUND_FOR_SELECTED_INPUT ->`
    `PASS_B_RUNNING -> VALUES_READY -> COEFFICIENT_RECOMPUTE_READY`；
  - selected-input必须与coefficient arena同storage；V17—V31必须是一个37,464-element arena的连续view；
  - 49个DLPack参数view只在prepare建立，warm不创建descriptor；VM result owner为固定单字段；
  - 同时核对Torch current stream、TVM-FFI raw stream和prepared stream identity；
  - receipt固定`49/90/110` descriptor账、6 logical stage、6 persistent copy和全部false claim flags。
- 两个测试文件覆盖随机CUDA fixture、production ResNet2B fixture、结构门禁和负向篡改。

## 3. 对旧工作的复用

本轮没有把此前工作推倒重来：

- S4-1B0的zero→midpoint ternary语义直接成为Pass B input select；
- R31B2的seed/Linear/ReLU/Conv系数传播顺序仍是production owner；
- D1C/D2B两段residual kernel继续执行，A26/A20只是在stage1与stage2之间读取既有scratch slice；
- coefficient scratch仍为两个既有arena，Ainput selected value在live reader清零后原位复用；
- active alpha直接使用`[D,W]`，没有扩回`[2,1,D,W]`；
- S4-1A的ordered mutable-buffer ABI仍是后续正式组合owner，不建立第二套参数所有权。

## 4. 已验证结果

- 新增专项：`9 passed`；
- S4-0/S4-1A/S4-1B0/S4-1B/R3-D2B联合：`189 passed`；
- 全量：`2082 passed, 3 skipped`；三个skip分别为已有TVM重复编译规避1项、冻结VNN-COMP checkout
  缺失2项，均非本批回归；
- production冻结ResNet2B：真实19-action Pass A完成，六个TIR selector pack完成，Pass B六槽逐项满足
  `torch.testing.assert_close(rtol=2e-4, atol=2e-4)`；
- selector/action/module/runtime receipt的claim/hash/count篡改均拒绝；
- 五个交付文件mypy clean；
- 五个交付文件Pylint `10.00/10`；
- Black与`git diff --check`通过。

## 5. 明确未做

- 没有S4-1C compressed dα/dβ；
- 没有10-step/9-mutation optimizer trajectory；
- 没有same-solver exact-call替换；
- 没有计时、显存headline、speedup或10x投影；
- 没有formal multi-process artifact，也没有外审批准；
- 没有升级ASPLOS-ready claim。

## 6. 当前状态与下一步

当前只能标记：

`IMPLEMENTED-CORRECTNESS-CANDIDATE-S4-1B-SIX-SITE`

下一步是把本批固定commit和完整验证清单交给独立外审。外审通过后才可关闭S4-1B并另行开放S4-1C；
不得直接跳到timing或optimizer。
