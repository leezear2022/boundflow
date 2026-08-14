---
status: active
updated: 2026-08-14T12:25:00Z
type: plan
topic: boundflow
slug: fsg4-b3-five-fresh-correctness
stage: s01
---

# FSG4/B3 五组 Fresh B2/B3-C Correctness 预注册

## 1. 目标与非目标

目标是在正式B3计时前，用同一clean source、同一模型/性质、同一协议完成5组直接B2/B3-C语义比较。
每个配置必须是独立fresh Python/GPU进程，共10个worker。

本阶段只判定correctness/admission，不计算、不报告、不选择query/core speedup。raw worker虽然保留原始计时
字段，但pair report不得产生ratio、geomean、winner或performance claim。

## 2. 冻结输入

- model：VNN-COMP 2021 CIFAR10 ResNet2B ONNX；
- property：ResNet2B `prop_0_eps_0.008.vnnlib`；
- device：本机RTX 4060 Laptop CUDA；
- solver：固定αβ-CROWN/auto_LiRPA/VNN-COMP commit和B3-C source HEAD；
- protocol：seed=100、max_iterations=1、batch=64、α/β steps=`5/10`、attack skip；
- mode：全部control；不运行profile。

## 3. 冻结执行顺序

| Pair | Position 0 | Position 1 |
|---:|---|---|
| 0 | B2 | B3-C |
| 1 | B3-C | B2 |
| 2 | B2 | B3-C |
| 3 | B3-C | B2 |
| 4 | B2 | B3-C |

不得根据中间结果改顺序。每个position调用一次
`scripts/run_fsg4_b3_counter_diagnostic.py run`子进程；父进程不能在同一Python进程内复用CUDA context。

## 4. Raw-first与中断恢复

- root先写`protocol.json`，绑定source/code revision、顺序、模型/性质digest和CLI输入；
- 每个worker写入独立`runs/pair-XX/{b2|b3c}/`诊断artifact；diagnostic自身使用临时目录原子rename；
- 中断后`--resume`只能重放并接受完整、source一致的已有run；不完整run不得追认；
- 10个raw run全部完成后才生成`report.json`与root `manifest.json`；
- replay从raw worker重新解析typed run、重新比较语义和counter，不信任report聚合。

## 5. 每组验收

每个pair必须同时满足：

1. B2/B3-C source identity、protocol identity、GPU/runtime identity一致；
2. 两个environment gate均admitted，外部CUDA process/独立thermal slowdown均为0；
3. provider core/compute/update和fallback均为`0/0/0/0`；
4. B2与B3-C的status/success/visited domains、queue accounting、depth/history、shape、inf mask、final
   decision、split depth/batch、verified/splits全部exact；
5. lower/finite upper按冻结`atol=rtol=2e-4`比较，B3-C不得optimistic；
6. B2 physical counter符合冻结B2 schema；
7. B3-C physical counter符合`VALIDATED-B3-C-COUNTERS`：D2H=`0`、candidate/commit/backup/copy=
   `12/12/12/12`、optimizer=`10/9`、snapshots=`0`、forward=`4`、KFSB=`3/3`；
8. B3-C post-query audit存在且headline digest=`0`，audit/commit receipt闭环；
9. `performance_claimed=false`、`timing_admitted=false`。

## 6. 总门禁与Kill Gate

- 只有5/5 pair全部通过、10/10 worker完整且root replay/tamper通过，才可标记
  `VALIDATED-B3-FIVE-FRESH-CORRECTNESS`；
- 任一语义、environment、source、provider/fallback、counter、audit或replay失败即
  `BLOCKED-B3-FIVE-FRESH-CORRECTNESS`，停止正式计时；
- 不允许删除失败pair后补一个新pair来维持5个成功样本；失败历史必须保留并修复后整体重新开始新版本；
- 通过本门禁只开放36-process B0/B2/B3正式计时，不开放B4—B7。

## 7. 产物

- runner：`scripts/run_fsg4_b3_correctness_pairs.py`；
- artifact：`artifacts/fsg4-b3-correctness-pairs/resnet2b-prop0-v1/`；
- tamper report：同目录名加`-tamper-report.json`；
- closure：单独变更记录和外部审计交接。

## 8. Implementation Candidate（2026-08-14）

- paired runner、root protocol/report/manifest、historical-source replay与`--resume`已实现；
- 7类outer-resigned tamper probe已实现；
- static tests=`5 passed`，mypy clean，Pylint=`10.00/10`；
- 当前状态=`IMPLEMENTED-PENDING-CLEAN-SOURCE-RUN`，尚无10-worker artifact或correctness claim。
