---
status: preregistered-correctness-open
updated: 2026-08-26T02:00:00+08:00
type: plan
topic: boundflow
slug: r3-3-s-anchor-active-beta
stage: s01
---

# R3-3 S-anchor Active-β Correctness 预注册

## 1. 准入与目标

R3-D2-B 已以 `VALIDATED-R3-D2B-WRAPPER-RESEARCH` 关闭，因此按冻结 DAG 只开放 R3-3 correctness。
R3-3 固定 anchor=`semantic-active-beta-gemm-14`、start node=`31/Gemm_14`，验证 structured sparse
source owner 在非空 β 下的 forward、α VJP、β VJP、location/sign 与 unowned gradient 语义。

本轮不修改 D2-B P-anchor wrapper，不记录 timing，不扩 adjacent site。已有 B4-B2 B2-2 sparse Linear TIR
实现可复用，但旧测试/外审不自动关闭 R3-3；必须重新生成绑定当前 source/code/capture 的五 fresh artifact。

## 2. 冻结输入与候选

- capture：`artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1/run_00..04.pt` 的第 0 anchor；
- independent oracle：B4-B1 PyTorch reference；
- candidate：`run_b4b2_sparse_linear_tir_v1`，first-class sparse Linear template/schedule/instance；
- active β shape=`(6,1)`，location count=`6`，sign count=`6`；compressed α feature count=`27/domain`；
- empty β negative control：同 capture 的 P-anchor（anchor ordinal 1）必须在 template admission 时拒绝，
  不得复用 active specialization 或伪造零长 β gradient。

## 3. 五 fresh correctness 门禁

五个 capture/process 分别重建 IR 与执行：

1. `output_lower_a`、`output_bias`、compressed α gradient、compressed β gradient 全部
   `atol=rtol=2e-4`、sign exact；
2. β gradient shape=`(6,1)`、6/6 nonzero；native β gradient 存在；
3. projected owned α/β 与 candidate exact，unowned native α/β gradient 恰为零；
4. 27 个 α feature index 严格递增唯一；6 个 β location 合法且 sign∈{-1,+1}；
5. forward/backward launch=`1/1`，DLPack pointer exact=`21/21`，fallback/eager=`0/0`；
6. forbidden dense α/β global workspace count=`0`，persistent dense state=`0`；
7. template/schedule/module receipt 五次稳定；五个 fresh worker 各自使用空缓存，
   因此必须各自报告 cold miss；另起一个独立的同进程 cache-sequence probe，以同一 module cache
   依次执行五个 capture，必须得到 `miss,hit,hit,hit,hit`。两类证据不得混用；
8. fully re-signed tamper 覆盖 β tensor/hash、location、sign、projection、unowned-zero、launch、
   empty-specialization、protocol 与 summary gate。

## 4. 边界与下一步

通过只允许 claim `VALIDATED-R3-3-S-ACTIVE-BETA-CORRECTNESS`，并开放另行预注册的 R3-3 isolated
timing；R3-4 adjacent sites、R3-6、same-solver、query/queue 继续关闭。若任何 β ownership/gradient/
specialization 门禁失败，R3-3 当前 variant NO-GO，不得退化为 empty β 或借用 P-anchor 性能数字。

## 5. 正式前协议勘误

初版第 3.7 条同时要求“五个独立进程”与“同一缓存 miss→4 hit”，两者在进程隔离的
实现下不能由同一组 launch receipt 同时成立。本节在任何 R3-3 formal raw 生成前勘误为
“fresh correctness 各自 cold miss + 独立 cache-sequence probe”。不改 correctness、ownership、workspace、
tamper 或下一阶段边界。
