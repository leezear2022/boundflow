# BoundFlow MR2 Production CROWN Subgraph/Owner Inventory 预注册

> 日期：2026-08-26  
> 性质：既有证据的只读 contract inventory；无GPU、无solver、无计时  
> 前置：MR1-S full-graph direct replacement=`0/51 eligible`  
> 性能声明：`performance_claimed=false`

## 1. 目标

MR1-S 已排除“把 CIBC 17-op IBP整图直接塞进 activation-BaB call”。MR2 改为真实调用内部的
CROWN subgraph/owner边界：现有P-anchor Conv与S-anchor Linear证据，距离一个可接入production
exact-call的typed contract还缺什么。

本轮只建立**证据与缺口账本**，不把局部correctness/speedup伪装成production coverage，不改
αβ-CROWN、不实现bridge、不计时。

## 2. 冻结输入

1. RVIR-v3 production inventory `inventory.json`：
   `ab0595bb002b79d80be8b78abd7a795a8aa634b17c9a8524df5a1b9b5fe19e06`；
2. R3-0 P-anchor structured contract bundle：
   `2f6aaca66be142db2a72c99f601085026da6e0b7a9d36c72f486b998c1584a25`；
3. R3-D2B P-anchor 10/9 trajectory correctness manifest：
   `d43db41d3a6b16dc9a73306581499fdea521bd8508c351d325d27536ab28359f`；
4. R3-3 S-anchor active-β correctness manifest：
   `3413c578ca3dae83f6d5d18683124a758c8fb12f8e6801b4f8b719d1539f0a4c`；
5. B4-B2 CIBC v2 P-anchor local fusion manifest：
   `372651d3c0f9f516d6af6812a537ae9d33e7323e4827d9d6de473225afae2511`；
6. B4-B2 v1 physical NO-GO manifest：
   `b84d74d9b8398ca7593b78481da613bb8fb0acb6196dd2dd5fa26b364ac813d7`；
7. MR1-S ledger/summary/manifest必须replay通过。

局部性能数字只用于给evidence row标历史状态，不参与MR2 admission或route排序。

## 3. 固定 candidate sites

### P-anchor Conv

- start node=`25/Conv_8`；
- compressed α=`[2,1,6,86]`，β shape=`[6,0]`且必须absent；
- lower/upper=`[6,16,8,8]`，weight=`[16,16,3,3]`；
- structured region=8 nodes、2 scratch slots、无saved dense coefficient；
- 已有single-site backward与10 evaluation/9 mutation correctness；
- 已知`production_connected=false`，不能偷改为true。

### S-anchor Linear

- site=`31/Gemm_14` / `semantic-active-beta-gemm-14`；
- active β必须存在且gradient非零；
- 已有compressed α/β VJP correctness；
- isolated timing已NO-GO，不能借P-anchor数字；
- 尚无10/9 production optimizer trajectory与production exact-call connection。

不得追加第三个synthetic site，也不得把IBP Conv horizontal fusion列作CROWN production site。

## 4. 七层 readiness ledger

每个site逐层给出`proven/missing/rejected/not-applicable`及唯一证据路径：

1. `production_site_identity`：start node、ordinal、shape、dtype/device、model/property；
2. `typed_input_output_abi`：incoming A/bias、bounds、weight、α/β、split/history、outputs；
3. `state_ownership`：α/β/split/history/optimizer mutation谁读、谁写、谁提交；
4. `forward_backward_correctness`：独立oracle、VJP、active/absent β语义；
5. `optimizer_trajectory_correctness`：真实evaluation/mutation/clamp/scheduler序列；
6. `multi_site_consumer_closure`：相邻site、dense A逃逸、consumer与saved-state生命周期；
7. `production_exact_call_connection`：真实provider call中恰一次dispatch/emit，fallback/shadow=0。

`ready_for_bridge_correctness`要求1–5均proven，6有明确fail-closed single-site边界，且唯一缺口是7。
任何缺字段按missing处理，不能从文档措辞推断proven。

## 5. 机械 route

- 若恰有一个site满足`ready_for_bridge_correctness`：只开放该site的
  `production-exact-call bridge correctness`预注册；不开放timing或multi-site；
- 若多个site满足：按固定顺序P-anchor→S-anchor选第一个，避免按历史性能挑选；
- 若0个满足：关闭当前两site bridge，回到production capture/schema补证据；
- 若发现现有证据互相矛盾：`BLOCKED-CONTRACT-CONFLICT`，停止。

MR2通过不意味着生产接入已完成。下一阶段仍必须保持provider baseline、同一solver状态和atomic
commit，先做correctness five-fresh，再决定是否能单独预注册timing。

## 6. 工件与篡改

正式工件固定为
`artifacts/measurement-recovery/mr2-production-crown-subgraph-owner-inventory-v1`，包含input
snapshot、`site_ledger.jsonl`、`gap_matrix.json`、`summary.json`、manifest、replay receipt和至少
12类fully re-signed tamper。覆盖site identity、β presence、trajectory、production_connected、
evidence digest、gate status与route。

## 7. Claim boundary

最多允许claim“哪个冻结site在既有证据下离production bridge correctness最近，以及具体缺口”。
不得claim production coverage、same-solver speedup、局部历史数字可传播、multi-site已解决或
ASPLOS-ready。

