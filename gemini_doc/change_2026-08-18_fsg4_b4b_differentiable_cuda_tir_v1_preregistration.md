# FSG4/B4-B Differentiable CUDA/TIR v1 预注册记录

日期：2026-08-18

状态：`PREREGISTERED-B4-B-V1-NOT-IMPLEMENTED`

## 背景

B4-0已冻结可微 lower-only CROWN opportunity；B4-A已外审关闭为性能NO-GO，其约
1.9% core改善不得进入B4 cumulative baseline。B4-B因此需要以B3为直接基线，
独立证明可微状态所有权和CUDA/TIR物理收益。

## 冻结决策

- S-anchor：`node31/Gemm_14`，incoming coefficient=`[6,1,100]`，producer=`/input-28`，
  active beta=`(6,1)`，用于证明α/β/incoming-A gradient和optimizer mutation所有权。
- P-anchor：`node25/Conv_8`候选，incoming coefficient=`[6,1,16,8,8]`，producer=
  `/input-20`，用于冻结高占比Conv transpose-contraction的production signature和物理收益。
- 第一工程步只做B4-B0：optimizer evaluation 0的gradient-active、read-only exact-call capture，
  两锚点各至少5 fresh process，含raw tensor/output/gradient、root replay与tamper。
- B4-B0通过前不实现TIR；不改宽`boundflow/runtime/fused_crown.py`的PR-12 plain capability。
- capture明确分离production compressed α/β映射源与native dense α/β/
  `relu_pre_add_coeff_l`算子输入；gradient归属于native dense leaf，不把压缩源伪造为
  exact-region autograd leaf。
- 未来TIR必须有独立schema/cache/dispatch，forward与custom backward同时通过；单shape局部
  speedup不得外推为whole-core/query claim。

## 量化边界

B4-0冻结的B3 core share为`0.6771722591`。只在未来B4-C真实覆盖该全share时，
whole-core `1.10x/1.20x`才分别对应region约`1.15510x/1.32647x`。B4-B v1必须从
activation receipt重算measured eligible share，不得直接67.72%。

## 路由

下一唯一工程动作是B4-B0 production exact-call capture。详细准入、数值、性能、
activation与kill gates见
`gemini_doc/BOUNDFLOW_FSG4_B4B_DIFFERENTIABLE_CUDA_TIR_V1_PLAN_2026_08_18.md`。
