---
status: b4-closed-cibc-conv-validated-reduced
updated: 2026-08-24T15:25:00+08:00
type: report
topic: boundflow
slug: b4-original-plan-and-cibc-final-status
stage: s01
---

# B4 Original Plan and CIBC Final Status

## 起点与原计划

原始目标不是只优化一个selected-CROWN计时片段，而是沿BoundFlow全栈推进：typed production state
ownership → first-class compiler/schedule/module receipt → CUDA/TIR算子 → cumulative exact-call →
materialization frontier → whole-core/query传播。B4的内部DAG为：

1. B4-A：复用optimizer terminal evaluation，消除重复CROWN；
2. B4-B0/B1：冻结真实S/P anchors及PyTorch数学oracle；
3. B4-B2-0—B2-4：identity ABI、dense Linear、sparse Linear、dense Conv、sparse Conv；
4. B4-B3：把候选接入production exact-call；
5. B4-C：逐步扩大累计region和materialization ownership；
6. B4-D及后续：只有累计门禁通过才做更大图/JIT/runtime传播。

该顺序的目的，是先证明数学和ownership，再计时；不允许把孤立microbenchmark直接外推到求解器。

## 实际完成与停止点

| 阶段 | 结果 | 判定 |
|---|---|---|
| B4-A | terminal复用正确性与计时完成；core `1.018995x < 1.03x` | externally approved performance NO-GO；mechanism/correctness保留 |
| B4-B0/B1 | 两个真实anchor与独立oracle完成 | correctness validated |
| B4-B2-0—B2-4 | typed ABI、Linear/Conv dense+sparse TIR完成 | correctness validated |
| B4-B3 | 10 evaluation/9 mutation production exact-call完成 | mechanism validated |
| B4-C1 | provider-owned lower累计core约0.948x | NO-GO |
| B4-C2 | 6 materialization sites为0.337—0.349x，显存1.34x | hard NO-GO |
| B4-D | 因上游kill gate未开工 | 正确关闭 |

失败根因不是“没写TIR”，而是纵向alpha-CROWN候选保留了跨层dense autograd状态；局部kernel省下的
计算被梯度图和materialization成本淹没。继续在同一路线调block size不能修复ownership结构，因此
按原计划执行kill gate，而不是为了正结果修改门槛。

## 为什么转向CIBC后结果不同

CIBC针对的是另一种融合维度：在IBP/forward bound内，把lower/upper、center/deviation及相同shape的
计算横向并成一个kernel。它没有alpha-CROWN optimizer mutation和跨层autograd retention，因此局部
收益可以稳定传播到整张IBP图。

本轮不是简单调用TVM包装四次Conv，而是：

- manual TIR在一次reduction traversal中同时产生lower/upper；
- 一个packed launch取代weight clamp、四次PyTorch Conv和若干elementwise kernel；
- DLPack view、输出buffer、module与schedule由plan拥有并缓存；
- 64/128/256在真实6 Conv上预注册选择；
- baseline/candidate都纳入CUDA Graph，完整图计时包含输入copy。

因此，用户自己BoundConv原型“局部几十倍”与本轮结果并不矛盾：正式逐算子范围为`9.14—22.67x`，
geomean=`12.80x`；完整ResNet2B IBP图受未融合Linear/ReLU/add等部分限制，仍得到`2.456x`，不能把
最高的`22.67x`写成whole-model数字。

## 当前距总目标的距离

已经成立的是“编译器能在真实生产shape上生成并调度一个语义封闭的IBP BoundConv水平融合kernel，
且其收益能传播到单模型完整IBP图”。尚未成立的是“BoundFlow比auto_LiRPA快多少”“对真实BaB query
加速多少”“跨模型/跨GPU泛化”和“ASPLOS完整系统故事”。

下一研究层若继续，应按证据增量依次为：CIBC Linear水平融合 → 第二model family → same-workload
auto_LiRPA adapter → solver/query传播 → memory/runtime多分支并行。它们是扩大claim的后续路线，
不是本轮closure的遗漏验收项。

α-CROWN 若未来恢复，不能复活 C2；新的 region-level structured owner/custom VJP 设计、生命周期
门禁和外部评审入口见
`BOUNDFLOW_R3_STRUCTURED_OWNER_CUSTOM_BACKWARD_REDESIGN_PLAN_2026_08_24.md` 与
`BOUNDFLOW_R3_STRUCTURED_OWNER_EXTERNAL_REVIEW_PROMPT_2026_08_24.md`。该设计目前不开放实现。
