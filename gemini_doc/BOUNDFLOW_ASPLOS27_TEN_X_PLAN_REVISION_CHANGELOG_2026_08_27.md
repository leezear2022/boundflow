# BoundFlow ASPLOS’27 / 10× 总体计划 v6 修改记录

date: 2026-08-27
status: documentation-only-draft-for-user-review
source-plan: `BOUNDFLOW_README_PIPELINE_END_TO_END_ACCELERATION_DRAFT_PLAN_2026_08_26.md`
revision: v6-asplos27-ten-x
execution-authority-changed: false
code-changed: false
performance-claimed: false

## 1. 修改原因

用户明确要求把总体计划改为以ASPLOS’27投稿为目标，并以引入通过门禁的全栈优化后约`10×`总体性能为研究
目标。原v5主要回答production region如何capture/import、是否需要新IR以及Relax/TIR/RVIR的ownership，仍以
M0—M6长期技术依赖为主，没有把ASPLOS’27最后截稿、rapid review、论文贡献收束和10×全栈速度预算放在总控
位置。

本次修订保留v5技术设计，但新增投稿总控层，并让它覆盖未来13天的实际执行顺序。

## 2. 核心修改

1. 标题和metadata升级为`v6-asplos27-ten-x`，冻结ASPLOS’27 September cycle、PL/compiler主柱、GPU
   runtime/accelerator副柱及`10x-same-solver-complete-query`北极星；
2. 把论文贡献收束为三项：verification-semantic lifting、joint compiler optimization、
   trajectory-preserving system integration；不把capture/schema/Plan/Task/Schedule/receipt分别包装成贡献；
3. 明确CIBC是2023年AAAI’24既有研究起点，BoundFlow必须继承其BoundConv/fusion/lowering/autotuning，并在
   resubmission note中解释production solver、CROWN/optimizer、lifetime/VJP/runtime与same-solver新增量；
4. 把“总体10×”唯一绑定到同一official αβ-CROWN的direct B0→final complete-query；warm、cold、amortized、
   fixed-trajectory和TTV/solved分别披露；
5. 新增`u/h/r`全栈Amdahl预算、10×可达性表和`sum_i(s_i/r_i)+u+h<=0.10`残差门禁；
6. 把“所有优化”改为O1—O7累计candidate：CIBC tensor fusion、coarse CROWN、representation/VJP/memory、
   optimizer transition、prepared runtime、batching/branching、证据驱动并行/specialization；
7. 明确禁止把嵌套或重叠scope的历史局部数字相乘；每个阶段必须从B0和前一累计candidate直接重测；
8. 增加S0—S5累计里程碑；10×不可达时关闭headline，但允许按真实2—5×形成较弱、诚实的系统结果；
9. 新增2026-08-27至09-09的13天submission sprint和48小时S0可行性门禁；S0失败则不提交未完成稿；
10. 修订Amdahl kill gate：旧`r_required>10×`仅保留给1.15/1.20单bucket路线；10×总目标改用全栈残差和
    roofline/20×探索cap；
11. 修订新颖性措辞：正式论文与arXiv/preprint分开；并发preprint需比较但不当成正式发表；
12. 增加rapid-review前两页、11页正文预算、匿名系统名/镜像、CIBC changes note、AI使用披露、宣传静默期与
    ASPLOS artifact准备要求；
13. 第一刀从通用capture bake-off改为official B0 + full-stack attribution + cumulative smoke + two-page
    feasibility；长期M0—M6保留为技术依赖，不再成为13天串行总前置。

## 3. 10×数学修订

同scope native归一化为1：

```text
S = 1 / (u + (1-u)/r + h)
r_required(S=10) = (1-u) / (0.1-u-h)
```

因此：

- `u+h>=0.10`：10×物理不可达；
- `u=3%, h=0`：其余全栈平均至少`13.857×`；
- `u=5%, h=0`：至少`19×`；
- `u=8%, h=0`：至少`46×`；
- 实际工程资格要求full-stack覆盖`>=97%`、unclassified`<=3%`。

历史FSG1 fixed-16-iteration prefix只作为诊断先验：operator约61.3%，solver/runtime约38.7%。历史CIBC graph
`2.45631×`和局部CROWN TIR `4.89834×`的乐观组合约`1.83×`；即使operator全部达到CIBC operator约
`12.795×`，host不变时总体也只有约`2.3×`。10×要求operator路径约12×的同时，把host/runtime大体压到
约原来的1/8—1/9；具体分母必须由S0在最终B0 scope重新测得。

## 4. 旧成果与NO-GO的处理

以下成果继续直接复用：CIBC Conv/TIR/schedule、IBP graph；B4-B2 P/S Linear/Conv TIR；R3 structured
owner/custom backward/residual reuse/arena；RVIR exact-call/state/publish；GC0 schema/legality；NRIR lifetime与
artifact/replay/tamper。

以下NO-GO保持：B4-C2 dense retention、R3早期细粒度launch、MR3/MR5 production wrapper、MR6 guard-only、
selected-CROWN 7% share、CIBC 17-op对activation-CROWN exact-call `0/51`直接替换、无crossover的global planner
与无法摊销的默认JIT。v6允许改变的是region/ownership和累计执行形态，不允许复活已失败variant。

## 5. ASPLOS’27官方约束写入

- final full-paper deadline：2026-09-09 AoE；
- double-blind rapid review只读前两页，论文必须实质推进PL/compiler或其他ASPLOS核心pillar；
- 11页双栏10pt正文，references/appendices不占正文页数，但正文必须自包含；
- 从其他venue重投的工作需约一页changes-since-previous-submission说明；
- 公开系统需在匿名稿中改名，并使用匿名repo；
- 必须完整披露生成式AI使用；
- 截止前两周至结果公布禁止公开宣传相关投稿；
- artifact evaluation自愿，但现有raw/replay/tamper体系应转成匿名可复现artifact。

权威来源：

- <https://www.asplos-conference.org/asplos2027/cfp/>
- <https://www.asplos-conference.org/asplos2027/artifact-evaluation/>
- <https://www.asplos-conference.org/asplos2027/ae-for-authors/>

## 6. 修改文件

- `gemini_doc/BOUNDFLOW_README_PIPELINE_END_TO_END_ACCELERATION_DRAFT_PLAN_2026_08_26.md`
- `gemini_doc/BOUNDFLOW_ASPLOS27_TEN_X_PLAN_REVISION_CHANGELOG_2026_08_27.md`
- `gemini_doc/README.md`
- `.docops/ev.jsonl`、`.docops/s.md`将在DocOps变更/验证登记时更新。

## 7. 验证

已完成：

- `git diff --check`：PASS；
- metadata、ASPLOS deadline、10× hypothesis、`performance-claimed: false`与关键gate静态检查：PASS；
- 10×`u/h/r`表用独立AWK公式重算：`u=0/1/2/3/5/8/9%`分别得到
  `10/11/12.25/13.857143/19/46/91×`，与计划一致；
- DocOps change：`ev015782`，slug=`asplos27-tenx-plan-v6`；
- DocOps validation：`ev015784`，result=`pass`；
- `dol lint --soft`：`ok=true`、`miss=[]`、`rule=[]`。

本轮为纯文档修改，不运行Python单测、GPU benchmark或TVM rebuild，也不升级任何性能claim。
