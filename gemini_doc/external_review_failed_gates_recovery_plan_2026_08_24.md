---
status: external-review-complete
updated: 2026-08-24T11:30:00+08:00
type: review
topic: boundflow
slug: external-review-failed-gates-recovery-plan
stage: s01
---

# 外部评审：失败门禁诊断与恢复计划(2026-08-24)

- 评审对象:`BOUNDFLOW_FAILED_GATES_DIAGNOSIS_AND_RECOVERY_PLAN_2026_08_24.md`(446 行)、
  `BOUNDFLOW_FAILED_GATES_EXTERNAL_ADVISOR_PROMPT_2026_08_24.md`(118 行)、
  `BOUNDFLOW_FAILED_GATES_DIAGNOSIS_AND_RECOVERY_CHANGELOG_2026_08_24.md`(71 行)
- 评审方法：不信任文档自述;全部 headline 数字回到 `artifacts/*/summary.json` 与 raw JSON 独立
  重算;mypy/pylint/dol/git 全部现场重跑;图结构用 VNN-COMP 2021 `resnet_2b.onnx` 独立计数。
- 评审环境:分支 `feat/rvir-v4-production-state-ownership-v1` @ `f87f737`,conda env boundflow。

## 总体 Verdict:**APPROVE-WITH-MINOR**

主文档的事实陈述与仓库证据高度一致:抽查的 20+ 个数字全部与 raw/外审报告逐位或按标称精度吻合,
无一处把未运行写成失败、无一处把局部数字写成系统数字。恢复路线(R0→R1→按 share 选 R2 支路、R3
不复活 C2)方向正确且纪律严格。不批准升级为 approve 的唯一原因:R1.3 的 `r_required` 路由公式
缺少一个**预注册的下一档目标 T**,使"按实测 share 选支"在操作上暂时不可计算;另有若干 minor
证据入口缺口。无 blocker、无 major(见 Findings 中 M-1 定为 minor+ 偏 major 边界,按整体判断
不阻塞文档成立,但阻塞 R1 预注册定稿)。

## 一、分类账准确性核对(复核要点 1)

逐项核对结果:

| 文档条目 | 文档值 | 独立复核值 | 来源 | 结论 |
|---|---:|---:|---|---|
| NRIR49A queue/complete share | 0.0709863/0.0705233 | 0.07098631834282758/0.070523288963519 | NRIR49A 外审(external_audit_nrir49a_g1_closure_2026_08_06.md L67-70)逐位一致 | ✓ |
| deletion-only 上限 | 1.076410x | 1/(1-0.07098631834282758)=1.0764104 | 本人重算 | ✓ |
| B0/B2 query/core | 0.9084x/0.5168x | query_wall geomean=0.90839955,core_wall geomean=0.51676701 | artifacts/fsg3-same-solver-timing/resnet2b-prop0-v5/summary.json | ✓ |
| B3 B2/B3 core/query | 1.0716x/1.0066x | 1.07161748/1.00662290 | artifacts/fsg4-b3-same-solver-timing/resnet2b-prop0-v1/summary.json | ✓ |
| B3 B0/B3 query | 0.9100x | 0.9100012637918488 | 同上 | ✓ |
| B3→parity / →1.15x | 1.09890x/1.26373x | 1.0988996/1.2637345 | 本人重算 | ✓ |
| B4-A core/query worst | 1.0190x/0.99695x | 1.0189949992/0.9969470224 | change_2026-08-18_fsg4_b4a_external_audit_closure.md L20-21 + artifact v5 | ✓ |
| B4-B2 v1 geomean/worst/alloc | 0.42484x/0.37769x/0.474638x | 0.4248423874/0.3776925294/0.4746376811 | artifacts/fsg4-b4b2-b2-5-formal-microphysics/.../summary.json | ✓ |
| v1 kernel inventory | 3 fwd+3 bwd | forward_kernel_count=3,backward_kernel_count=3,shared_mem/vector token=0 | 同上 kernel_inventory | ✓ |
| v2 Triton 对 PyTorch | 2.83772x | 2.837719484336988 | BOUNDFLOW_FSG4_B4B2_V2_CIBC_TRITON_FORMAL_CLOSURE_2026_08_24.md L49 | ✓ |
| v2 manual TIR | 4.89834x/对 Triton 1.68273x | 4.898339978572916/1.6827270318064584 | artifacts/fsg4-b4b2-v2-cibc-tir-formal/.../summary.json | ✓ |
| B4-C0 core/worst/alloc | 0.94034x/0.93418x/1.04818x | 0.9403411451/0.9341801912/1.0481830751 | artifacts/fsg4-b4c0-cumulative-core/.../summary.json | ✓ |
| B4-C1 core/worst | 0.94815x/0.94547x | 0.9481500116/0.9454748158 | artifacts/fsg4-b4c1-provider-owned-lower/.../summary.json | ✓ |
| B4-C2 三轮/显存 | 0.348761/0.337448/0.346003;1.3401x | 逐位一致 | B4C2 kill changelog L25-27 + CIBC 外审 L133-134 | ✓ |
| CIBC operator/graph | 12.7951x/9.1423x;2.45631x/2.45091x | 12.795107698179335/9.14229089216829;2.456310282102286/2.4509075978286576 | artifacts/cibc-ibp-horizontal-formal/.../summary.json | ✓ |
| per-op best schedule ≈12.929x(+~1.05%) | 12.929x | 本人从 3 份 operator raw 逐算子取 max 重算 geomean=12.929023,+1.047% | raw/operator_{64,128,256}.json | ✓ |
| 整图未融合部分 | 2 Linear/6 ReLU/2 add/flatten | ONNX 节点计数:Conv6/Relu6/Add2/Flatten1/Gemm2 | resnet_2b.onnx(VNN-COMP 2021 固定 checkout) | ✓ |
| candidate 剩余 0.071–0.072ms | — | 6 worker median ∈ [0.071252,0.072015] | raw/model_0*.json | ✓ |

(a) **B4-A 重新分类**:有据。B4-A 外审 closure 原文(change_2026-08-18 L5、L25)写明"唯一合法分类是
`VALIDATED-NO-GO-B4-A-PERFORMANCE`",机制证据保留但 1.9% 不得计入基线——本文档把 B4-A 列为正式
失败并在 §4.3 写"机制正确但区域太小",与权威 closure 一致。注意 `BOUNDFLOW_B4_ORIGINAL_PLAN_AND_
CIBC_FINAL_STATUS_2026_08_24.md` 表格把 B4-A 写成"reduced",那才是两份文档间的措辞不一致点;本
评审对象选择了与外审 closure 一致的口径,不算错误,但建议在 R0 顺手统一 B4 状态文档的措辞(minor)。

(b) **口径区分**:清楚。B4-B2 v2 的 4.89834x 在总账表(§3)与 §10 均明确标注"局部 differentiable
kernel";CIBC 12.7951x 与整图 2.45631x 分行列出,§5.3 显式说明算子级到整图的稀释机制,且 §10
明令禁止把 12.8x 写成 whole-model。与外审 info-2 的建议一致。

(c) **"未运行不写成失败"**:全文贯彻。§3 末行 B4-D/B5/B6/B7/complete solve 标 CLOSED/UNTESTED;
§2.2 末、§10"不能说"第 5 条均重复此纪律;NRIR49A §4.1 正确引用外审的范围纠正(1.0764x 仅为
单区域 deletion-only 上限,不外推全栈)。

## 二、失败归因技术正确性(复核要点 2)

- **B4-C 诊断与证据一致**。C2 kill changelog 的 root cause(6 层 dense coefficient + autograd history
  跨层存活,~2.9x 回退 + 34% 显存)与主文档 §4.7 一致;C0 的 native/TIR 双算(§4.5)、C1 的"拿到
  value 所有权但没拿到表示/生命周期所有权"(§4.6)与两份 formal closure 及 CIBC 外审 L128-137 的
  判定链一致。三层归因是递进而非重复叙述,符合 artifact 数字的单调叙事(0.940→0.948→0.34x)。
- **R0 修复项全部真实对应外审遗留 findings**,本人现场重跑确认:
  - `mypy boundflow/domains/interval.py` → 11 errors,其中恰好 3 条 arg-type 位于 83-85
    (stride/padding/dilation 的 tuple[int,...] vs tuple[int,int]),8 条 attr-defined 为既有——与外审
    AC6/minor-1 逐字吻合;
  - `pylint --enable=C0415` → interval.py:74 import-outside-toplevel 1 条,吻合;
  - 3e-4/1 ULP:CIBC closure 目前只写 `0.000244140625 < 3e-4`(L74),未写明 2^-12=该量级 float32
    1 ULP 的依据,外审 info-1 确为未闭环项;R0 立项真实;
  - steady-state:closure 未显式披露 TIR 编译与 plan 构造在计时区外,外审 info-3 确为未闭环项;R0
    立项真实。
- B4-B2 v1 "wrapper、launch 和标量 reduction 开销大于节省的计算" 属于文档标注的推断,但
  kernel_inventory(6 个标量式 kernel、0 shared-mem token、0 vector token)给了充分间接证据,且 v2
  用 1+1 kernel 翻到 4.898x 构成自然对照实验,推断可信(inference 标注在文档中已足够诚实)。

## 三、恢复路线评审(复核要点 3)

- **R0→R1 顺序合理**。R0 不改性能语义、只闭环审计卫生,成本低收益确定;R1 是唯一开放的研究动作
  且为只读测量,符合"先归因再投资"的纪律,也正确回避了"凭直觉挑算子"。
- **G1 归因可行性**:在 ~0.071ms、约 17 个节点的 CUDA Graph 上做 NVTX+CUPTI correlation id+graph
  node trace 工程上可行(Nsight Systems 支持 cudaGraphTrace;单 kernel 微秒级,CUPTI activity 时间戳
  分辨率足够)。文档预注册了 control/profile 成对 fresh worker、扰动 ≤1.05、四口径分离、raw-first,
  fail-closed 姿态正确。**两个缺口**:(i) 未提 GPU globaltimer 与 CPU/NVTX 时钟域对齐的校准步骤,
  70µs 尺度下时钟域偏差不可忽略;(ii) 单 stream capture 下 residual branch 很可能串行,"overlap-
  adjusted share" 可能退化为 kernel sum——文档 §7 R1.1 已意识到("若 graph 中存在独立 residual
  branch"),但没有写退化时以哪个口径为 headline。
- **Amdahl 路由**:公式 r_required = s/(1/T-(1-s)) 代数正确(本人独立推导一致),且"分母非正即
  不许立项"的 INFEASIBLE 规则与 NRIR49A 外审 L52-53 的反解先例一致。**但全文没有为 R2 支路冻结
  下一档目标 T**:CIBC 整图已过 ≥1.5x,2.4563x 之上的"下一系统门槛"是什么(2.8x?3x?还是直接
  挂到 same-solver query 1.15x?)未定。没有 T,r_required 无法计算,R1 之后的 GO/NO-GO 排序就没有
  可证伪的分母。这是本评审认定的**最薄弱处**(见 M-1)。
- **α-CROWN 恢复条件**:structured owner + custom backward + minimal saved state 正面针对 C2 根因
  (dense A 跨层存活),三条结构门禁(不存 dense A、最小保存/重算、可观测 release receipt)+
  saved_tensors_hooks 测量 receipt + 单 site→双 site→六 site 逐级门禁,设计上确实避开了 C0(双算:
  由"control worker 严禁 native shadow"覆盖)、C1(提前 dense:由"global dense materialization
  count=0"覆盖)、C2(跨层 retention:由 live-set/peak ≤1.0x 覆盖)。残留风险:kernel 内重算 dense
  adjoint 可能把 C0 的"双份计算"换成"kernel 内重算",其净收益在单 site 门禁(≥1.20x wrapper-
  inclusive)下可被证伪,kill condition 存在,可接受。

## 四、claim 边界(复核要点 4)

- 全 grep:主文档中 auto_LiRPA/αβ-CROWN/BaB/ASPLOS-ready 仅出现在 §2.1(禁止外推)、§10"不能说"
  与参考文献,无越界表述;§10"能说"清单的五条全部有 raw 支撑且标注了局部/单模型/steady-state 口径。
- artifact 层 `performance_claimed=false` 与文档层 VALIDATED-REDUCED 的关系沿用外审 info-4 的
  自洽解释,未见漂移。12.8x 与 2.4563x 的口径区分在 §3、§5.3、§10 三处一致。

## 五、流程(复核要点 5)

- `git rev-parse HEAD` = `@{u}` = f87f737cebff…,已推送属实;HEAD commit 即 f87f737
  "docs: diagnose failed gates and close CIBC audit"。
- `git diff --check HEAD~1..HEAD` 干净(exit 0)。
- `dol exchange status cibc-ibp-horizontal-20260824` → status=closed,approved_round=1 ✓;
  closure/audit/audit_report_full/delivery 均在 `.docops/exchange/cibc-ibp-horizontal-20260824/` 内。
- `dol lint --soft` → {"ok":true} ✓;`.docops/s.md` next=preregister-cibc-g1-optimized-graph-
  attribution,与文档"下一步"一致。
- changelog 自述行数 446/70/118,实测 446/71/118——changelog 把自身行数写错 1 行(info 级,
  不影响任何实质内容)。
- 工作树 `M .docops/ev.jsonl` 与未跟踪 `docs/CIBC_for_DAC.pdf` 与 changelog Decisions 末条自述一致;
  本评审未触碰二者。

## Findings

| # | severity | 位置 | 证据 | 建议 |
|---|---|---|---|---|
| M-1 | minor+(阻塞 R1 定稿,不阻塞本文档) | 主文档 §7 R1.3 / §8 | r_required 公式正确但全文未冻结 R2 路由所用的下一档目标 T;CIBC 已过 ≥1.5x,"下一系统门槛"未定义 | R1 预注册时一并冻结 T(建议直接挂 same-solver query 1.15x 反推所需整图增益,或显式冻结整图下一档门槛值),并写明单 stream 退化时 headline 口径 |
| M-2 | minor | 主文档 §7 R1.2 | 未提 GPU globaltimer 与 CPU/NVTX 时钟域校准;70µs 图上下偏移不可忽略 | 预注册加一条时钟域对齐与校准 receipt(CUPTI 同步点/nsys 导出核对) |
| m-1 | minor | 主文档 §12 | 证据入口缺 fsg3-same-solver-timing(B0/B2 0.9084/0.5168 的来源)、NRIR49A artifact、B4-C2 raw artifact 路径 | R0 顺手补齐三条路径 |
| m-2 | minor | B4 最终状态文档 vs 本计划 | 前者把 B4-A 写"reduced",与本计划(及 B4-A 外审 closure)的 NO-GO 不一致 | R0 统一措辞,以 change_2026-08-18 closure 为准 |
| i-1 | info | changelog Validation | 行数自述 446/70/118,实测 446/71/118 | 可忽略 |
| i-2 | info | 主文档 §6 | 引用一手资料均为官方/论文来源,未核实链接可达性(本评审范围内未逐条 fetch) | 无 |

## 文档未覆盖的风险

1. **CIBC 2.4563x 进入 same-solver 后的稀释未量化**:IBP forward-bound 只是 complete query 的一段;
   B3 的 profile attribution(B0 provider_core share≈0.9994)提示 query 内可摊薄的区域占比需要先测,
   否则 R2 的整图增益可能在 query 层被稀释到噪声以下。需要哪项 raw:B3/B0 同 workload 内 IBP
   forward 占 query wall 的 share(R1 可顺带产出)。
2. **8 GB 显存的"自然 workload"未构造**:R4 memory 路线仍缺非人为放大 batch 的 workload 定义;
   advisor prompt Q6 已点名,主文档未答。
3. **第二 model family 全部推迟到 R5**,期间所有 claim 仍是单模型;若 R2 投入大,建议把第二模型
   的 smoke 提前作为廉价防过拟合闸门。
4. **CUDA Graph private pool 显存未比较**(外审已披露,本计划 R4 才处理),若 R2 加深融合改变
   allocation 图,memory parity 应前移为每个 R2 分支的门禁而非 R4 专属。

## 对 R0→G1 顺序与四支选择逻辑的评价

顺序正确:R0 是纯卫生闭环,R1 只读归因且自带 kill 纪律(§13),符合"最短证伪"原则。四支
(R2-A Linear 横向融合 / R2-B Conv 深调 / R2-C 图融合 / R2-D runtime/copy)的触发条件都绑定了 R1
实测 share 而非直觉,且每支都预写了物理手段、对照组和禁止外推条款——这在同类路线文档中属于纪律
最严的一档。唯一操作性缺陷是 M-1(缺冻结的 T);补上后,r_required 排序即可机械执行。α-CROWN
恢复(R3)的门禁设计可证伪、逐级、有 kill condition,不复活 C2 的边界写得足够硬。

## 唯一下一动作建议

一个提交完成 R0 全部四项(interval.py cast、C0415 限定、3e-4/1-ULP 披露、steady-state 披露),并在
同一提交内把 R1 预注册草稿中的目标 T 与时钟域校准条款补齐——此后才允许任何测量或实现动作。
