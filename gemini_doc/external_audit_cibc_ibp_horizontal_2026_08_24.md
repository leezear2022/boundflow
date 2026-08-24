# CIBC IBP Conv 横向融合正式关闭 — 独立外部审计报告(Round 1)

- task: cibc-ibp-horizontal-20260824 / r001
- auditor: external-model(独立子代理,未参与实现)
- 审计对象: scope baa4503..b94e61a + artifacts/cibc-ibp-horizontal-formal
- 审计时间: 2026-08-24
- 审计环境: 同机 RTX 4060 Laptop(sm_89)、conda env boundflow、Python 3.12.12、torch 2.12.1+cu132

## 总体 Verdict

**APPROVE** — 同意 `VALIDATED-REDUCED-CIBC-IBP-CONV-HORIZONTAL` 关闭。

所有公开数字均从 9 份 raw 记录独立重算并逐位一致;protocol/代码/提交链哈希全部穿透核验;
baseline 公平性穿透检查通过(baseline 即 BoundFlow production 四-Conv 公式逐字复现,两侧同为
CUDA Graph 且均计入输入 copy);10/10 全重签 tamper 独立重跑全部拒绝,另自建 3 类变体(第 11–13
类)亦全部拒绝;float64 oracle 对全部 6 个 production Conv 独立抽查通过。无 blocker、无 major。
minor 2 项、info 4 项,均不影响 claim 成立,见 Findings。

## AC1 — protocol 冻结与提交链: PASS

- HEAD=95cc8f8==origin/feat/rvir-v4-production-state-ownership-v1;提交链
  f4f6b39(B4 frontier 关闭 03:22)→ baa4503(CIBC feat 03:37)→ a52b177(freeze 03:47)→
  b94e61a(validate 04:04)→ 95cc8f8(deliver exchange),顺序成立。
- `git diff a52b177..b94e61a` 不含任何 boundflow/ 代码或 worker/artifact 脚本改动(仅 artifact
  JSON、docs、tamper probe 新增):门禁与 schedule 集合在正式运行前冻结,无事后挪门禁。
- protocol.json 的 `code_revision` 8 个文件 SHA256 与当前 worktree、a52b177 blob、b94e61a blob
  三方逐一比对全部一致(独立脚本重算)。
- manifest 内 11 个文件哈希、protocol_hash、summary_hash、9 份 raw 的 worker_hash 全部独立重算一致。
- source capture SHA256(f42229dd…)在 artifacts 内多个副本核验一致;model(resnet_2b.onnx)
  SHA256=791aa24d… 与 VNN-COMP 2021 固定 checkout 实文件一致(fetch 脚本钉死 commit 90419aad)。
- 门禁值(operator geomean≥2.0/worst≥1.2、model geomean≥1.5/worst≥1.2、semantic atol/rtol=3e-4)
  随 a52b177 的 artifact 脚本首次引入且此后未变;protocol changelog 文档自 a52b177 起无后续修改
  (`git log --follow` 仅 1 次提交)。
- 关于 3e-4 vs 项目既往 2e-4:本任务 protocol 在正式运行前即冻结 3e-4(a52b177,先于 b94e61a 运行),
  非事后放宽;且实测整图 max diff=0.000244140625=2^-12,恰为 float32 在量级 [16,32) 的 1 ULP,
  属舍入形式差异的物理下限量级(见 AC2 oracle 证据)。记录为 info。

## AC2 — 语义等价: PASS

- 9 份 raw 中全部 30 组/算子字段独立重算:6 个算子 × 3 schedule 的
  maximum_absolute_difference ∈ [4.77e-7, 1.72e-5],sign_exact 全 true;6 个 model worker 的
  中间+最终 max diff 均为 2.4414e-4 < 3e-4,sign exact。
- 独立 float64 oracle(自写脚本 /tmp/audit_cibc_f64_oracle.py,加载同一 capture+ONNX,逐 Conv
  以 float64 精确区间卷积为参照):6/6 Conv 的 fused 输出对 oracle 的误差(6.7e-7–1.2e-5)与
  float32 baseline 对 oracle 的误差(2.4e-7–1.2e-5)同量级;fused-vs-baseline32 差异与 raw 报告值
  逐一吻合;oracle 下 lo≤up 成立;fused 输出符号与 float64 oracle 完全一致。
- TIR 数学形式核对:center/deviation 公式在 w≥0/w<0 两支分别退化为 l·w/u·w,与 baseline
  正负拆分公式代数等价,差异仅来自浮点舍入顺序。

## AC3 — 算子级性能: PASS

- 从全部 3×6×30 组 raw 独立重算:schedule geomean 64/128/256 = 11.672159/12.795108/12.725013,
  与 summary 逐位一致;argmax 选中 128;6 算子 speedup=[9.1423, 14.6166, 11.5923, 22.6715,
  11.2109, 11.1448],geomean=12.795107698179335、worst=9.14229089216829,与 summary/closure 一致。
- schedule 候选集(64/128/256,argparse choices 硬约束)与选择规则(6 算子 geomean argmax)在
  a52b177 冻结,先于任何正式计时;每个候选由独立 fresh 进程测量,顺序 BC/CB/BC 与 protocol 一致。
- warmup=20 次/侧、每组 500 次、30 组取 median,median 对组内首组慢启动漂移稳健(已核组级数据:
  首组 baseline 离群 ~0.0636ms vs 稳态 ~0.048ms,median 不受影响);launch_count=15021=
  1+20+30×500,无静默跳过;无 OOM/fallback 记录路径(fail-closed 设计,context 下不支持 shape
  直接 raise)。
- 计时方式为 wall-clock(perf_counter)+同步,两侧对称;非 CUDA event,但 loop 吞吐量测量对
  两侧公平,且 protocol 未规定 event 计时。
- baseline 公平性穿透:worker 的 baseline_call(weight clamp 正负拆分+4 次 F.conv2d+2 次 bias
  broadcast-add)与 production boundflow/domains/interval.py 的 conv2d 分支逐语句一致,即被替换的
  真实 production 公式,非故意慢化;两侧各自 20 次 warmup,编译/加载发生在计时区外(见 info-3)。
- 独立物理重跑(本机同时刻重跑 threads=128 operator worker 至 /tmp):见末节"物理重跑"数据,
  6 算子 median 与正式 raw 偏差在 ±6% 以内,geomean 复现。
- info:算子级 baseline 每 call ~48µs 已接近 CPU launch-bound 区(6 算子 baseline 之和 ~0.29ms
  超过整图 CUDA-Graph baseline 0.176ms),算子级 12.8x 主要反映 eager 路径 kernel 数量与 launch
  开销的消除;有系统意义的数字是整图 2.456x。closure 文档已明确禁止把 22.67x 写成整图数字,
  claim 边界未漂移。

## AC4 — 整图级: PASS

- 6 份 model raw 独立重算:median=[0.175928/0.071569, 0.175088/0.071252, 0.176010/0.071814,
  0.176809/0.071858, 0.176808/0.072015, 0.176560/0.071895] ms,speedup=[2.45815, 2.45731,
  2.45091, 2.46054, 2.45517, 2.45580];geomean=2.456310282102286、worst=2.4509075978286576、
  bootstrap(seed=20260824, 10000 次, 2.5 分位)lower=2.4538553447313016,全部与 summary 逐位一致。
- 输入 copy 计入:replay() 每次先 copy_ lower/upper 入静态 buffer 再 graph.replay(),计时 lambda
  包裹完整 replay → copy 在计时区内;raw 字段 input_copy_included=true 与代码行为一致。
- CUDA Graph parity:baseline(threads=None,走 interval.py 四-Conv)与 candidate 使用同一
  CIBCIBPCUDAGraphPlanV1 捕获,两侧同 warmup(构造期 3 次+计时前 20 次)、同输入、同 copy 语义。
- 6/6 覆盖:candidate.launch_count==6 在 plan 构造时强制(≠6 即 raise),6 份 raw conv_coverage
  全为 6;2 个 Linear 与 ReLU/add/flatten 保持同图未融合,已披露。
- 组级 sanity:每 worker speedup 组内范围 [1.97, 2.94],median 稳健;BC/CB 交替无系统性顺序偏差
  (CB worker 的 speedup 不低于 BC)。

## AC5 — replay 与 tamper: PASS

- root replay 独立重跑退出 0;replay 从 9 份 raw 重算 median/winner/geomean/bootstrap/semantic/
  coverage 后与 summary 全字段比对(含 summary_hash),为真语义重算,非仅哈希校验。
- tamper probe 独立重跑至 /tmp:10/10 拒绝,且生成 report 与仓内 tamper_report.json 逐字节语义
  一致(report_hash 匹配)。逐类核对了 probe 代码:每类 mutate 后对 worker_hash/protocol_hash/
  summary_hash/manifest 全部外层重签,拒绝全部来自语义交叉校验而非哈希失配。
- 自建第 11–13 类变体(篡改 raw 单组 baseline 计时并全重签;翻转 operator sign_exact 并重签;
  对换 operator_64/operator_256 文件)均 fail-closed 拒绝,拒绝原因分别为 timing group
  derivation / worker semantic derivation / worker identity。

## AC6 — 测试与静态检查: PASS(附 minor)

- 全量 pytest 独立重跑:1492 passed, 3 skipped, 6 warnings,652.5s — 与交付一致。
  `-rs` 核对 3 个 skip:1 个为 TVM 可用时跳过 no-TVM smoke(防重复编译,文档化);2 个为
  NRIR-43/44 replay 测试依赖 /tmp/boundflow-vnncomp2021-nrir43 冻结 checkout(环境边界,
  与本 claim 无关)。
- targeted 三文件共 7 个测试与交付 "targeted=7" 吻合。
- black --check 10 个触及文件全部通过;git diff --check baa4503~1..HEAD 干净。
- dol lint --soft 通过;dol validate 报 3 个 duplicate event id,经比对为 HEAD 既有的
  ev.jsonl 历史重复(ev009180/ev009388/ev010862),非本轮引入(见 minor-2)。
- mypy:新引入的 3 个模块与 3 个脚本全部 `Success: no issues found`;但触及的
  boundflow/domains/interval.py 存在 11 个 mypy 错误,其中 3 个(arg-type,stride/padding/
  dilation 的 tuple[int,...] vs tuple[int,int],interval.py:83-85)为本次改动新引入,8 个
  attr-defined 为既有。pylint 新模块 10.00/10,interval.py 7.01/10(新增 1 条 C0415
  import-outside-toplevel,其余多为既有)。delivery 的 "Black/Mypy/Pylint=10.00/10" 只对
  新文件成立,对触及的 interval.py 不成立 — 记 minor-1。

## AC7 — claim 边界: PASS

- 四处一致:closure 文档、asplos_claims_map.md(L4-8)、asplos_execution_memo_v1_0.md
  (L2148-2152)、current_status_after_pr13.md(L3-8)均把 claim 限定为 RTX 4060/sm_89 +
  ResNet2B prop0 + IBP + 相对 BoundFlow 自身四-Conv baseline;均显式排除 auto_LiRPA、
  alpha-CROWN/BaB/query、memory、跨模型与 ASPLOS-ready。
- production 默认路径未改:interval.py 仅在显式 context 下走 fused 分支,无 context 时
  `execute_active_cibc_ibp_conv_v1` 返回 None 回落原公式。
- 披露保留:2 个 Linear 与非 affine op 未融合、CUDA Graph private-pool memory 未比较,均在
  delivery known-limitations/risks 与 closure "Remaining Boundary" 中。
- 未发现任何文档把 22.67x 单算子峰值写成整图数字,未发现相对 auto_LiRPA 的暗示。

## B4 kill gate 合规: PASS

- B4-C1:geomean=0.9481500115566288x(≈0.948x),closure 文档
  BOUNDFLOW_FSG4_B4C1_PROVIDER_OWNED_LOWER_FORMAL_CLOSURE_2026_08_24.md 判定
  VALIDATED-NO-GO,依据为预注册 no-regression(<1.0 即关闭,C1 changelog L51 预注册)。
- B4-C2:三 fresh worker paired speedup=[0.348761, 0.337448, 0.346003](0.337–0.349x)、
  peak allocated ratio=1.3401085408885496(1.34x),按预注册 kill gate 关闭,未进 6-fresh/B4-D;
  B4-D 因上游 kill gate 未开工而正确关闭。
- git 顺序:f4f6b39(B4 frontier 关闭)早于 baa4503(CIBC 开工),"先完成原 B4 再进 CIBC"成立;
  memo L2130-2133 与 B4/CIBC 最终状态文档一致声明 CIBC 不继承 B4 claim。

## Findings

- minor-1 | boundflow/domains/interval.py:83-85 | mypy 3 条新 arg-type 错误 + pylint 新增 C0415;
  delivery "Black/Mypy/Pylint=10.00/10" 对触及的 interval.py 不成立(新文件确为 10.00)|
  建议:为 stride/padding/dilation 加 cast(tuple[int,int], ...) 并在文件头补 pylint disable,
  或在交付文档中把 lint 声明范围限定为新文件。
- minor-2 | .docops/ev.jsonl | dol validate 报 3 个 duplicate event id(ev009180/ev009388/
  ev010862),为 HEAD 既有历史问题,非本轮引入;lint --soft 通过 | 建议:后续独立任务去重,
  不阻塞本轮。
- info-1 | protocol semantic_atol=3e-4 | 项目既往阶段多用 2e-4;本轮 3e-4 在运行前冻结且实测
  diff=2^-12 恰为 float32 单 ULP 量级,2e-4 对该量级中间值不具备可达性 | 建议:在 protocol
  changelog 补一句量级依据,便于后续审计。
- info-2 | scripts/run_cibc_ibp_horizontal_worker.py | 算子级 wall-clock 计时下 baseline 已接近
  CPU launch-bound;算子级 speedup 主要刻画 eager kernel 数量/launch 开销消除 | 建议:未来扩大
  claim 时以整图 CUDA-Graph 数字为准(现文档已如此表述)。
- info-3 | 算子/整图计时均不含 TIR 编译与 plan 构造(编译在 warmup 前完成),closure 文档未显式
  披露编译成本排除 | 建议:在 closure 文档补一行说明 steady-state 口径。
- info-4 | protocol/summary 中 performance_claimed=false 与 docs 的 VALIDATED-REDUCED claim 的
  关系:artifact 自身不携带 claim,claim 仅存在于文档层,语义自洽(正式运行前禁 claim)|
  无需行动。

## 不可现场复核项

- 正式运行时刻(2026-08-24 04:04 前后)的机器负载/温度等物理状态无法回溯;以 6+3 fresh 进程、
  BC/CB 交替、30 组 median 与本人独立物理重跑的一致性作为替代证据。
- protocol changelog front-matter 的 updated=2026-08-24T12:40+08:00 晚于其引入提交 a52b177
  (03:47+08:00)的 commit 时间,疑为文档时间戳填写口径问题;文件内容自引入后无修改,不影响
  冻结效力,仅作记录。

## 物理重跑(独立复核)

在本机 GPU 空闲时段,用同一冻结 capture+ONNX 独立重跑两类 worker(输出至 /tmp,不触碰仓库):

- operator worker(threads=128, order=CB, 含全部 6 Conv、独立重编译):逐算子 speedup=
  [9.040, 14.605, 11.346, 22.927, 11.281, 11.280],geomean=12.7865、worst=9.0398;与正式 raw
  (12.7951/9.1423)偏差 ≤1.2%;逐算子 maxdiff 与正式 raw 完全一致(如 op10=1.717e-5)、
  sign exact。
- model worker(ordinal 0, BC, threads=128):baseline 0.172238ms / candidate 0.070062ms,
  speedup=2.45837 vs 正式 2.45815(偏差 0.01%);maxdiff=2.4414e-4、sign exact、coverage=6。

结论:正式 raw 数字在当前时刻可物理复现,非历史环境偶然产物。

## 关键命令与输出摘录

- 独立重算统计量: `python3`(自写脚本,见审计会话)→ schedule geomeans
  {64: 11.672159, 128: 12.795108, 256: 12.725013};operator geomean 12.795107698179335 /
  worst 9.14229089216829;model geomean 2.456310282102286 / bootstrap lower
  2.4538553447313016 / worst 2.4509075978286576;全部 match summary: True。
- `python scripts/run_cibc_ibp_horizontal_artifact.py --artifact ... --replay` → exit 0,
  输出 summary 与仓内一致。
- `python scripts/probe_cibc_ibp_horizontal_tamper.py --artifact ... --output /tmp/...` →
  exit 0,10/10 rejected,与仓内 tamper_report.json 一致;自建 3 变体全拒。
- `pytest tests -q -rs` → `1492 passed, 3 skipped, 6 warnings in 652.51s`。
- `black --check`(10 文件)→ all unchanged;`git diff --check` → clean;
  `dol lint --soft` → ok;`dol validate` → 3 个 HEAD 既有 duplicate event id。
- `mypy` 新模块/脚本全 Success;`mypy boundflow/domains/interval.py` → 11 errors
  (3 新引入 arg-type + 8 既有 attr-defined);`pylint boundflow.domains.interval` → 7.01/10,
  新模块 10.00/10。
- float64 oracle(/tmp/audit_cibc_f64_oracle.py)→ 6/6 Conv fused-vs-oracle 误差与 baseline32
  同量级,sign 与 oracle 一致,lo≤up 成立。
- 物理重跑:operator threads=128 → geomean 12.7865/worst 9.0398;model ordinal 0 → 2.45837。

## 结论

同意关闭 VALIDATED-REDUCED-CIBC-IBP-CONV-HORIZONTAL。claim 边界无漂移,数字全部从 raw
独立重算一致,baseline 公平性穿透检查通过,完整性机制(replay/tamper)经独立重跑与自建变体
验证为真 fail-closed。
