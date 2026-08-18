# FSG4 B4-A formal timing NO-GO audit

- task: fsg4-b4a-formal-timing-20260818
- doc: fsg4-b4a-formal-timing-20260818/request
- from: codex -> to: external-model
- executor: codex / auditor: external-model
- base commit: adc175b
- created: 2026-08-18T03:37:19Z

## Original request

# FSG4/B4-A 正式计时外部审计交接

## 1. 审计目标

请不要采信 closure、summary 或 executor 的聚合数字。请从 v5 formal raw 独立判断 B4-A 是否应关闭为
`VALIDATED-NO-GO-B4-A-PERFORMANCE`，并确认 B4-A 只保留 correctness/mechanism evidence、不能累计为
B4 performance candidate。

审计对象：

- artifact source：`46a8493557c49f327df4e70d7cdd7649227b14b9`；
- executor closure commit：`d387a7c`；
- artifact：`artifacts/fsg4-b4a-formal-timing/resnet2b-prop0-v5/`；
- tamper report：`artifacts/fsg4-b4a-formal-timing/resnet2b-prop0-v5-tamper-report.json`；
- runner：`scripts/run_fsg4_b4a_formal_timing.py`；
- worker：`scripts/run_fsg4_b4a_same_solver_worker.py`；
- base environment worker：`scripts/run_fsg3_same_solver_timing.py`；
- prereg：`gemini_doc/BOUNDFLOW_FSG4_B4A_FORMAL_TIMING_PLAN_2026_08_18.md`；
- closure：`gemini_doc/change_2026-08-18_fsg4_b4a_formal_timing_internal_closure.md`。

## 2. Acceptance criteria

### AC1 — Source、protocol 与 raw identity

- 独立核对 source/code revision 19 文件、five-fresh manifest、模型/property、三个外部仓库 commit；
- 独立重算 manifest/protocol/summary/file SHA256/hash链，确认24个worker与所有派生文件完整；
- 检查artifact与tamper report无`/home/`、`/tmp/`路径泄漏。

### AC2 — Fresh sequence 与环境合同

- 6 block、24 fresh worker顺序与protocol逐元素一致，control/profile各自角色不混用；
- 24/24 raw均满足AC power、同一GPU/runtime、无external compute process与worker overlap；
- 逐worker核对preflight绑定`nvidia_powerd=inactive`与`enforced.power.limit=55.0 W`；
- 从`environment_before/after`独立重算thermal/power counter区间delta，确认coupled/independent/admitted
  投影；不得以生命周期累计值相等代替区间增量相等。

### AC3 — Correctness 与 activation

- 从raw独立比较6个control pair的solver discrete/final lower及19个terminal export tensor；
- 独立重算max abs diff、sign exact与`atol=rtol=2e-4`门禁；
- B3/B4-A handoff/rerun分别为`0/1`和`1/0`，B4-A lineage=`6`；
- profile raw的forward=`4`、optimizer bound evaluation=`10`、optimizer
  trace/evaluation/update=`1/10/9`、provider/fallback=`0/0`。

### AC4 — 性能分类独立重算

- 只使用6个control pair，独立从worker raw重算core/query wall/GPU及peak allocated/reserved逐pair ratio、
  geomean、min/max；不得使用profile latency形成headline；
- 判断core wall geomean是否通过`>=1.03x`，六个query wall pair是否全部通过`>=0.98x`；
- 检查`performance_candidate_admitted=false`与`validated-no-go-b4a-performance`是否是冻结规则唯一结果；
- 不得因接近阈值而修改门禁、剔除pair或重跑挑样。

### AC5 — Profile attribution 与 claim 边界

- 独立核对12个profile worker closure/residual均过`1%/3%`；
- 从profile raw重算B3/B4-A各span mean，确认terminal backward/export局部收益与其他span波动；
- 局部span、kernel或profile数据不得升级为whole-core speedup；memory ratio=`1.0`不得声称显存收益；
- `kernel_launch_delta=DEFERRED-TO-B4-A-KERNEL-DELTA`与`performance_claimed=false`保持。

### AC6 — Replay 与 outer-resigned tamper

- 独立运行root replay并核对stdout/summary hash；
- 重跑14类tamper probe，或逐项确认其修改payload、同步重签manifest file digest与manifest hash后仍被
  semantic replay拒绝；特别检查environment-counter-delta与power-policy攻击不是只靠外层digest拒绝。

### AC7 — Regression、DocOps 与后续路线

- fixed related 11文件=`73 passed`，full=`1356 passed, 3 skipped`；
- Black/Mypy/Pylint 10.00/diff、`dol exchange validate`、`dol lint --soft`通过；
- 若批准，只同意B4-A以validated mechanism / NO-GO performance关闭；B4-A不得进入累计性能基线；
- B4-B/TIR在本exchange关闭前不得启动。关闭后是否开放B4-B，只能依据已外审B4-0的67.72%
  opportunity和B4总路线另行决定，不能把B4-A约1.9%收益计入累计candidate。

## 3. 建议独立命令

```bash
conda run -n boundflow python scripts/run_fsg4_b4a_formal_timing.py replay \
  --artifact-dir artifacts/fsg4-b4a-formal-timing/resnet2b-prop0-v5

conda run -n boundflow python scripts/probe_fsg4_b4a_formal_timing_tamper.py \
  --artifact-dir artifacts/fsg4-b4a-formal-timing/resnet2b-prop0-v5 \
  --report /tmp/fsg4-b4a-audit-tamper.json

conda run -n boundflow python -m pytest -q \
  tests/test_fsg3_same_solver_worker.py \
  tests/test_fsg4_b4a_terminal_lower_adjoint_handoff.py \
  tests/test_fsg4_b4a_correctness_pairs.py \
  tests/test_fsg4_b4a_correctness_pairs_artifact.py \
  tests/test_fsg4_b4a_formal_timing.py \
  tests/test_fsg4_b4a_formal_timing_tamper.py \
  tests/test_fsg4_b4a_formal_timing_artifact.py \
  tests/test_fsg4_b3_explicit_counters.py \
  tests/test_fsg4_b3_same_solver_timing.py \
  tests/test_fsg4_b3_same_solver_worker.py \
  tests/test_fsg4_b3_same_solver_artifact.py
```

性能与语义数字请优先用 Python 标准库直接解析 raw，不要调用被审计方 summary helper。

## 4. 审计输出格式

请按 AC1—AC7 逐项给出 `PASS/FAIL`、独立证据和重算数字；findings 分
`blocker/major/minor/info`。最终 verdict 只能为 `approve`、`approve-with-findings` 或 `reject`。若批准，
请明确是否同意关闭 `VALIDATED-NO-GO-B4-A-PERFORMANCE`，以及下一步是否仅允许另行预注册 B4-B。



## Scope

B4-A formal raw correctness environment performance classification replay tamper and route boundary

## Acceptance criteria

- AC1 source protocol raw identity
- AC2 fresh sequence and environment contract
- AC3 correctness and activation
- AC4 performance classification
- AC5 profile attribution and claim boundary
- AC6 replay and outer-resigned tamper
- AC7 regression DocOps and next-route boundary
