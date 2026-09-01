# FSG4 B4-B1 typed pure-PyTorch reference external audit

- task: fsg4-b4b1-typed-reference-20260818
- doc: fsg4-b4b1-typed-reference-20260818/request
- from: codex -> to: external-auditor
- executor: codex / auditor: external-model
- base commit: 88e0e7a
- created: 2026-08-18T07:52:51Z

## Original request

---
status: ready-for-external-audit
updated: 2026-08-18T16:24:00+08:00
type: audit-handoff
topic: boundflow
stage: s01
---

# FSG4/B4-B1 Typed Pure-PyTorch Reference 外审交接

## 1. 请求判定

请独立审计commit `e62b387b9c92370db92c54f3c5b1e941574a4065`及其B4-B1祖先提交，判定是否同意关闭：

`VALIDATED-B4-B1-TYPED-PYTORCH-REFERENCE-PENDING-EXTERNAL-AUDIT`

批准只允许下一阶段另行预注册B4-B2 CUDA/TIR。不得将本轮解释为TIR已实现、性能/显存提升、
whole-core/query speedup或ASPLOS-ready。

## 2. 审计边界

核心代码：

- `boundflow/ir/differentiable_lower_region.py`；
- `boundflow/runtime/fsg4_b4b1_pytorch_reference.py`；
- `boundflow/runtime/fsg4_b4b1_reference_capture.py`；
- `scripts/run_fsg4_b4b1_pytorch_reference_artifact.py`；
- `scripts/probe_fsg4_b4b1_pytorch_reference_integrity.py`；
- `tests/test_fsg4_b4b1_pytorch_reference.py`。

正式证据：

- capture raw：`artifacts/fsg4-b4b1-reference-five-fresh/resnet2b-prop0-v1`；
- reference v2：`artifacts/fsg4-b4b1-pytorch-reference/resnet2b-prop0-v2`；
- integrity：`artifacts/fsg4-b4b1-pytorch-reference/resnet2b-prop0-v2-integrity-report.json`；
- v1：`artifacts/fsg4-b4b1-pytorch-reference/resnet2b-prop0-v1`，仅为superseded负向历史。

## 3. AC1：Typed IR / Instance 分层

独立核对：

1. 静态IR不携带tensor payload或单次`base_capture_hash`；
2. instance绑定base/reference capture及完整tensor digest；receipt绑定IR/instance/result/target；
3. α direction/spec、feature indices/shape、β location/sign与`-value*split-sign`、lower sign-select、
   intercept reduction、Linear/Conv transpose contraction与bias carry均在合同内；
4. lower-only、dense、single-consumer、default stream、no alias及shape/dtype/layout/presence fail closed；
5. S/P的static IR hash在5 fresh各自唯一：`f5085dde...a08`、`f781e56c...f67`。

## 4. AC2：从raw独立数值重算

不要采信summary数字。直接加载5个PT的10 captures，独立执行或审阅reference，重算：

- capture=10；metric=60；elements=196,380；
- max abs diff=`6.109476089477539e-07`；
- allclose/sign exact=true；
- S native α/β gradients=5/5；
- P native α/incoming-A gradients=5/5；
- P production β empty且gradient absent，不得用zero tensor冒充；
- S production incoming-A无gradient target，forced clone micro与独立eager分解一致。

确认reference只用公开PyTorch算子，不调用TVM或`crown_ibp.py`私有oracle。

## 5. AC3：Deterministic v2 replay

v1首次full出现唯一失败：exact record受前序测试遗留线程数影响。审计需确认：

- v2 protocol冻结threads=1、deterministic algorithms=true、precision=highest、MKLDNN=false；
- runner退出后恢复调用方全局状态；
- 从入口threads=1/4/8重算records完全一致；
- v1被新protocol fail-closed拒绝，未被静默接受；
- v2 manifest=`14923b03...3167`、summary=`becd8ae5...d744`，root replay逐字节通过。

## 6. AC4：完整性与独立攻击

正式probe两案均修改全部5 run并同步重签内部capture、source summary/manifest与derived protocol：

1. incoming lower bias；
2. output lower-A adjoint。

两案旧capture-sufficiency层仍可通过，但numerical reference应`2/2 rejected`。独立确认report=
`6a3192f6...9313`绑定source git、probe hash、完整reference code revision与B4-B1a manifest。

请至少新增一类未注册协调攻击，例如all-run operator-bias value rewrite或derived
IR/instance/receipt联动改写，并说明拒绝发生在身份层还是数值语义层。

## 7. AC5：回归与静态门禁

独立运行并记录：

```bash
source /home/lee/miniconda3/etc/profile.d/conda.sh
conda activate boundflow
python -m pytest -q tests/test_fsg4_b4b1_pytorch_reference.py
python -m pytest -q tests/test_fsg4_b3*.py tests/test_fsg4_b4b*.py
python -m pytest -q
```

声明值：targeted=`23 passed`；related=`131 passed`；full=
`1405 passed, 3 skipped, 6 warnings`。另核Black、scoped Mypy、Pylint 10.00、diff与DocOps。

## 8. AC6：Claim 边界

核对memo、claims map、current status、README、预注册与closure：

- 只主张typed reference correctness/gradient parity；
- v1明确superseded，首次失败未隐瞒；
- `performance_claimed=false`、`tir_admitted=false`全链路保持；
- B4-B2/TIR、性能、显存、whole-core/query与ASPLOS-ready仍关闭。

## 9. 输出格式

请给出：

1. verdict：`approve`或`request_changes`；
2. AC1—AC6逐项PASS/FAIL与独立数字；
3. findings按blocker/major/minor/info分级；
4. 新增攻击的构造、是否全链重签、拒绝点；
5. 不可现场复核项；
6. 是否批准executor关闭exchange，以及批准后是否只开放B4-B2预注册。


## Scope

B4-B1 typed IR, pure-PyTorch forward/VJP, deterministic v2 artifact, coordinated rewrite integrity, regression and claim boundary

## Acceptance criteria

- AC1 typed IR/instance/receipt ownership and fail-closed contracts independently pass
- AC2 five-fresh raw recomputation reproduces 10 captures, 60 metrics, 196380 elements, max diff and sign gates
- AC3 deterministic v2 replays across inherited thread policies and restores global state; v1 fails closed
- AC4 two registered all-run full-resign attacks plus at least one independent attack are rejected at an explained layer
- AC5 targeted, related, full regression, static, diff and DocOps gates pass
- AC6 no B4-B2 TIR performance memory whole-query or ASPLOS-ready claim drift
