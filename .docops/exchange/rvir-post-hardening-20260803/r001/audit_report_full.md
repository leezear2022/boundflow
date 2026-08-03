# RVIR 审计后加固外部复审报告(完整版)

- task: rvir-post-hardening-20260803 / round r001
- auditor: external-model(独立复核,未采信执行方数字)
- 审计基线:`b01225b3ffa7a4bfc6fcdfe42ab9fe0973d631ae`
- 待审结果:`main@1a6eb65`;HEAD `ffe0071` 多出的 3 个提交经 `git diff --stat 1a6eb65..ffe0071`
  确认为纯 docs/DocOps 同步(9 个文件,+663/-6,无代码变化)
- 审计时间:2026-08-03
- 环境:conda env `boundflow`,`/home/lee/miniconda3/envs/boundflow/bin/python`,CPU 机器(无 CUDA)

## 1. 总体 verdict

**approve**(无 blocker / major finding;1 条 info 级观察,见 §4)

PR #5—#8 对原审计五个 minor finding 的加固全部闭环,未引入 claim drift,未产生回归。
RVIR 总体状态保持 VALIDATED-REDUCED(CPU correctness/integration),IR-5 保持
VALIDATED-NO-GO。本结论不升级为 performance、CUDA、fused-kernel、完整 verifier E2E
或 ASPLOS-ready 结论。

## 2. AC1—AC6 逐项结论

### AC1(RVIR-2 拒绝路径已固化)— PASS,F1 closed

阅读 `tests/test_real_verifier_ir_integration.py`(210 行):

- 四条专用负向测试存在且均用 `pytest.raises(ValueError, match=...)` 断言具体错误,
  非任意异常:
  - `test_external_call_rejects_undeclared_backend_before_execution`(:131)
    match `"backend differs"`(来自 `boundflow/ir/task_v1.py:407`);
  - `test_external_schedule_rejects_missing_emit`(:157)
    match `"result emission|free order"`(`boundflow/ir/schedule.py:952/956`);
  - `test_external_schedule_rejects_duplicate_launch`(:173)
    match `"unknown or duplicated|more than once"`(`boundflow/ir/schedule.py:903`,
    `boundflow/ir/task_v1.py:443`);
  - `test_external_call_rejects_local_semantics_ownership`(:197)
    match `"retain external ownership"`(`boundflow/ir/bound.py:766`)。
- backend mismatch 在 exact call 前拒绝:`execute_external_verifier_call`
  (`boundflow/runtime/verifier_ir_integration.py:338-355`)第 344 行先
  `compilation.validate()`、第 346 行再检查 `reference_implementation_id`,
  `exact_call()` 在第 348 行;测试同时用 `called` 标志断言 `assert not called`(:154)。
- 正向路径仍恰一次 launch/emit:`test_activation_query_compiles_through_all_ir_layers`
  断言 `len(launches) == len(emits) == 1`(:104)。
- 本轮范围 `git log b01225b..1a6eb65 -- boundflow/` 为空:生产代码未改动,本轮只加测试,
  门禁逻辑来自既有实现,测试是对既有拒绝路径的固化而非新造。

亲自运行:

```
$ python -m pytest -q tests/test_real_verifier_ir_integration.py
......                                                             [100%]
6 passed in 1.02s
```

### AC2(Phase6H Python 路由不依赖调用者 PATH)— PASS,F2 closed

阅读 `scripts/run_phase6h_artifact.sh:23-35`:解释器选择顺序为
`PHASE6H_PYTHON` → `${CONDA_PREFIX}/bin/python`(需可执行)→ `python`;
`command -v` 检查(:31-35)在 `mkdir -p "${OUT_DIR}"`(:37)之前,fail closed。
`tests/test_phase6h_artifact_runner_smoke.py` 两条测试分别覆盖受限 PATH + override
与缺失解释器退出码 2。

亲自实测(四条全部独立运行,非只跑测试):

- (a) `PATH=/usr/bin:/bin PHASE6H_PYTHON=<conda python> bash scripts/run_phase6h_artifact.sh .tmp/audit-ac2/a`
  → 退出码 0,artifact 完整(jsonl/csv/summary.md/env.txt/pip_freeze/conda_list/figs),
  `env.txt` 记录 `python: /home/lee/miniconda3/envs/boundflow/bin/python`、
  `torch_version: 2.12.1+cu132`;
- (b) 无 override、`CONDA_PREFIX=/home/lee/miniconda3/envs/boundflow`
  → 退出码 0,`env.txt` 选中 `${CONDA_PREFIX}/bin/python`;
- (c) `PHASE6H_PYTHON=/nonexistent/missing-python` → 退出码 **2**,
  stderr `[phase6h] error: Python interpreter not found: ...`,输出目录未创建;
- (d) `PHASE6H_PYTHON=/usr/bin/python3`(该解释器无 torch,已直接验证
  `ModuleNotFoundError: No module named 'torch'`)→ 退出码 **1**,
  torch 缺失的 traceback 原样出现在输出中,未被掩盖为通过;
- 测试文件本身:`2 passed in 4.41s`。

### AC3(DocOps exchange 不可变性)— PASS,F3 closed(协议保持)

- `git log --oneline b01225b..1a6eb65 -- .docops/exchange/rvir-20260803` 为空:
  旧 task 在本轮范围零改动;旧 r001 五件(audit/audit_response/delivery × .md/.json)在位。
- 新复审使用全新 task 目录,未覆盖旧 round。
- 亲自运行:
  - `dol exchange validate rvir-20260803` → `{"ok":true,"errors":[],"tasks":1}`(exit 0);
  - `dol exchange validate rvir-post-hardening-20260803` → `{"ok":true,...}`(exit 0);
  - `dol lint --soft` → `{"ok":true,...,"why":"ok","soft":true}`(exit 0)。

### AC4(v2 在线原始证据可独立语义重放)— PASS,F4 closed

独立解析(自己写脚本,未用执行方数字):

- `online_queries.jsonl` 377 行、`online_typed_ir.jsonl` 377 行;
- query_id 集合一致、无重复;sequence_number 映射两文件完全相同、连续且唯一;
- root/parent = **30/347**;全部 347 条 child 的 parent sequence 均小于自身;
- `online_execution.json`:`query_count=377, completed=377, root_query_count=30,
  parent_link_count=347, semantics_owner=external_verifier, performance_claimed=false`;
- manifest `files` 五个 SHA256 由我独立重算,全部 match。

replay 语义门禁(阅读 `scripts/run_real_verifier_ir_artifact.py` 确认):

- `replay_artifact`(:433-461)先验全部文件 digest(:446-448),再对 394 条历史 admission
  逐行重新编译并整行比较(:452-458);
- v2 追加 `_replay_online_artifact`(:272-313)→ `_online_row_summary(recompile=True)`:
  对 377 条 query 逐条 `compile_external_verifier_call` 重编译五层 IR,并逐行比较
  `ir_hashes`(:180-185);`hashes()` 覆盖 bound_module/plan_template/plan_instance/
  task_module/schedule 五层(`verifier_ir_integration.py:265-277`);parent 先于 child
  从原始 query 重算(:160-164),不信任摘要;
- tamper 测试 `test_v2_replay_rejects_rehashed_online_ir_record_tamper` /
  `test_v2_replay_rejects_rehashed_parent_order_tamper`:重写 payload **并同步更新
  manifest 与内嵌 source digest** 后,replay 仍分别以
  "online typed-IR replay mismatch at row 0" / "online query parent does not precede
  its child" 失败——失败来自语义重算而非旧 digest。

亲自运行:

```
$ python scripts/run_real_verifier_ir_artifact.py replay --artifact-dir artifacts/rvir/rvir-cpu-correctness-v2-20260803
{"activation_call_count":394,"performance_claimed":false,"status":"replayed"}  (exit 0)
$ python scripts/run_real_verifier_ir_artifact.py replay --artifact-dir artifacts/rvir/rvir-cpu-correctness-v1-20260803
{"activation_call_count":394,"performance_claimed":false,"status":"replayed"}  (exit 0)
$ python -m pytest -q tests/test_real_verifier_ir_artifact.py
....                                                                     [100%]
4 passed in 27.26s
```

- v1 artifact 在本轮范围 `git log b01225b..1a6eb65 -- artifacts/rvir/rvir-cpu-correctness-v1-20260803`
  为空,未被修改且仍可 replay;
- 历史 fused replacement coverage `0/394` 未被改写:`asplos_claims_map.md` L13/L408/L423、
  `real_verifier_ir_integration_closure_2026_08_03.md` §4、memo L319/L432 等处一致保留,
  394 条 activation_calls 的 `effective_method` 全部为 `alpha_beta_crown`、无 fused 标记。

### AC5(ResNet 原始数值在固定 external 环境重跑)— PASS,F5 closed

在 `mktemp -d`(/tmp/rvir-audit-ac5.TkvakQ,未触碰 vendored submodule)独立重建环境:

- αβ-CROWN clone 后 checkout `e5c7e17bf0488843acb77b7519f59876717a49f4` ✓;
- `auto_LiRPA` submodule HEAD = `5a098e8f9fb5786a428a024981d833d303921f2d` ✓;
- vnncomp2021 sparse checkout `90419aadcf06cf543ce5c1706cae1059dc9fa6cf` ✓;
- 两个 checkout `git status --porcelain` 均为空(无 tracked 修改);
- sha256sum:`resnet_2b.onnx` = `791aa24d…4a6d` ✓;`prop_0_eps_0.008.vnnlib` =
  `89edf06…3769ff` ✓(均与 `change_2026-08-03_rvir_resnet_raw_rerun.md` §2 一致)。

fresh CPU 重跑(`--warmup 0 --repeats 1 --backends pytorch_eager`)退出码 0,
`status=ok`。独立读 `manifest.json` 与 `payload.pt`(torch.load + 自算 SHA256):

- 12 个冻结字段与 `artifacts/rvir/rvir-cpu-correctness-v2-20260803/resnet_semantics.json`
  逐项比较:**NONE mismatch**(abcrown_commit、model_sha256、vnnlib_sha256、device=cpu、
  bound_count=6、source=external_verifier、intermediate_bounds_hash=
  `d51615b04dfb205afd67d2c21680ece4ca92f693157da1e32c7f8202a8e08cf1`、
  slope=adaptive、allclose=True、max_abs_diff=`3.0994415283203125e-06`、sign 9/9);
- 8 个非空 tensor 的逐字节 SHA256 与变更文档 §5 表格**全部一致**
  (input_lower `206e69cd…`、input_upper `9768a85f…`、linear_spec_c `02c50b97…`、
  external_lower `e03cb7a8…`、boundflow_pytorch_eager_lower `ebba8a73…`、
  boundflow_pytorch_eager_upper `899d27e6…`、boundflow_nominal_output `798343a8…`、
  onnx_nominal_output `ebe5063f…`);`external_upper=None` ✓;
- 本次只作 correctness 重放,不引用任何 timing。

### AC6(总体回归与 claim boundary)— PASS

亲自运行:

- targeted 三文件:`12 passed in 34.80s`(与执行方声称的 12 passed 一致);
- 全量:`460 passed, 37 skipped, 5 warnings in 85.97s`(exit 0);`-rs` 全量列出 skip:
  36 条为 CUDA not available/required,1 条为 "TVM is available; skip allow-no-tvm smoke"
  ——全部为 CUDA/环境边界,无一条被计作 CUDA 验证;
- `black --check` 4 文件:`4 files would be left unchanged`(exit 0);
- `mypy` 2 文件:`Success: no issues found in 2 source files`;
- `pylint` 4 文件:`10.00/10`;
- claim drift 检查:`git diff b01225b..1a6eb65` 中
  `asplos_claims_map.md`、`current_status_after_pr13.md`、closure 文档及 4 篇新增
  change 文档的增量全部为"强化可审计性、保留限制"的措辞(如"不把 CPU correctness
  升级为 CUDA 或 performance claim"、"不形成 CUDA、latency、throughput 或
  ASPLOS-ready claim");未发现 performance/GPU/BoundFlow 本地 αβ kernel/
  完整 VNN-COMP E2E/ASPLOS-ready 的主张漂移;
- request §6 八条限制在 closure §4 与 claims map 中原样保留;IR-5 仍为
  VALIDATED-NO-GO(current_status L6/L25/L36/L51,claims map L12),ASPLOS-ready 仍为 NO。

## 3. F1—F5 状态

| finding | 状态 | 依据 |
|---|---|---|
| F1 / M1 | **closed** | AC1:四条负向测试存在、断言具体错误、exact call 前拒绝,6 passed |
| F2 / M2 | **closed** | AC2:四条场景全部实测通过,fail closed,错误不掩盖 |
| F3 / M3 | **closed** | AC3:旧 exchange 零改动,两个 validate + lint 全 PASS |
| F4 / M4 | **closed** | AC4:377/377 原文 + 语义 replay + tamper fail-closed + v1 不可变 |
| F5 / M5 | **closed** | AC5:固定环境独立重建,12 字段与 8 tensor digest 全部复现 |

## 4. Findings

无 blocker / major / minor。

- info | `scripts/run_phase6h_artifact.sh:28` |
  当 `PHASE6H_PYTHON` 与 `CONDA_PREFIX` 均未设置时回退到 PATH 中的 `python`,
  此时仍依赖调用者 PATH;且 runner 依赖 env.sh 注入的 PYTHONPATH(完全 strip 环境后
  sweep 无法 import boundflow)。属文档化行为,非缺陷 |
  无需处理;如追求彻底自包含可在 runner 内自检 `import boundflow` 并给出友好报错。

## 5. 不可现场复核项

无。AC1—AC6 全部在现场独立完成,包括需要网络的 AC5(clone + 固定版本 + digest 核对 +
fresh CPU 重跑均成功)。CUDA 相关路径因本机无 GPU 不可复核,但这与原审计口径一致,
且所有 CUDA 测试如实 skip 并被报告。

## 6. Claim boundary

无漂移。RVIR 保持 VALIDATED-REDUCED(CPU correctness/integration);fused 0/394、
394/394 typed admission、377/377 online exact-call、无 fresh CUDA、无性能合同、
无完整 verifier E2E、IR-5 VALIDATED-NO-GO、ASPLOS-ready=NO 等限制全部保留,
本轮文档增量只降不升。

## 7. 建议的下一工程动作

1. 按 DocOps 流程提交本审计结果(audit-submit,verdict=approve),关闭
   rvir-post-hardening-20260803;
2. RVIR 路线在 correctness/integration 维度已闭环;若未来要触及 performance/CUDA,
   必须另立新 task 与公平 lower-only 性能合同 + fresh GPU protocol,不得复用本轮
   correctness artifact;
3. 可选工程改进(info 级):phase6h runner 增加 `import boundflow` 自检,提升
   受限环境下的报错可读性。
