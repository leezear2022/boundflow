---
status: external-audit-approved
updated: 2026-08-23T04:30:00Z
type: audit
topic: boundflow
slug: fsg4-b4b2-b2-0-identity-tir-external-audit
stage: s01
---

# 独立外审:FSG4/B4-B2 B2-0 Identity CUDA/TIR ABI 探针

- 审计对象:分支 `feat/rvir-v4-production-state-ownership-v1` @ `712ca03`(预注册 `57be636`);
- 审计方式:不信任 changelog 自述,全部独立复核;GPU 实测为审计方现场重跑,非引用冻结日志;
- 审计环境:`/home/lee/miniconda3/envs/boundflow/bin/python`,`source env.sh` 后执行。

## 总体 Verdict

**approve**(附 2 项 minor、3 项 info 观察项,均不构成本切片门禁问题)。

B2-0 的实现、实测证据与预注册门禁逐条吻合;预注册门禁无事后挪动;claim 边界无漂移;
下一唯一动作 B2-1 与预注册 DAG 一致。

## 逐项结论

### 1. git/顺序 —— PASS

- `HEAD == origin/feat/rvir-v4-production-state-ownership-v1 == 712ca038...`;工作树唯一改动
  `M .docops/ev.jsonl` 为 DocOps hook 自动事件,与 B2-0 内容无关(未触碰)。
- `git show 57be636 --stat`:11 文件全为文档/索引/claims map,无任何代码——纯预注册提交。
- `git show 712ca03 --stat`:`boundflow/` 下 3 个源文件 + probe 脚本 + 测试均为 **A(新增)**,
  无 M;`git diff --name-status 57be636 712ca03 -- boundflow/ scripts/ tests/` 确认纯新增,
  未动任何 production 路径。
- 时序:B4-B1 exchange `fsg4-b4b1-typed-reference-20260818` closure=`approved`(Round 2,
  `closure.json` resolution=approved,2026-08-23T02:37:44Z)→ 预注册 57be636(02:49:56Z)
  → B2-0 712ca03(03:20:40Z),顺序合法。
- 前序 exchange 全部 closed/approved:`fsg4-b4a-formal-timing-20260818`(Round 1 approved)、
  `fsg4-b4b0-five-fresh-20260818`(Round 2 approved)、`fsg4-b4-0-kernel-attribution-20260816`
  (Round 1 approved)——各 `state.json` 亲验。
- B2-1 dense TIR 未越序混入:712ca03 中无 Linear/Gemm 语义 TIR;template 校验把
  `abi` 钉死为 `"identity-probe-v1"`(`boundflow/ir/differentiable_lower_tir.py:124`),
  测试 `test_b4b2_identity_ir_rejects_scope_and_schedule_mutations` 拒绝 `abi="dense-semantic-v1"`。

### 2. 预注册一致性 —— PASS(无事后挪门禁)

- `git diff 57be636 712ca03 -- ...PREREGISTRATION_PLAN...`:仅 frontmatter 状态
  (`preregistered-not-implemented`→`validated-b2-0-next-b2-1`)、顶部 B2-0 关闭注解、§6 B2-0
  节末尾"内部结果"段;**所有门禁条文、criteria、claim ledger 一字未改**。
- 预注册 §6 B2-0 门禁逐条对照:
  - typed lowering skeleton + round-trip IR → `DifferentiableLowerTIR{Template,Instance,Schedule}V1`,
    canonical JSON(`canonical_tir_hash`,sort_keys+紧凑分隔符+`allow_nan=False`)、
    `from_dict`/`to_dict` round-trip 由测试覆盖;
  - identity TIR forward/backward probe → 双 symbol 独立 PrimFunc
    (`differentiable_lower_identity.py:83-116`),schedule 冻结 1D thread binding、
    无 workspace、candidate ordinal=0(`differentiable_lower_tir.py:262-294`);
  - DLPack data_ptr、current-stream、module/cache/launch receipt → 全部实现并现场验证(见 §3/§4);
  - 默认功能关闭 → `enabled_by_default=False`/`performance_admitted=False`/
    `performance_claimed=False`,且在 `validate()`/`validate_against()` 中强制
    (`differentiable_lower_tir.py:133-134,385,541`),篡改即 fail closed(测试覆盖)。
- 提交序列符合预注册 §11 第 1、2 步;DocOps `ch`+`va(result=pass)` 记录存在
  (`.docops/ev.jsonl` ev010925/ev010926、ev010936/ev010937),`dol lint --soft` → `{"ok":true}`。

### 3. 代码事实 —— PASS

- **(a) DLPack 零拷贝**:`runtime/fsg4_b4b2_identity_tir.py:236-243`,
  `tvm.runtime.from_dlpack` 视图回转 `torch.from_dlpack` 后比较 `data_ptr`,四个 round-trip
  任一不 exact 即 raise;launch receipt 校验强制四 ptr exact 且 output/input-gradient 与
  input/upstream `data_ptr` 互不相同(`differentiable_lower_tir.py:526-531`);另强制
  `output_tensor_hash == instance.input_tensor_hash`(identity 语义位精确,行 534-535)。
- **(b) current stream**:`fsg4_b4b2_identity_tir.py:224-233`,取
  `torch.cuda.current_stream(source.device)`,`tvm_ffi.use_torch_stream` 后比对
  `tvm_ffi.get_raw_stream(...) == current.cuda_stream`,不等即 raise——无 default stream 假设。
  现场重跑在显式非默认 stream 上执行,`stream_id=94556585521120`(非 legacy/default)。
- **(c) cache miss→hit 与 key 完整性**:`DifferentiableLowerIdentityModuleCache.get`
  (行 147-188)显式 in-process cache,返回 `"miss"/"hit"`;`expected_cache_key`
  (`differentiable_lower_tir.py:389-405`)含 schema、template hash(经其绑定 static IR hash/
  mapping hash/operator attrs/gradient targets/dtype/ABI)、schedule hash、双 symbol、target、
  compute capability、TVM/FFI commit——覆盖预注册 §4 全部要求项;动态 tensor hash 只在
  instance,不进 compile key。
- **(d) forward/backward 各恰一次、无 fallback**:executor 计数器二次 launch 即 raise
  (行 256-257, 267-268);launch receipt 强制 `forward/backward=1/1、fallback/eager=0/0`
  (行 537-540);代码中不存在任何 eager/cuDNN fallback 分支(全文亲读);backward 不含
  PyTorch 数学算子,直接走 TIR module。
- **(e) fail closed**:`_validate_probe_tensor`(行 117-132)拒绝非 CUDA/非 float32/
  非 contiguous/numel 不符/requires_grad 不符/含非 finite;instance 绑定后输入 hash 不符即拒
  (行 318-321);higher-order gradient 显式拒绝(行 286-287,`create_graph=True` 负向测试覆盖);
  receipt 篡改(performance_claimed、tvm_commit、fallback_count、alias)负向测试全拒。
- **(f) 只增不改**:见 §1;probe 不经任何 production 调用链,功能默认关闭。

### 4. 实测证据 —— PASS(现场重跑,非引用冻结日志)

仓库无冻结 probe artifact(证据原以 changelog hash 形式存在),故按审计要求现场重跑:

- 环境冻结项亲验(预注册 §1.3):Torch `2.12.1+cu132`、CUDA build `13.2`、
  GPU `NVIDIA GeForce RTX 4060 Laptop GPU`、capability `(8,9)`=sm_89、TVM `0.23.dev0`、
  子模块 commit `6248b5db...` / tvm-ffi `438f6439...`——与代码中 `FROZEN_TVM_COMMIT`/
  `FROZEN_TVM_FFI_COMMIT` 及预注册全部一致。
- 现场命令:`python scripts/run_fsg4_b4b2_identity_tir_probe.py`,输出(审计方本次运行):
  - `status=probe-passed`,`device=NVIDIA GeForce RTX 4060 Laptop GPU`,`sm_89`;
  - `template_hash=f927994b5dd02dd37269aa956d4a59645712a5dd451d52aea4245114ac2ea0fe`;
  - `schedule_hash=3bc85e3022e5262884bae856421c7c3be2d1968110c55bb340b6a3c3a1dd1a42`;
  - `module_receipt_hash=ba765577a70b7a1cab9dbfc0b51861663767be38f02948b84d3b22bc4cfc1474`;
  - **三个 hash 与 changelog §Validation 逐位一致**(确定性重建);
  - `cold_cache_event=miss`→`warm_cache_event=hit`;`forward/backward_launch_count=1/1`;
    `fallback_count=0`、`eager_backward_count=0`;`zero_copy_exact=true`;
    `output_aliases_input=false`、`input_gradient_aliases_upstream=false`;
    `enabled_by_default=false`、`performance_claimed=false`;
  - probe 内 `assert_close(..., rtol=0, atol=0)` 双向位精确(脚本行 61-62)。

### 5. 测试/静态 —— PASS

- 专项:`pytest tests/test_fsg4_b4b2_identity_tir.py -q` → **12 passed**(GPU 实测,
  与 changelog "targeted=12" 一致)。
- 组合:B4-B1+targeted=**44 passed**;`-k fsg4_b4b`=**66 passed**,均与 changelog 一致。
- 全量:`pytest -q -rs` → **1426 passed, 3 skipped, 6 warnings in 453.98s**,与 changelog
  逐字一致;`-rs` 显示 3 个 skip 为既有环境边界(phase5d 去重 smoke、两个 VNN-COMP frozen
  checkout 缺失),与 B2-0 无关。
- 静态:`black --check`(5 文件)全过;mypy 4 源文件 `Success: no issues found`;
  pylint 5 文件 **10.00/10**;`git diff --check 57be636..712ca03` 干净;
  `bash scripts/rebuild_tvm.sh` 完成无需重编译。
- DocOps:`dol lint --soft` → `{"ok":true,"miss":[],"rule":[]}`。

### 6. claim 边界 —— PASS(无漂移)

- `VALIDATED-B4-B2-B2-0-ABI-PROBE` 的措辞在 changelog、预注册头部注解、claims map、
  执行备忘录、current_status、master plan、README 七处一致限定为 ABI 接线正确性
  (first-class receipt、DLPack/stream/cache、launch 1/1、一阶 autograd),每处均显式排除
  region 数学、sparse-source 融合、timing、micro/system speedup、memory、ASPLOS-ready。
- `performance_claimed=false` 不止于文字:receipt `validate_against` 把它作为硬门禁
  (`differentiable_lower_tir.py:385,541`),篡改测试拒绝。
- 代码内无任何 timing/perf 路径(grep `time.|perf_counter|cuda.Event|elapsed|benchmark`
  在 5 个新文件中零命中)。
- B2-1 下一步表述("S-anchor `semantic-active-beta-gemm-14` dense semantic TIR
  forward/backward,5 个 B4-B1 raw instances correctness,不计时")与预注册 §6 B2-1 节、
  §11 提交序列第 3 步逐字一致;`.docops/s.md` next=`implement-b4b2-b2-1-s-anchor-dense-tir-correctness` 对齐。

## Findings

| severity | 位置 | 证据 | 建议 |
|---|---|---|---|
| minor | `boundflow/runtime/fsg4_b4b2_identity_tir.py:366-367` | `fallback_count=0`/`eager_backward_count=0` 为硬编码常量而非计数器 | B2-0 无 fallback 路径故语义等价;B2-1 wrapper 复杂化后应改为真实计数并纳入 ledger |
| minor | changelog §Validation "rebuild_tvm.sh=ninja: no work to do" | 审计方重跑 tail 显示 "Rebuild Complete!",未逐字复现 ninja 行 | 仅为措辞;功能上确认无需重编译,可不处理 |
| info | 无冻结 probe stdout artifact | 证据以 changelog hash 存在;本次审计现场重跑逐位复现三 hash 已弥补 | B2-5 formal artifact 阶段保留 probe stdout raw 文件 |
| info | 预注册 §5"异常退出后 stream/device/global policy 不漂移" | B2-0 门禁未要求,亦无专项测试 | 留待 B2-1+ 随 dense ABI 一并门禁 |
| info | 工作树 `M .docops/ev.jsonl` | hook 自动事件,内容与 B2-0 无关 | 无需处理 |

## 不可现场复核项

- changelog 中 "ninja: no work to do" 的逐字输出(见 minor #2);
- 执行方当次 probe 的原始 stdout(无冻结 artifact)——已由审计方现场重跑等价替代,
  三个确定性 hash 逐位一致,风险已消除。
- 其余全部项目均由审计方在本机独立复跑/亲读验证。

## 对 B2-1 的评价

B2-1(S-anchor dense Linear/Gemm semantic TIR forward/backward,对 5 个 B4-B1 raw instances
逐项比较 forward A/bias 与 incoming-A/native α/active native β gradient,确定性 correctness
schedule、不计时)与预注册 §6/§7/§11 完全对齐,是当前唯一合法下一动作。提示:B2-1 起
预注册 §2 的离散导数所有权(`A==0`、`a==0/1` 边界与 PyTorch `where`/`clamp` 选择逐元素一致)
与 §7 容差门禁将成为核心审计点;本切片硬编码 counter 的 minor 项应在 B2-1 一并改为真实计数。

## 关键命令记录

```bash
git rev-parse HEAD @{u}                      # 均为 712ca038...
git show 57be636 --stat                      # 11 文件,纯文档
git diff --name-status 57be636 712ca03 -- boundflow/ scripts/ tests/   # 5 文件全 A
git diff 57be636 712ca03 -- ...PREREGISTRATION_PLAN...                 # 仅状态/结果注解
python scripts/run_fsg4_b4b2_identity_tir_probe.py                     # probe-passed,三 hash 逐位一致
python -m pytest tests/test_fsg4_b4b2_identity_tir.py -q               # 12 passed
python -m pytest -q -rs                                                # 1426 passed, 3 skipped, 6 warnings
python -m black --check <5 文件>; mypy <4 文件>; pylint <5 文件>         # 全过 / no issues / 10.00/10
python3 .../dol.py lint --soft                                         # {"ok":true}
```
