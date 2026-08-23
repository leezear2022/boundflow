# B4-B2 B2-3 外部审计报告(Round 1)

- 任务:`fsg4-b4b2-b2-3-dense-conv-20260823` / r001
- 审计对象:P-anchor `performance-conv-8-candidate` dense Conv TIR forward/backward correctness
- 仓库:`/home/lee/Codes/boundflow`,branch `feat/rvir-v4-production-state-ownership-v1`
- HEAD=`2710d01`(== origin,已核实);source commit=`73070706935f2e6610d4e12903e1d9b4f67b0f83`,base=`c28c903`
- 审计环境:RTX 4060 Laptop GPU(sm_89),torch 2.12.1+cu132,tvm 0.23.dev0(vendored 6248b5db),python=conda env `boundflow`
- 审计时间:2026-08-23;审计方:独立外部模型(未采信交接数字,全部现场复核)

## 总体 Verdict

**APPROVE**。0 blocker、0 major、0 minor、2 info。同意关闭 B2-3,并按预注册 DAG 只开放
B2-4 P-anchor sparse-source schedule search;timing、B2-5 formal artifact、B4-B3 保持关闭。

---

## AC1 git/范围/receipt:PASS

- `git log` 确认 `7307070` 直接后继于 `c28c903`("docs: close B4-B2 B2-2 external audit"),
  即实现前 B2-2 外审已批准,顺序正确。
- `git show 7307070 --stat`:17 文件,+2586/-8。核心新增 5 文件
  (IR receipt / backend / runtime / runner / conv 测试)+ dense linear 测试补 24 行 +
  文档/DocOps 状态更新。无 production、optimizer、vendored TVM/TVM-FFI/auto_LiRPA 改动;
  无 timing API、无 B2-4 sparse-source schedule search、无 B2-5 artifact、无 B4-B3。
- 预注册计划(`...PREREGISTRATION_PLAN_2026_08_23.md`)与 B4-B 总计划本轮 diff 仅为
  front-matter 状态戳 + 追加"内部结果"段,B2-3 门禁条文
  ("实现Conv transpose-contraction dense correctness schedule,冻结`[6,1,16,8,8] x [16,16,3,3]`与
  全部attrs。P incoming-A/native α gradient必须通过;empty β保持absent")未被事后改动。
- first-class receipt 机制与 B2-1/B2-2 同构:Template/Instance/Schedule/Module/Launch
  五级 receipt,`canonical_tir_hash` stable hash,from_dict/to_dict round-trip,
  `validate*` fail-closed(`boundflow/ir/differentiable_lower_dense_conv_tir.py:184-217,
  316-332,385-399,477-507,602-645`)。`performance_admitted`/`performance_claimed` 在
  template/schedule/module/launch 四级均被硬编码为 `is not False` 拒绝篡改(亲测见 AC4)。
- TVM 版本锚定:superproject gitlink TVM=`6248b5db`(== `FROZEN_TVM_COMMIT`,现场
  `git -C boundflow/3rdparty/tvm rev-parse HEAD` 一致);TVM 自身 gitlink 锁定嵌套
  tvm-ffi=`ae346ec92`(即实际 import 的副本);superproject 独立 tvm-ffi 子模块=
  `438f6439`(== `FROZEN_TVM_FFI_COMMIT`)。两级 pin 均与工作区一致。

## AC2 数学:PASS(独立 float64 闭合公式重算)

亲自通读 `boundflow/backends/tvm/differentiable_lower_dense_conv.py` 并独立推导语义:

- forward ReLU lower-bound 选择语义:incoming `a>=0` 取 lower slope
  (L>=0→1;U<=0→0;ambiguous→clamp(α,0,1));`a<0` 取 upper slope
  (L>=0→1;U<=0→0;ambiguous→U/max(U−L,ε),ε=float32 eps)与 intercept `−L·upper_slope`
  (仅 ambiguous)。与 α-CROWN 风格 ReLU 下凸包理论一致。
- forward 收缩 `out[d,s,co,oh,ow]=Σ_{ci,kh,kw} relu_a[d,s,ci,oh+pad−kh·dil,ow+pad−kw·dil]·W[ci,co,kh,kw]`
  (带边界掩码),stride=1/pad=1 下即 PyTorch `conv_transpose2d` 语义
  (weight layout `[in_c,out_c,kh,kw]`,输出空间 (8−1)−2+2+1=8 ✔)。
- `output_bias = incoming_bias + Σ a·selected_intercept + Σ_{c,h,w} relu_a·operator_bias[c]`
  —— intercept 与 operator-bias reduction 均正确(第二项 = ⟨relu_a, ob⟩)。
- backward adjoint `gin[ci,ih,iw]=Σ_{co,kh,kw} gout[co,ih−pad+kh·dil,…]·W[ci,co,kh,kw]`
  与 forward 收缩精确互为转置(索引映射自洽,亲推导)。
- `native_alpha_gradient = Σ_s adjoint_relu·a`,gate = `a≥0 ∧ L<0 ∧ U>0 ∧ 0≤α≤1`
  (ambiguous/clamp endpoint 所有权与我推导一致;`adjoint_relu = adjoint_conv + g_bias·ob[c]`
  正确合并了 output_bias 通路对 α 的 VJP)。
- `incoming_lower_a_gradient = adjoint_relu·selected_slope + g_bias·selected_intercept`
  —— 对 `output_lower_a` 与 `output_bias` 两条通路求导后合并,与我推导的闭合式一致。
- β gradient absent 的设计正确性:P-anchor capture 的 `production_beta` 形状=`(6,0)`、
  beta location/sign 均空(亲读 `run_00.pt` capture[1]),`beta_active=False`,reference 中
  `native_beta_gradient=None`;runtime 显式拒绝 reference beta gradient present
  (`fsg4_b4b2_dense_conv_tir.py:632-633`),receipt 硬编码 `beta_gradient_present is not False`
  拒绝。ABI 不出现伪零 beta tensor ✔。

独立 oracle(非 repo reference、无 autograd、float64,手写 padding+einsum 索引映射,
脚本 `/tmp/b2_3_oracle.py`)对 5 份 raw 四路输出全部重算:

```
run 0..4 一致量级:
  output_lower_a             |oracle64−tir32| ≤ 2.175e-08  sign_exact=True finite=True
  output_bias                |oracle64−tir32| ≤ 3.341e-07  sign_exact=True finite=True
  native_alpha_gradient      |oracle64−tir32| ≤ 3.730e-08  sign_exact=True finite=True
  incoming_lower_a_gradient  |oracle64−tir32| ≤ 1.830e-06  sign_exact=True finite=True
WORST = 1.8300339528209975e-06  (gate 2e-4)
```

TIR float32 输出对 float64 真值最差 1.83e-06,远小于 2e-4 门禁;交接声明的 TIR-vs-
reference max diff `2.384185791015625e-06` 与我的 oracle-vs-TIR 1.83e-06 相互吻合
(三角不等式内自洽),且最大误差出现在 incoming gradient——该输出链最长
(adjoint conv 144 项 reduction + slope/intercept 乘加),比 Linear(单 matmul VJP)深,
2.8× 于 Linear 的 8.6e-07 在数值上合理。

## AC3 现场实测:PASS

现场重跑 `python scripts/run_fsg4_b4b2_dense_conv_tir_correctness.py`(完整 JSON 留存
`/tmp/b2_3_runner_out.json`):

- run/metrics/elements = 5/20/92190 ✔;allclose=true、sign_exact=true ✔;
- max diff = `2.384185791015625e-06` ✔(与交接逐位一致);
- template=`950f20535ab55120e497401c7d17513c5f2118fd65401e4e87d3a081567c4dc2` ✔;
- schedule=`1de607ad7faf39ff1b45ee81b90013e3cc841c69e97fd3aabba0f135893cc7ec` ✔;
- module receipt=`4511fbc51159cea516e568f025636fa9fee0cf97225f032ddf877f8239dbad79` ✔;
- cache=`miss,hit,hit,hit,hit` ✔;每 run forward/backward launch=1/1、fallback/eager=0/0 ✔;
- beta_gradient_present=false ✔;
- DLPack 19/19:forward 7 输入+2 输出=9,backward 8 输入+2 输出=10,合计 19,
  receipt 硬校验 `dlpack_pointer_count==19 且 exact==count`(我在 AC4 脚本独立打印 19/19,
  stream_id==tvm_ffi_stream_id ✔);
- GPU identity:runner 报 `NVIDIA GeForce RTX 4060 Laptop GPU`、sm_89,与
  `artifacts/env/host-doctor.json`(torch 2.12.1+cu132 / tvm 0.23.dev0 / 驱动 610.43.03)一致 ✔。

观察(非问题):5 份 raw 之间仅 `incoming_lower_a` 有 ≤7.45e-09 的 float 抖动
(生产重跑噪声),其余输入逐位相同;因此 metric hash 中 `output_bias`/`incoming_grad` 跨
run 相同、`output_lower_a` 因 conv 收缩放大抖动而不同。这是 B4-B1a 冻结 artifact 的固有
性质(前五轮已审),且 delivery.md 已如实声明 "five independently captured production raws"。

## AC4 结构/tamper:PASS

- 亲自遍历 scheduled TIR `Block.alloc_buffers`(独立 walk 代码,`/tmp/b2_3_ac4.py`):
  observed = `[('adjoint_conv',(6,1,16,8,8)), ('output_bias_delta',(6,1))]`,与
  `DENSE_CONV_WORKSPACE_INVENTORY` 及 repo observed 三方一致;`Allocate` 节点=0,
  `relu_lower_a`/`adjoint_relu` 已 compute_inline(script 中无残留块)。本轮为真结构遍历,
  非 script 子串计数。
- tamper 实测(全部拒绝):
  - 序列化 TIR 中把 `adjoint_conv` 改名重载后 walk → inventory 失配,结构门禁必然触发
    (`build_dense_conv_tir_modules` 在 backend:457 处 RuntimeError fail-closed);
  - schedule `workspace_inventory` 篡改 → "schedule differs";
  - module receipt observed inventory 篡改 / `structural_workspace_check=False` /
    `performance_claimed=True` / cache_key 伪造 → "module receipt differs";
  - 伪造完整 module receipt dict(篡改 inventory 且 flag=True)→ from_dict 拒绝;
  - launch receipt `performance_claimed=True`、`beta_gradient_present=True`、
    launch 计数 2、DLPack 18、fallback 1 → "launch receipt differs";
    伪造完整 launch dict 同样拒绝;
  - schedule `thread_extent=256` → "schedule differs"。

## AC5 负路径与遗留:PASS

- `tests/test_fsg4_b4b2_dense_conv_tir.py` 实测覆盖且断言具体(match 串):dtype(double)、
  device(cpu)、nonfinite(nan)、alpha range(1.5)、invalid interval(lower>upper)、
  S-anchor/scope broadening("P-anchor differs")、missing symbol 异常后
  device/stream/determinism policy 不漂移、fallback/eager 真实计数(1,1)、higher-order
  gradient 拒绝、instance(fresh_run_ordinal=5)/module(structural flag)/launch
  (performance claim)篡改。8 项测试全部在 43 passed 中真实运行(非 skip)。
- 上轮遗留 info#1(dense Linear 缺 dtype/device/nonfinite 专项拒绝用例):本轮
  `tests/test_fsg4_b4b2_dense_linear_tir.py` 新增
  `test_b4b2_dense_linear_dtype_device_and_nonfinite_rejected`(double dtype / nan /
  cpu device,断言具体 match),已关闭 ✔。
- 上轮遗留 info#2(sparse linear forbidden workspace 为 script 子串计数):B2-2 侧代码本轮
  未改(该 info 当时已被接受);B2-3 dense Conv 侧采用了正确的结构遍历方案,未延续该缺陷。

## AC6 测试/静态/DocOps:PASS

- B2 targeted(identity+dense linear+sparse linear+dense conv):**43 passed**(19.42s)✔
- B4-B related(`tests/test_fsg4_b4b*.py`):**97 passed**(26.69s)✔
- `tests/test_env.py`:**3 passed** ✔
- 全量 `pytest -q -rs`:**1457 passed, 3 skipped, 6 warnings in 467.73s** ✔;3 skip 均为
  既有环境边界(`test_artifact_phase5d_smoke` 的 allow-no-tvm 去重、两项 VNN-COMP frozen
  checkout 不可用),与交接声明一致。
- Black `--check` 6 文件 clean;Mypy 4 source `Success: no issues found`;
  Pylint `10.00/10`;`git diff --check 7307070~1 7307070` clean ✔
- TVM rebuild:本提交无 C++ 改动;`ninja -C boundflow/3rdparty/tvm/build-boundflow`
  现场执行 = `ninja: no work to do.` ✔
- `dol exchange validate fsg4-b4b2-b2-3-dense-conv-20260823` → `{"ok":true,...}`;
  `dol lint --soft` → `{"ok":true,...,"soft":true}` ✔

## Claim 边界:无漂移

- 交接/changelog/claims map/备忘录/master plan/current_status/delivery 措辞一致限定
  P-anchor dense Conv correctness pending external audit;
  `performance_admitted=false`/`performance_claimed=false` 由 receipt 硬校验且亲测篡改被拒;
- 新代码无 timing/benchmark API(grep 确认);无 B2-4/B2-5/B4-B3/ASPLOS-ready 措辞;
- "5 raw 为五个独立 capture 的 production raw;B2-5 formal 独立进程 artifact 仍关闭"在
  delivery.md Risks 与交接 §10 中保留 ✔。

## B2-4 门禁评价

预注册 DAG:B2-3 approve 后唯一开放 B2-4(P-anchor sparse-source fused schedule search,
预登记变换族、ledger ≤12 hash、无 performance claim);交接 §10 与之一致。本次
APPROVE 只开放 B2-4;timing、B2-5、B4-B3 继续关闭。符合预注册。

## Findings

| severity | path | evidence | advice |
|---|---|---|---|
| info | `boundflow/ir/differentiable_lower_dense_conv_tir.py:477-507` | module receipt 的 `unscheduled_tir_hash`/`scheduled_tir_hash`/`device_source_hash` 字段在 validate 时仅做格式校验(64-hex),不重编译比对——内容自证,篡改单个 hash 字符串本身不会被 validate 检出(但链上 template/schedule/workspace/cache_key 篡改均 fail-closed,且 workspace inventory 锚定冻结常量)。与 B2-1/B2-2 已批准设计相同 | 可接受;B2-5 formal artifact 阶段若引入独立进程 replay,可由 replay 侧重编译比对这些 hash |
| info | `tests/test_fsg4_b4b2_dense_conv_tir.py` | dense Conv 无单独的 shape-mismatch 拒绝用例(形状校验存在于 runtime:227-229 且与 dtype/device 共用同一 raise 分支,已被间接覆盖);上轮同类 info 的补齐也只覆盖 dtype/device/nonfinite | 后续切片可顺手补一条 shape 失配用例;不阻塞 |

## 不可现场复核项

- 无。5 份 raw、TIR 编译、GPU 执行、结构遍历、tamper、静态检查、全量测试均在本机现场完成。

## 结论

APPROVE。0 blocker/major/minor,2 info。同意关闭 B2-3 并只开放 B2-4。
