# ASPLOS'27 S4-0 mutable-state admission 实现修改记录

status: implemented-local-correctness-not-formally-closed
date: 2026-08-30
performance-claimed: false
timing-recorded: false
same-solver-claimed: false

## 1. 本轮结果

S4-0 已从纯施工合同推进为可执行的 production-shaped mutable-state admission：

    snapshot + canonical topology + R31 production plan + 12 live alpha/beta owners
      -> tensor-free canonical admission receipt
       + non-serializable process-local strong-reference lease
      -> one-shot transfer to the next prepared-runtime phase

正式冻结 ResNet2B 投影在真实 CUDA 上通过：

- ReLU slot / mutable path：6 / 12；
- alpha stored / active / preserved：8,496 / 4,248 / 4,248；
- beta slot / active slot / active element：6 / 1 / 6；
- live tensor / element / logical bytes per capture：12 / 8,502 / 34,008 B；
- 双 content capture：2 passes / 24 logical D2H / 68,016 B；
- candidate kernel / candidate CUDA allocation：0 / 0。

这些是 admission 正确性和逻辑传输账，不是性能结果，也不是 CUPTI 物理 transaction 计数。

## 2. 代码改动

新增：

- boundflow/runtime/asplos27_s4_mutable_state_admission.py
- tests/test_asplos27_s4_mutable_state_admission.py

实现包含：

1. 六站点 alpha/beta 的 canonical slot、receipt、稳定 hash 与逐层 projection；
2. pinned provider exact built-in dict/list/Tensor extractor；
3. receipt 前后双 live capture；
4. object、raw storage、shape/dtype/device、stride/offset、_version、content 与 alias 门禁；
5. PID、owner thread 与 current CUDA stream 身份；
6. receipt 与 lease 分离：receipt 可 JSON 化，lease 强引用原 Tensor 且禁止 copy/deepcopy/pickle；
7. prepared wrapper 单次 transfer，失败后关闭 owner，禁止 retry/fallback 复用；
8. 六个 private failure-injection phase，只供测试，不扩大公共 API；
9. process-global query exclusivity 保持 false，并以专用 reason 拒绝提前升级。

## 3. 实施中纠正的事实

冻结 formal snapshot 的 optimizer_policy.deterministic=false。旧施工文档误写为必须 true，会把真实
production snapshot 错拒。现改为：

- lower=true、upper=false、fix-intermediate=true 仍是准入条件；
- deterministic 按 snapshot 原值进入 optimizer policy hash；
- 不把 deterministic=true 写死为 admission 条件。

## 4. 负向门禁

专项测试覆盖：

- 非法 exact-call ID；
- snapshot/plan schema 与 residual validator；
- 非 lower-only optimizer policy；
- topology 缺失、重复与错误 native binding；
- live path 缺失、多余、非字符串 key；
- dict subclass、nested custom dict、tuple beta collection、Tensor subclass/Parameter；
- object alias、nonempty shared storage alias；
- same-content clone、same-object storage rebind、普通 in-place version drift、.data content bypass；
- admission 双 capture 间 object/storage/version/content/stream read race；
- exact-call、thread、stream 漂移；
- copy/deepcopy/pickle、重复 transfer、close 后使用；
- receipt accounting/claim/slot-order 篡改；
- coherent full-resign receipt 无法跨越原 live lease admission hash。

最终专项为 63 passed。测试条目数量不直接冒充 construction package 要求的“56 个独立 stable-negative
类别”；正式关闭仍需生成逐类 registry、5 fresh real-provider worker、stdlib replay 与 artifact tamper。

## 5. 验证

- S4-0 专项：63 passed；
- 相邻 RVIR-v4/R3 回归：87 passed；
- 全量仓库回归：1947 passed, 3 skipped；
- mypy（新增两文件）：clean；
- pylint（新增两文件）：10.00/10；
- Black：通过；
- performance_claimed=false、timing_recorded=false。

## 6. 未关闭项

S4-0 尚不能正式关闭，剩余：

1. 真实 alpha-beta-CROWN provider object 的 5 fresh 独立进程 admission；
2. canonical raw artifact、manifest 与 stdlib replay；
3. 56 类独立 negative registry 的机械计数和逐项 exact reason；
4. fully re-signed tamper 集；
5. 外部审计。

S4-1A buffer implementation、TIR evaluator、optimizer、same-solver timing、share、query 与 10x claim 均未开放。
