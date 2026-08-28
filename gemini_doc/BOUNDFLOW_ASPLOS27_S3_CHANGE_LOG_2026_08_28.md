---
status: active
updated: 2026-08-28T13:20:00+08:00
type: change-log
topic: boundflow
slug: asplos27-s3-change-log
stage: s03
---

# ASPLOS'27 S3 修改记录

## 2026-08-28：冻结 S3 optimizer/runtime 大批次

- 新增 S3 预注册，固定 P-anchor 10/9 local wrapper 的 N/D/P 三方语义与六全排列性能协议；
- 明确 direct custom VJP，不复用历史 autograd Function registry；
- 冻结 `3.00x/2.50x/1.50x` research gate 与 reduced/no-go 分支；
- 明确 10/9 仅为冻结 artifact trajectory，host policy 保留，禁止升级为通用 optimizer IR；
- 保留现有 S1+S2 pending external exchange，外审按用户要求延后到下一轮；
- 当前只完成预注册，代码、性能与 claim 均未完成。

### 验证

- 文档格式与 DocOps lint：待运行；
- 代码/性能：明确 deferred，待后续提交。

## 2026-08-28：实现 direct-VJP optimizer wrapper

- 新增 `asplos27_s3_optimizer_pipeline.py`；
- 每个 sample 只建立一次 sample owner，逐 ordinal 直接执行 S2 `forward/backward`；
- Adam、clamp、scheduler 和每轮 policy cut 保留在 host；
- 新增聚合 receipt，固定 10/9、graph replay、direct VJP、无 autograd registry、无 fallback 与无 dense-A
  saved state；
- correctness capture 可选，计时路径不复制 step tensor 到 CPU。

### 验证

- 静态/专项/GPU correctness：待测试提交；
- formal performance：仍关闭。

## 2026-08-28：增加 S3 correctness 与 fail-closed 测试

- 增加独立 native eager/autograd 10/9 trajectory oracle；
- 逐 step 比较 lower、compressed dα、α before/after、Adam step/m/v 与符号；
- 增加 receipt counter、saved-state、fallback 与 claim 篡改拒绝；
- 静态断言 S3 hot wrapper 不经过旧 `_candidate_evaluate`、executor registry 或 `autograd.grad`。

### 验证

- targeted GPU：`4 passed in 10.86s`；
- mypy：2 files clean；
- full regression：待 formal source-exact 后运行。

## 2026-08-28：增加 S3 raw-first formal 工具链

- 新增 N/D/P 六全排列 fresh worker，计时覆盖完整 10/9 wrapper；
- raw 保存每一步 lower、gradient、α 与 Adam moments，replay 用标准库重算 tensor digest、容差、符号、
  counters、速度和 verdict；
- manifest 绑定 source commit、代码 blob、protocol、raw、logs 与 summary；
- 新增 10 类 outer-resigned tamper probe 与 artifact 回归测试；
- 单次非正式诊断 N/D/P 中位数约 `118.47/58.24/31.89 ms`，P/N 约 `3.72x`；该数字不形成 claim，
  正式结果只来自 source-exact 后的六 fresh artifact。

### 验证

- worker 单次 GPU smoke：PASS；
- formal artifact：待 source-exact 提交后生成。

## 2026-08-28：S3 v1 正式 NO-GO，并冻结 v2 稳健协议

- v1 六 fresh 的 P/native geomean=`2.5695746x`、worst=`0.7595405x`，按冻结门槛如实关闭为 NO-GO；
- raw 显示一个完整 fresh 进程中 N/D/P 同时持续退化，而非单样本长尾；
- 保留 v1 全部 raw，不删除或替换异常 worker；
- 新增 v2 预注册：每 order 三个 fresh 重复，共 18 worker，以六个 order 内中位数形成 headline，并披露
  全部 raw 指标；
- 性能阈值、语义、receipt 与 claim 边界不变。

### 验证

- v1 raw/replay：待 negative artifact tamper 与 manifest 收口；
- v2：尚未实现或运行。

## 2026-08-28：实现 S3 v2 18-worker 稳健 formal

- worker 增加预注册的 `replicate_index`，不改变单 worker 的 5/30 和三方路径；
- 新增 v2 artifact/replay，以每 order 三个 fresh pair 的中位数形成六顺序 headline；
- 全部 18 行仍逐行执行 v1 的完整 trajectory、receipt、memory 与 tensor digest 验证；
- summary 同时披露 18-worker raw geomean/worst 与每 order 三个原始 speedup；
- 新增 v2 10 类 outer-resigned tamper 与回归测试。

### 验证

- 静态检查与 source-exact：待下一提交；
- 18-worker formal：尚未运行。
