---
status: formal-tooling-ready-pending-publication
date: 2026-08-31
stage: s04
performance-claimed: false
---

# S4-1B0 formal artifact 工具链修改记录

## 结论

为 S4-1B0 ternary input endpoint 新增一套独立于性能声明的 correctness artifact 工具链。
工具链固定执行 5 个 positive fresh process、1 个 cache fresh process 和 5 个 fault fresh process，
并用 stdlib-only replay 从原始二进制 sidecar 重算分类器、midpoint、selected output、cache 与
fail-closed 语义。只有本批代码提交、推送且 formal activation gate 再次返回 `PROCEED` 后，才允许
生成正式 artifact。

## 新增文件

- `scripts/run_asplos27_s4_1b0_ternary_worker.py`：单进程 positive/cache/fault worker；
- `scripts/run_asplos27_s4_1b0_ternary_artifact.py`：11-process raw-first artifact 生成器；
- `scripts/replay_asplos27_s4_1b0_ternary_stdlib.py`：不 import BoundFlow/Torch/TVM 的语义 replay；
- `scripts/probe_asplos27_s4_1b0_ternary_tamper.py`：10 类外层 coherent-resign 篡改探针；
- `tests/test_asplos27_s4_1b0_ternary_artifact.py`：artifact replay、claim 边界和 tamper gate。

## 冻结边界

- production fixture 固定为 S3 streamed suffix 的真实 ResNet2B boundary capture；
- positive sidecar 固定保存 coefficient/lower/upper/selector/selected 五个张量；
- 正式 artifact 必须绑定生成时的 git revision、全部工具源码 blob、冻结合同、外部仓库 commit、
  model/property hash 和 endpoint construction hash；
- replay 必须逐元素以 IEEE-754 bit rule 重算 `+1/-1/0/-128`，并验证 canonical qNaN、midpoint
  运算顺序、selected bit pattern、module receipt 与 module 文件；
- fault 必须绑定 stable detail code、verification reason、reject-before-launch 和 cleanup；
- 10 类篡改均会同步重签 worker/summary/manifest 外层 hash，拒绝必须来自派生语义重算；
- coherent full-resign E0 边界继续显式披露；
- `formal_authority=true` 只表示可生成 correctness artifact；`timing_authority=false`、
  `performance_claimed=false` 始终保持。

## 本批静态与局部验证

- Black：5 个文件 clean；
- mypy：5 个文件逐文件 clean；
- Pylint：`10.00/10`；
- targeted：endpoint `19 passed`，artifact tests 在正式 artifact 生成前按合同 `3 skipped`；
- `git diff --check`：PASS。

正式 11-process 数据、replay、tamper 与完整回归数字不在本记录预写；它们必须在已发布源码上
现场生成后写入下一份 closure/handoff，避免把计划数字误写成实测结果。
