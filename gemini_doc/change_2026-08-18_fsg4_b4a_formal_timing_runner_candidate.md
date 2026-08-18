---
status: implemented-pending-clean-source-formal-run
updated: 2026-08-18T12:20:00+08:00
type: change
topic: boundflow
slug: fsg4-b4a-formal-timing-runner-candidate
stage: s01
---

# FSG4/B4-A 正式计时 Runner 候选修改记录

## 修改目的

按已冻结的 B4-A 正式计时计划，实现独立于 five-fresh correctness artifact 的 B3/B4-A 正式性能
实验。当前只建立可审计测量工具，不形成性能结论。

## 代码修改

- 新增 `scripts/run_fsg4_b4a_formal_timing.py`：固定6 block、24 fresh process顺序；control进入
  headline，profile只作归因；每个worker前执行冻结的GPU环境preflight。
- runner将source/code blobs、five-fresh admission、模型/property及外部仓库commit写入protocol；保留
  venv symlink，使用确定性路径别名，并对partial resume和worker timeout fail closed。
- 每个control pair重新核对final solver semantics及19个terminal export raw tensor；同时核对
  B3 handoff/rerun=`0/1`、B4-A=`1/0`、lineage=6及provider/fallback=0。
- root replay从24个raw worker、metadata与日志重建paired ratio、profile closure和冻结分类；正式外审前
  始终保持`performance_claimed=false`。
- 新增 `scripts/probe_fsg4_b4a_formal_timing_tamper.py`，固定12类outer-resigned攻击：latency、worker
  delete/order、activation、raw export tensor、runtime、worker protocol、formal preflight、outer protocol、
  paired ratio与summary。

## 测试修改

- `tests/test_fsg4_b4a_formal_timing.py`：冻结顺序与唯一性、精确门槛边界、四类NO-GO边界、venv
  symlink、路径清洗及timeout失败记录。
- `tests/test_fsg4_b4a_formal_timing_tamper.py`：冻结12类攻击清单。

## 已执行验证

- 固定related 8文件：`46 passed`；
- 新增formal单测：`10 passed`；
- 全量（加载 `env.sh`）：`1350 passed, 3 skipped`；首次未加载 `env.sh` 的尝试在collection阶段因
  vendored TVM不可见而退出，属于shell环境错误，不计为代码回归；
- Black check通过；
- Mypy两脚本clean；
- Pylint四个新增文件：`10.00/10`；
- `python -m py_compile`两脚本通过。

## 当前边界与下一步

状态为 `IMPLEMENTED-PENDING-CLEAN-SOURCE-FORMAL-RUN`。本轮没有使用 correctness artifact latency，
没有形成B4-A speedup claim，也没有打开B4-B/TIR。下一唯一动作是提交clean source，随后从position 0
生成24-process正式GPU artifact，再执行root replay、12/12 tamper、全量回归与外部审计。
