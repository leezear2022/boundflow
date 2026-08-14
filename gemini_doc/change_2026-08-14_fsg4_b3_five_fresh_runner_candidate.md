# FSG4/B3 五组 Fresh Correctness Runner 实现候选

日期：2026-08-14
状态：`IMPLEMENTED-PENDING-CLEAN-SOURCE-RUN`

## 实现

- 新增`run_fsg4_b3_correctness_pairs.py`，按预注册顺序启动10个独立diagnostic子进程；
- root `protocol.json`在worker前落盘，绑定source、完整B3-C code revision、顺序和模型/性质digest；
- 每个raw run使用原子临时目录rename；`--resume`只接受diagnostic replay完整通过且configuration匹配的
  已有run；
- 完成后从10个raw worker重新解析typed run，执行5组B2-reference/B3-C-candidate direct semantics；
- pair report不包含ratio/geomean/winner，只记录semantic hash、environment、provider/counter/audit gate；
- root replay重新验证historical code revision、完整递归file inventory、10个nested diagnostic、5组语义与
  report projection；
- 新增7类outer-resigned tamper probe：report、protocol、nested counter/journal、worker semantic、audit
  receipt、position swap和raw-run deletion。

## 验证

- static protocol/tamper inventory：`5 passed`；
- Black clean；mypy两个runner clean；Pylint `10.00/10`；
- source-clean gate尚未运行，因为实现文件未提交；这是预期状态。

## 下一步

提交runner与预注册文档，从该clean commit启动artifact v1。任何worker失败都保留root/raw历史并停止；只
允许修复后整体新版本，不允许补位。
