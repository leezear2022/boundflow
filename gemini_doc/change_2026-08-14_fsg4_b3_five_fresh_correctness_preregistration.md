# FSG4/B3 五组 Fresh Correctness 预注册记录

日期：2026-08-14
状态：`PREREGISTERED-PENDING-IMPLEMENTATION`

## 变更

- 冻结5组、10个独立GPU进程和B2/B3-C交替顺序；
- 冻结同source/同protocol/同environment及逐字段direct semantic comparison；
- 明确本阶段禁止计算或报告performance ratio；
- 冻结raw-first、原子run目录、`--resume`只接受完整replay通过run的中断恢复规则；
- 冻结B2/B3-C physical counter、post-query audit和provider/fallback门禁；
- 冻结5/5总门禁与“失败不得删样本补位”的kill rule。

## 下一步

实现pair runner、root replay和outer-resigned tamper probe；提交clean source后才执行10个fresh worker。

## 链接

- `gemini_doc/BOUNDFLOW_FSG4_B3_FIVE_FRESH_CORRECTNESS_PLAN_2026_08_14.md`
- `gemini_doc/change_2026-08-14_fsg4_b3c_device_atomic_commit_closure.md`
