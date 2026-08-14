# FSG4/B3 IR/Graph/Plan/Schedule Reuse 预注册 — 外部审计交接

> 历史交接：B3-0现已以`VALIDATED-B2-COUNTERS`关闭。当前正式审计入口为
> `gemini_doc/fsg4_b3_0_counter_external_audit_handoff_2026_08_14.md`；本文末尾“是否允许启动B3-0”只保留
> 预注册时的审计问题，不再是当前执行指令。

## 审计目标

请审计
`gemini_doc/BOUNDFLOW_FSG4_B3_IR_GRAPH_PLAN_SCHEDULE_REUSE_PLAN_2026_08_14.md`是否在实现前形成了
可证伪、无结果污染的B3计划。不要审计B3 speedup；当前尚无B3 candidate或性能结果。

## 必核事实

1. FSG3 v5是否真的支持B2 query/core=`0.908400x/0.516767x`及五区域share；
2. 源码是否确有module binding move、10份optimizer step clone、重复terminal forward、12-path CPU
   candidate/digest与重复validation；
3. B3-A/B/C是否只属于IR/graph/Plan/Schedule复用，没有偷混B4—B7变量；
4. terminal-only与device-resident commit是否仍保持10/9、KFSB child work、rollback和fail-closed；
5. physical activation counter能否区分“对象存在”和“真实执行改变”；
6. 36-process B0/B2/B3协议、ratio方向与Go/Reduced/No-Go分类是否合理；
7. B3失败是否只关闭该实现，不被错误外推为B4—B7或全栈NO-GO。

## 建议独立命令

```bash
python scripts/run_fsg3_same_solver_experiment.py replay \
  --artifact-dir artifacts/fsg3-same-solver-timing/resnet2b-prop0-v5

rg -n "clone|stable_hash|validate|_forward_ibp_trace_mlp|module.bindings" \
  scripts/run_rvir_v4_live_return_capture.py \
  boundflow/runtime/rvir_v4_native_optimizer.py \
  boundflow/runtime/rvir_v4_native_backward_export.py \
  boundflow/runtime/rvir_v4_native_kfsb.py \
  boundflow/runtime/rvir_v4_atomic_copy_out.py \
  boundflow/runtime/rvir_v4_live_return.py
```

## Verdict格式

请给`APPROVE / APPROVE-WITH-MINOR / REQUEST-CHANGES`，按blocker/major/minor/info列finding，并明确回答：

- B3边界是否过宽或过窄；
- `>=1.15x core + B0 query parity`主门槛是否合理；
- digest移出timed core是否仍有足够transaction integrity；
- 是否允许启动B3-0显式counter diagnostic。
