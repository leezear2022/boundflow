# MR4 Production Conv Site Census v1

这是冻结ResNet2B property-0真实beta-split optimized exact call的无计时Conv-site census artifact。

- worker source：`1fa4f0f952bae344a24b78aab8b3ca72e6bcd244`
- 5个独立provider process，150 rows
- direct edges：C0 `/input-4←/input`、C1 `/input-12←/input-8`、C2 `/input-24←/input-20`
- 每site=`50 evaluations / 45 grad-enabled / beta numel 0 / handoff 50/50`
- global semantic max diff=`3.516674041748047e-06`，sign exact
- MAC units C0/C1/C2=`1,327,104 / 1,769,472 / 884,736`
- total/P=`4.5x`，new-sites/P=`3.5x`
- projected independent candidate launches=`30 forward / 27 backward`
- minimum candidate materialization=`344,136 B/evaluation`、`3,441,360 B/outer call`
- fully re-signed tamper=`16/16 rejected`

状态：`OPEN-MR5-MULTI-CONV-PRODUCTION-BRIDGE-CORRECTNESS-PREREGISTRATION`。MAC ratio只表示静态
结构机会，不是GPU time share或speedup；MR5只能先做correctness，timing仍关闭。

```bash
python scripts/run_mr4_production_conv_site_census_formal.py \
  --artifact artifacts/measurement-recovery/mr4-production-conv-site-census-v1 \
  --replay
```
