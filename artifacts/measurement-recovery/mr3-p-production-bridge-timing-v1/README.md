# MR3 P-Anchor Production Bridge Timing v1

这是冻结ResNet2B property-0上single-site P-anchor production bridge的正式physical timing artifact。

- worker source：`2d788ad6608d7d4da9ac9937efa3cdeb11d36f27`
- generator：`5e7751c6e4efffa99e723744ef08dd67f2ce5c03`
- 顺序：6 pair/12独立process，`PB/BP/PB/BP/PB/BP`
- `raw.json`内逐run保存全部12份worker payload；不另存重复worker副本
- headline：完整beta-split optimized outer exact call host wall
- diagnostic：同一current stream CUDA event pair
- correctness：每pair 9,540元素，global max diff=`3.11434268951416e-06`，sign exact
- bridge：forward/backward=`10/9`，fallback/eager/native shadow=`0/0/0`
- host speedup geomean/bootstrap 95% lower/worst=
  `0.9797271338044103x / 0.939359906459521x / 0.9160939561911633x`
- absolute peak allocated/reserved worst ratio=
  `1.0322398954625331x / 1.032258064516129x`
- host/event方向一致=`6/6`
- fully re-signed非法变体=`16/16 rejected`

机械结论：`VALIDATED-NO-GO-MR3-P-PRODUCTION-BRIDGE-PHYSICS`。保留此前production bridge
correctness，但不开放same-solver complete-query timing，不形成speedup、query/queue、B0/B3 parity、
multi-site或ASPLOS-ready claim。

重放：

```bash
python scripts/run_mr3_production_bridge_timing_formal.py \
  --artifact artifacts/measurement-recovery/mr3-p-production-bridge-timing-v1 \
  --replay
```
