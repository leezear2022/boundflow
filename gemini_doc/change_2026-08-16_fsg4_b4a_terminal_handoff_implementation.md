# 2026-08-16 FSG4/B4-A terminal lower/lA handoff 实现候选

## 实现

- 新增B4-A专属terminal optimizer producer：前9次evaluation保持可微CROWN+Adam/clamp/scheduler，
  第10次无update evaluation同时产出terminal lower与六层native lA；
- 新增immutable typed handoff、correlation-parent operator lineage、one-shot lease与no-CROWN typed
  export assembly；B3默认API/schema与control路径保持不变；
- same-solver executor新增显式B4-A opt-in，沿用B3 prepared core、KFSB、device atomic commit、post/queue；
- runtime热路径只验证state/topology/shape/dtype/device/layout与单次消费；tensor content、terminal state、
  forward trace和完整export digest在query结束后的排除计时audit中绑定；
- correctness worker保存terminal lower、六层lA与六组intermediate lower/upper的标准base64 float32 raw，
  供不同fresh process按`atol=rtol=2e-4`与sign exact直接比较；
- 新增5-pair交替顺序runner、root semantic replay与固定code/raw/protocol/manifest绑定；正式artifact尚未生成。
- five-fresh首次启动在worker执行前发现解释器symlink被`resolve()`展开为裸Python；runner已改为保留
  virtualenv symlink的absolute path，失败未生成artifact、未消耗样本。

## 当前验证

- 新B4-A及worker单元/相关测试：`22 passed`；
- 固定B3/B4-A related集合（含device/KFSB）：`40 passed`；
- Mypy clean；Pylint目标模块待最终10.00复核；
- 独立GPU smoke确认handoff=`1`、terminal export CROWN rerun=`0`、lineage=`6`、provider/fallback=`0`；
- 一组热路径修正后的B3→B4-A smoke：lower max diff=`7.152557373046875e-07`、discrete/sign exact，
  core ratio约`1.02894x`、query ratio约`1.00137x`。

上述单pair数字仅用于smoke/诊断，未满足five-fresh和clean-source门禁，`performance_claimed=false`，不得
作为B4-A性能结论。

## 状态与下一步

状态=`IMPLEMENTED-B4-A-PENDING-CLEAN-SOURCE-FIVE-FRESH`。下一唯一动作是完成全量/静态检查、提交clean
source，再按冻结顺序生成5 fresh B3/B4-A correctness artifact。five-fresh通过前不得运行正式性能门禁，
B4-B/TIR、B4-C/D与B5—B7继续关闭。
