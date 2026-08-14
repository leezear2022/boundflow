# FSG4/B3-C Device-Resident Atomic Commit 正式关闭记录

日期：2026-08-14
状态：`VALIDATED-B3-C-COUNTERS`

## 结论

source `72bec5ee1bdabfdefbf51201ac49395489eeef65`的fresh GPU真实same-solver call证明：12条
mutable candidate在CUDA设备上完成projection、backup和direct commit，headline candidate D2H从B3-B
的12次降为0；post-query audit在CUDA query event与wall timer结束并同步后执行。冻结语义保持，
但本阶段没有timing或speedup claim。

## 正式证据

- artifact：`artifacts/fsg4-b3-counter-diagnostic/resnet2b-prop0-b3c-v1/`；
- manifest internal hash：`091f6ac8f57ba3de642e5ae0f390bc8b005904b64d6eea02526dd93bb166e1c2`；
- report internal hash：`72812a3565e33e0103ce734090c728b535ca32fefa8aa4b928b0b09d93e5cb44`；
- manifest file SHA256：`45a1a8f55b154b86cac85ba600e45a7bd9613ec6969276d6c5da1e05191845aa`；
- report file SHA256：`9b65e304a22dbc51095cae52920173a5b5d2254ca3ffa8b93f0059411df04a5d`；
- event journal SHA256：`d6a203cfe7870314f9e5cc8612c912f59f53943903eee60c853886aef105348e`；
- worker SHA256：`c9bde576684e89620ba5317c98eb0a823d57ab56398dbcbca2df812d13daf029`；
- post-query audit hash：`b0a978ae16fc386e17795bbcc79d3ff97a1e9fdf44c5742a6070d792acdd0ac8`；
- atomic commit hash：`f3c68d76f25e6e4560ed0d5c8ffb466327269157799a76bb057476d9f0910576`；
- tamper report internal hash：`af772e09d570e504ea54177805473e5d901d57fe3c72fb372623b85519045ce1`；
- tamper report file SHA256：`ef00708080c255b35591f6c1a8a2bf250d6767c0926ee07d2414521f8f731cad`。

## 物理与语义结果

- 1484条显式event独立重放；
- template compile/hit=`1/1`、module move=`0`、scope=`1`；
- optimizer evaluations/updates=`10/9`，full snapshots=`0`，forward builds=`4`；
- KFSB candidate/child batches=`3/3`；
- device candidate/commit/backup/copy=`12/12/12/12`；
- timed candidate D2H=`0`；
- provider core/compute/update和fallback=`0/0/0/0`；
- 24次GPU content hash全部属于post-query audit；headline assembly content digest=`0`；
- post-query audit耗时`5,473,097 ns`，字段明确标记`excluded_from_timing=true`；
- 12条path的before/candidate/committed digest闭环，audit与commit receipt交叉绑定；
- 与FSG3 v5六个冻结B2 control逐项语义一致；
- artifact replay通过，六类outer-resigned counter/journal/semantic/provider/code攻击6/6拒绝。

## 失败历史

source `a3ac761`的首次fresh run在mutation前因opaque provider host字段不可序列化而fail closed；未生成
artifact。修复后的pre-host version绑定完整key inventory、只版本化三项retained字段；candidate/post
仍要求exact三字段。该失败不计入正式证据，记录见
`gemini_doc/change_2026-08-14_fsg4_b3c_host_packet_version_fix.md`。

## 验证

- targeted：`54 passed in 8.77s`；
- full：`1279 passed, 3 skipped, 6 warnings in 450.76s`；
- Black clean；mypy touched source clean；Pylint `10.00/10`；DocOps lint PASS。

## 边界与下一步

本关闭只证明一个fresh真实call的结构激活和正确性，不构成正式性能样本。完整B3计时仍被门禁关闭：
下一步必须完成至少5组fresh B2/B3 correctness pairs；每组都要保存独立raw worker、语义比较、环境
admission、provider/fallback和B3 physical counter。只有5/5通过后，才允许启动36-process B0/B2/B3正式
计时。B4 TIR、B5 JIT、B6 runtime与B7 arena继续关闭。
