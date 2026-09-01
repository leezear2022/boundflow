# RVIR-v4 V4-3B Native Backward Export Formal Closure

日期：2026-08-13

## 结论

V4-3B以`VALIDATED-NATIVE-BACKWARD-EXPORT`关闭，V4-3C native KFSB candidate evaluation准入。

本阶段证明BoundFlow可从terminal native α/β/split和共享pre-result intermediate bounds独立导出六层
lA及最终lower；它尚未替换KFSB、whole `update_bounds_core`或live GPU host integration。V4-3、B2与
性能claim继续关闭。

## 正式身份

- source commit：`762b642b7cb063d36a72aec018ff4d6f32e99b0c`；
- artifact：`artifacts/rvir-v4-native-backward-export/resnet2b-core-v1`；
- manifest file SHA256：`110dfd637ca8444ff9f9341762578161bc47812554dce27dffa84022713c8269`；
- manifest internal hash：`e253dffc338608035d8957d27e0ab87a33b6f2982b5a0c04cfd27676627ad490`；
- export/summary hash：`0d59ec1c…e276` / `4efcd694…08a0`；
- tamper report SHA256：`4cdd2231a44a037a7fe7429a72021ea97151c686c38fe30364ac70e0ba7ce355`；
- frozen V4-2E/V4-3A source manifest：`b76ee573…0136` / `0e6ed721…9818`。

## 正式结果

- 六层lA shape/dtype与sign exact，最大绝对差`9.238719940185547e-07`；
- 六层intermediate lower/upper共12 tensors，shape/dtype与sign exact，最大绝对差
  `6.079673767089844e-06`；
- final lower sign exact，最大绝对差`3.0994415283203125e-06`；
- 全部满足`atol=rtol=2e-4`；
- provider core/`compute_bounds`/`update_bounds` callback=`0/0/0`，fallback=`0`；
- export明确标记intermediate来源为`shared-pre-result-external-bounds`，不虚构为本阶段重新求紧的
  native intermediate refinement。

## 篡改门禁

- recorded lA full resign：拒绝；
- recorded intermediate full resign：拒绝；
- recorded final lower full resign：拒绝；
- topology outer resign：拒绝；
- V4-3A truth source outer resign：拒绝。

前三类均同步重签export metadata、file digest和outer manifest后，由native semantic reexecution拒绝；
五类攻击全部拒绝。

## 验证

- targeted V4-3B/V4-3A/V4-2E：`9 passed`；
- full：`1183 passed, 3 skipped`；
- mypy：3个相关source clean；
- Pylint：相关source/tests=`10.00/10`；
- Black、`git diff --check`、DocOps validate/lint均通过。

## 设备与claim边界

V4-3A reference truth来自RTX 4060 Laptop GPU真实αβ-CROWN生产调用；V4-3B formal native export为
CPU single-thread deterministic semantic replay。它证明跨实现数值/结构所有权，不证明GPU kernel接管、
GPU时延或speedup。真正的live GPU candidate接入仍在V4-3D，B2仍不得启动。

## 下一动作

只启动V4-3C：复刻KFSB top-3候选、用BoundFlow执行三组24-child lower，逐candidate与final decision
对照V4-3A truth。不得调用provider `LiRPANet.update_bounds`，不得用truth child lower作为candidate输入，
不得同时引入TIR/JIT/fusion或性能变量。
