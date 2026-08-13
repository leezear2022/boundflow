# RVIR-v4 V4-3C Native KFSB Formal Closure

日期：2026-08-13

## 结论

V4-3C以`VALIDATED-NATIVE-KFSB`关闭，V4-3D live return assembly准入。

本阶段证明BoundFlow可由terminal native α/split、V4-3B lA、共享pre-result intermediate bounds和
query input/spec独立生成production KFSB的mask、三组top-3候选、72个child lower与最终decision。
它尚未构造真实`UpdateBoundCoreReturn`、提交live provider-owned state或接入official post/queue；
V4-3整体、V4-3E、B2与性能claim继续关闭。

## 正式身份

- source commit：`a2097c02da368378da5960c51fa54322d9494013`；
- artifact：`artifacts/rvir-v4-native-kfsb/resnet2b-core-v1`；
- manifest file SHA256：`28e4da09536c0c21fe08e4c3e5b8bbb3723dafd1679b6a2292ffc31f8f492ed8`；
- manifest internal hash：`405a03bed21763cd02afe9bf9466711b14ac5349de870b711fe10cee4cc661f3`；
- evaluation/summary hash：`2af0a666…2fe2` / `786550dc…1a0`；
- evaluation file SHA256：`489c9ebb977051a9aefecb5e13f15ee3e2b18005ad0542991d22ccedc80eb45d`；
- tamper report SHA256：`c197b5d5a80dd17d920207fd549f1ae462828bce5c7df7cba31f0a88d0f445f9`；
- frozen V4-2E/V4-3A source manifest：`b76ee573…0136` / `0e6ed721…9818`。

## 正式结果

- 六层unstable mask逐元素exact：shape总元素37464、true共4200；
- 三组top-3 candidate split共36项，层号、neuron index和顺序全部exact；
- 每组执行24个child domain，合计72个child lower；shape/dtype/sign exact，最大绝对差
  `3.0994415283203125e-06 <=2e-4`；
- final decision exact：`[[5,27],[5,32],[5,90],[5,90],[5,32],[5,90]]`；
- provider core/`compute_bounds`/`update_bounds` callback=`0/0/0`，fallback=`0`；
- V4-3A truth只进入独立comparator，candidate生成不读取truth candidates、child lower或final decision。

## 篡改门禁

六类recorded evaluation full resign均被native semantic reexecution拒绝：candidate split、child lower、
final decision、unstable mask、top-k score和candidate reduction。每类均同步重签typed metadata、summary、
replay stdout、file digests和outer manifest。

另外topology与V4-3A truth source两类outer resign也被拒绝。八类攻击8/8拒绝。

## 验证

- related targeted：`16 passed`；
- full：`1187 passed, 3 skipped`；三个skip均为既有TVM重复编译或frozen VNN-COMP checkout边界；
- mypy：三个相关source clean；
- Pylint：相关source/tests=`10.00/10`；
- Black、`git diff --check`、artifact replay与DocOps validate/lint均通过。

## 设备与claim边界

reference truth仍来自RTX 4060 Laptop真实αβ-CROWN生产调用；V4-3C正式native evidence为CPU
single-thread deterministic semantic replay。该结果证明KFSB策略与child-bound所有权，不证明live GPU
host integration、GPU kernel接管、时延或speedup。`performance_claimed=false`。

## 下一动作

只启动V4-3D：在真实αβ-CROWN GPU进程内消费V4-2 terminal copy-out、V4-3B backward export和V4-3C
branch decision，构造完整`UpdateBoundCoreReturn`并交给未修改的official `update_bounds_post`/queue；
必须保持provider core/compute_bounds/update_bounds/fallback=`0/0/0/0`，且任一失败恢复live tensors与host
packet。V4-3E five-fresh与B2 timing仍不得提前启动。
