# RVIR-v4 V4-3E / V4-3 Formal Closure

日期：2026-08-13

## 结论

V4-3E以`VALIDATED-FIVE-FRESH-CORRECTNESS`关闭；V4-3 whole-core replacement整体以
`VALIDATED-WHOLE-CORE-REPLACEMENT`关闭。B2 same-solver counterbalanced timing现已准入。

这并不是性能结论。十个fresh进程只记录correctness raw payload，不将worker wall、stdout或artifact
生成时延用于speedup。B2必须另按FSG3 measurement contract执行至少5个fresh AB/BA pairs。

## 正式身份

- source commit：`17d2d61df8648ceeac8b176a01192279c4b522b4`；
- artifact：`artifacts/rvir-v4-five-fresh/resnet2b-prop0-v1`；
- manifest file SHA256：`ca37bd56573b668006007a7edca1027a4136735fc1f0aede108843e2d8a4ada2`；
- manifest internal hash：`a745532bf9beeb0835e41ef0680e30790039f4bc14ff0768d38872f0ca95023a`；
- summary file/internal SHA256：`4b33f060909b3d628d0d0b44bd5e6cbf230e63e0c30d0927f60fb7e6e80d2242` /
  `cf0bec5aaa1fd90ecd26c8036ebb9a7f03d314b3c47bbd450df9676f16666adf`；
- tamper report SHA256：`bc41cde554c45d4334ba11e00bea609679c0d344f7ae7cf013c61db5d8129fc2`。

## 正式协议与结果

- 实际启动10个互相隔离的GPU进程，顺序exact=`O,C,C,O,C,O,O,C,O,C`；
- 五对映射exact=`(0,1)/(3,2)/(5,4)/(6,7)/(8,9)`，包含3个OC与2个CO顺序；
- 5/5 pairs的完整core/post/state/branch semantic parity通过；合计比较2255 tensors、1,065,300个
  float signs，sign exact，最大绝对差`1.0669231414794922e-05 <=2e-4`；
- 每个run queue before/input/accepted/pruned/after=`0/6/6/0/6`，depth全1；
- 每个run status/success/visited=`verified/true/[6]`，n_verified/n_splits=`0/6`，final decision exact；
- 五个original各24次provider call，共120次；五个candidate的provider core/compute/update callback与
  fallback全部为0；
- `five_fresh_correctness_admitted=true`、`whole_core_replacement_admitted=true`、
  `b2_same_solver_timing_admitted=true`、`performance_claimed=false`。

## Replay与篡改门禁

- static replay从十份raw payload重新构造每个run、pair与aggregate summary，逐文件digest与code
  provenance全部通过；
- 第三方solver原始`*.stdout.txt`按字节保留并受manifest digest约束；`.gitattributes`将其标记为
  `-diff`，避免Git把上游原始尾随空格误作本仓维护文本错误；
- candidate lA、original lA与candidate decision三类攻击重签内部truth hash及外层artifact；queue
  accounting、candidate callback与sequence order三类攻击重签outer artifact；6/6全部fail closed。

## 验证

- closing targeted（five-fresh + live-return）：`8 passed`；
- full：`1200 passed, 3 skipped`；
- Black、三个相关script mypy clean、Pylint=`10.00/10`；
- formal ten-process generate、static replay、tamper suite与DocOps在最终交接前全部验证。

## Claim边界与下一动作

V4-3关闭的是固定ResNet2B property 0、固定max-iteration=1 production whole-core replacement
correctness。它不证明多轮/完整query、多workload、时延、显存、TIR/JIT/fusion或ASPLOS system outcome。

下一动作是FSG3/B2 same-solver timing preregistration与执行：保持同一solver/config/GPU，分离cold setup与
measured core，至少5 fresh counterbalanced pairs，记录GPU排他、clock/power/temperature/background、
paired median/range/MAD/geomean及完整correctness gate。B2之后才根据全栈profile决定B3—B7的累积优化
顺序；不得直接宣称speedup。
