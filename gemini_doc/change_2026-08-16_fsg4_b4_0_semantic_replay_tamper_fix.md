# FSG4/B4-0 Semantic Replay 与 Outer-Resigned Tamper 修复记录

日期：2026-08-16  
状态：`FIXED-VALIDATED-PENDING-CLEAN-SOURCE-FORMAL-ARTIFACT`  
性能声明：`performance_claimed=false`

## 1. 触发事实

解释器修复后的fresh control/profile均成功完成，但初版derive用JSON完全相等比较semantic，因6个
`lower_values`的GPU浮点微差而拒绝。字段级复核：最大绝对差
`9.5367431640625e-07`，其余decision、sign、shape、queue、history、split、status全部exact。

该差值远小于B3正式协议已冻结并通过外审的`atol=rtol=2e-4`。因此失败属于B4 runner比较器与既有
B3协议不一致，不是求解语义失败。

## 2. 修复

- 直接复用B3 typed semantic parser/comparator及`2e-4/2e-4`阈值；
- 离散状态继续exact，额外要求lower sign exact；
- protocol绑定atol/rtol，summary保存max abs diff、sign/discrete/failure count；
- 新增9类outer-resigned tamper：marker count、raw phase、raw ordinal、raw duration、raw delete、
  semantic lower、protocol code revision、worker kind、summary opportunity；
- 每类攻击均重算被修改payload的内层hash以及protocol/file inventory/manifest外层digest，拒绝不得只靠
  文件SHA不匹配；正式generate自动运行9类攻击并把report纳入最终manifest/replay。

## 3. 验证

- B4 targeted：`15 passed`；
- B3 frozen replay：PASS，summary hash=
  `4c19afd43c18e0409932b86506efdaf6bfc3e07baabcc222dbe79c8149f99bac`；
- B3/B4相关：`54 passed`；
- full：`1329 passed, 3 skipped`；
- Black、Mypy、Pylint `10.00/10`、`git diff --check`：PASS。

## 4. 边界

上述完整raw来自旧source `fe04b52`，只用于定位并验证门禁修正，不得作为正式artifact。下一步必须从
包含本修复的新clean source重新生成control/profile、执行9/9 tamper与root replay。当前仍无B4
opportunity closure或性能claim。
