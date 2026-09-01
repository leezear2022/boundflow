# 2026-08-13 — FSG3 Profile Schema v2 Pre-Run Amendment

## 原因

FSG3-1的v1 raw run只保存`profile_closure_error/profile_residual_share`两个投影值，没有把产生投影的
span原始区间纳入canonical payload与stable hash。它可以检查阈值，却不能让第三方从raw evidence重算
分层归因、互斥性和closure，不能满足预注册的`profile_spans.jsonl`要求。

## 修正

- schema升级为`boundflow.fsg3-same-solver-timing/v2`；
- 每个profile run保存有序typed spans：scope/name、stack layer、solver phase、resource、cache state、
  monotonic start/end offset、wall与CUDA event duration；
- replay冻结各配置布局并拒绝删项、换序、重叠、wall投影篡改；
- closure明确只对exact `update_bounds_core`计算：
  `abs(core_wall - sum(core spans))/core_wall`；正向未覆盖部分另记residual；
- B2 compile和official post/queue保持独立scope，分别相对cold total和query wall报告share，不被伪装成
  core，也不把未由本阶段接管的solver初始化误算成core残差；
- summary从profile raw spans重算每层6-run wall/GPU与scope share。
- environment raw绑定Python/Torch/CUDA/cuDNN/driver/GPU memory runtime identity，跨run变化fail closed。

## 冻结布局

- B0：`provider_core | official_post_queue`；
- B1：`typed_pre_state | provider_core | official_post_queue`；
- B2：`compile | typed_pre_state | optimizer | backward | kfsb | atomic_commit |
  official_post_queue`。

## 时间边界与主张

本修订发生在正式36-process artifact之前。此前仅执行了不保存入仓、不产生performance claim的worker
smoke；其中control验证三路物理路径，profile smoke用于发现并关闭v1证据缺口。B1/B2 profile曾并行
执行，环境门禁按设计标记不准入；这些值不得进入正式统计。

修订不改变配置、顺序、重复数、正确性阈值、profile扰动阈值、closure/residual阈值或headline只用
control的规则。
