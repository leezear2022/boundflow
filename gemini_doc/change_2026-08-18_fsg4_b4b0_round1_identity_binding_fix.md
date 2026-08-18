# FSG4/B4-B0 Round 1 Identity Binding 修复记录

日期：2026-08-18

状态：`IMPLEMENTED-B4-B0-R1-F1-IDENTITY-BINDING-PENDING-V2`

## 外审 finding

Round 1为`request_changes`，F1=`major`。原replay只比较5次run相互一致；若同步改写全部
10个capture的`topology_hash`或lineage source tensor hashes并重签内部/PT/manifest，仍会被接受。
因此原`VALIDATED-B4-B0-FIVE-FRESH-PENDING-EXTERNAL-AUDIT`仅是内部判定，已被外审否决。

## 修复

- artifact/protocol/summary升级v2，同时保留合法v1 artifact的只读replay兼容；
- 在verifier代码中冻结source capture/model、source state、primal graph、split、topology、schedule、
  两锚点anchor hash、lineage hash、全部lineage source tensor与round-trip receipt hashes；
- v2 protocol携带完整`frozen_source_identity`，manifest绑定其canonical hash；
- manifest与protocol的source commit/code revision必须相同；
- 每个run及每个capture都必须匹配绝对冻结身份，不再只与run0比较；
- 新增coordinated-all-runs topology与lineage两类完整性负向用例。

## 当前验证

- 合法v1 raw artifact仍replay通过，summary hash不变；
- 原9类加两类coordinated用例=`11/11 rejected`；
- artifact测试含两类协调一致改写回归=`4 passed`；
- Mypy clean，Pylint 10.00/10。

## 下一步与边界

下一唯一动作是在本提交冻结后生成clean-source v2 five-fresh artifact、执行root replay和11类
完整性负向用例，再重跑回归并提交DocOps Round 2。B4-B0仍未获外审批准；B4-B1/B4-B2/TIR/
performance全部关闭。
