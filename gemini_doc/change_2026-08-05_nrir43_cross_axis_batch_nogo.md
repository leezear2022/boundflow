# NRIR-43 Cross-Axis Verification Batch Schedule Phase A NO-GO

## 变更

- 新增 typed ragged batch Plan/Instance/Task/Schedule/Trace 与 Phase-A evidence IR；
- 新增把多个 compile-owned scorer capsules 合成单一 child-lower launch 的 runtime；
- 新增 additive 31-node production queue，使 root 单节点与 15 个 sibling 双节点组分别走 typed batch；
- 新增 three-process counterbalanced generate/replay、完整 shard/manifest 与同步重哈希篡改单测；
- NRIR-42 frozen 源码、artifact 和 production 默认路径均未修改。

## 正式结果

- 两条 clause、三轮共 6 组 semantics exact；
- 每条 scorer physical launches 从 31 降为 16；
- clause 2 median ratio=`1.051134`，clause 3=`1.044573`，均未过 `<=0.85`；
- formal hash=`692b9e273661fce9f12129e134550547afa4023361e2a79d751c437c92f30390`；
- targeted `10 passed`，全量 `968 passed, 37 skipped`，Black/mypy/Pylint `10.00/10`。

## 判定

NRIR-43 以 `VALIDATED-NO-GO` 关闭；Phase B 不启动。结果证明 CPU 上减少发射次数不能替代真实
wall-clock gate。下一单变量转 NRIR-44 Root-Projection Floor Schedule：由 compiler consumer/liveness
contract 消除 ranking 不消费的深层 floor queue work。
