# 修改记录：MR3-0 Provider Hook Tamper Probe

> 日期：2026-08-26  
> 状态：实现完成，待 formal artifact 生成后执行

## 修改

- 新增 12 类 fully re-signed artifact attack；
- 覆盖 source/order/count、empty β、ReLU→Conv adjacency、α ABI、CUDA stream、outer result、
  target α、replacement count 与 missing hook；
- 每类攻击均重签 worker hash、raw digest 与 manifest hash，replay 必须依靠语义重算拒绝；
- 将 tamper probe 纳入 formal code revision，避免 artifact 生成后才补验证器。
- tamper report 现为 formal manifest 必需文件；生成阶段先写 `pending`，12 类攻击完成后原子替换为
  `validated` 并重签 manifest，最终 replay 会独立校验 report 自身 hash/count/result ledger。
