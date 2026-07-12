# 变更记录：修复 αβ 多首层 halfspace 的 autograd graph 重用

## 问题

PR-10 profile 在 add+concat DAG、domain batch 1、固定 αβ split 下稳定触发：

```text
Trying to backward through the graph a second time
```

infeasibility detector 会在多个首层 split halfspace 上优化 simplex logits，但 `a_mat/c_vec`
仍连接模型参数计算图；200 次 optimizer iteration 因而重复反传同一图。

## 修复

halfspace 系数对 certificate optimizer 是固定问题数据，只优化 logits，因此在堆叠后显式
`detach()`。这不改变 certificate 数学，也不阻断 α/β 主求解路径的 gradient。

## 回归

新增两个并行首层卷积分支、两个 split halfspace 的 batch=1 测试，同时覆盖 detector 与完整
αβ oracle。
