# 2026-08-13 — FSG3 Lower-Only Upper Sentinel Pre-Run Amendment

## 原因

V4-3正式路径为lower-only，upper使用`+inf`未请求哨兵；FSG3预注册误写成upper必须全部finite。该问题在
任何FSG3 real worker或timing结果产生前发现。

## 修订

- lower继续要求全部finite；
- upper改为finite canonical payload + exact positive-infinity mask；
- mask位置payload固定为`0.0`，NaN/`-inf`/未掩码`+inf`继续拒绝；
- replay跨B0/B1/B2及control/profile核对mask exact，只比较非mask位置的finite数值。

## 边界

不改变36-run顺序、计时scope、统计方向、环境/扰动门禁或后续状态机；没有读取性能数字。

## 验证

- FSG3 schema targeted=`14 passed`；
- Black/mypy clean、Pylint=`10.00/10`；
- post-amendment full regression显式延后到紧接的real-worker切片统一执行，不形成性能claim。
