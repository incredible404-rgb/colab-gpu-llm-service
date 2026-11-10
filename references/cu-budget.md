# Colab CU 预算

Colab 的可用 GPU、每小时 CU 消耗和套餐额度会变化。估算前先看当天订阅页和运行时页面；下面只是本机 2026-10-19 观察到的样例。

## 公式

```text
可用小时 = 剩余 CU / 每小时 CU
本次费用 = 运行时长（小时） × 每小时 CU
每次运行费用 = 读取和加载时间 × 每小时 CU + 实际使用时间 × 每小时 CU
```

样例：A100 40GB 约 `5.37 CU/小时`，Google AI Pro 每月 200 CU：

```text
200 / 5.37 ≈ 37.2 小时
```

按每次启动 3 分钟准备、实际使用 1 小时估算：

```text
每次约 5.64 CU
200 CU 约可用 35 次一小时会话
```

## 本仓库的 G4 费率（向用户确认花钱时要报的数）

G4（RTX PRO 6000 96GB）实测约 **10 CU/小时**：`debug-runs/2026-11-07-start.log` 与 `debug-runs/2026-11-07-g4-kernel-beat-start.log` 的 `start OK` 行都是 `rate 10.05 CU/h`。冷启动实测 111～140 秒（同两份日志的 `start OK in 111s` / `140s`，模型 `qwen38-fn`），换更大或更慢下载的权重会更久，这段同样计费。

```text
用 1 小时   ≈ 10 + 启动 (2～15 分钟 × 10/60 ≈ 0.3～2.5) ≈ 10～13 CU
用 2 小时   ≈ 20～23 CU
```

费率会变，开机前先 `python -X utf8 scripts/colab_cli.py status` 取当日实时值与余额，以 Colab 页面为准。确认时一并说明：没人用就停，保活进程 45 分钟无推理流量会自动 `colab stop`。

## 重要区别

- CU 按运行时被分配的时间扣，空闲也算。
- Drive 读取、模型加载、预热都算运行时间。
- 释放运行时后，只有 Drive 存储继续存在，CU 不再增加。
- “本月没用完的 CU 能否跨月”以及过期批次，以 Colab 订阅页的当前说明为准；不要凭记忆承诺可累计。
- 释放请求被执行不等于页面立即更新。以“重新连接”和“管理会话”为空作为结束凭据。

## 实时查账：`/tun/m`（零 OAuth）

已登录的 `colab.research.google.com` 标签页里可直接请求下列接口，请求头 `Accept: application/json`、`X-Colab-Client-Agent: colab-cli`，响应前缀 `)]}'\n` 需先剥掉：

| 接口 | 用途 | 关键字段 |
|---|---|---|
| `GET /tun/m/ccu-info?authuser=0` | 余额与速率 | `currentBalance`、`consumptionRateHourly`、`assignmentsCount`、`eligibleGpus`、`eligibleTpus` |
| `GET /tun/m/assignments?authuser=0` | 活跃会话 | `assignments[]`：`endpoint`、`accelerator`、`variant`、`machineShape` |
| `GET/POST /tun/m/unassign/<endpoint>?authuser=0` | 强杀 | GET 返回 `token`；POST 带头 `X-Goog-Colab-Token: <token>` |

`python -X utf8 scripts/colab_meter.py status` 一行给出 `Active assignments: N | Burn rate: X.XX CU/h | Balance: YYY.YY CU | Eligible GPUs: ...`。示意输出：`0 active / 0.00 CU/h / <你的余额> CU`，可用 GPU 取决于账号（例如 `H100, G4, A100, L4, T4`）。

## 双轨对账

```text
轨道 1（云端账单）  max(0, 开机余额 - 关机余额)
轨道 2（墙钟折算）  时长秒数 / 3600 × 每小时 CU
```

`colab_meter.py start` 记录开机余额与时间，`stop` 强杀并打印两条轨道。云端余额可能比真实消耗滞后几分钟：短会话里轨道 1 偏小，轨道 2 更接近预期；长会话两者应接近。差距持续很大时，先看 `status` 是否仍有活跃 assignment。

## 幽灵实例强杀

关掉笔记本页面或浏览器崩溃，不会释放运行时，`consumptionRateHourly` 仍大于 0。`python -X utf8 scripts/colab_meter.py stop --all`（或 `python -X utf8 scripts/colab_sync.py --release-all`）对每个 `endpoint` 做 GET 取 token、POST 释放两步握手，等 1.5 s 重查，最多 3 次，直到 `assignmentsCount == 0` 且 `consumptionRateHourly == 0`。结束凭据是这个查询结果，不是命令返回码。
