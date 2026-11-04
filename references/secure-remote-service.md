# 固定地址与鉴权

目标是把“每次启动换 URL”变成稳定入口，同时让未授权请求无法消耗 GPU。

## 为什么只固定域名不够

Quick Tunnel 的随机域名会让扫描器不容易命中，但它没有身份认证。Named Tunnel 只解决入口稳定性；如果 `llama-server` 仍无 API key，域名固定后反而更容易被持续调用。

最低组合是：

1. Named Tunnel 提供固定 hostname。
2. `llama-server` 用随机 API key 拒绝未授权请求。
3. Web 界面另加 Cloudflare Access 登录。

## 准备

需要一个已经托管在 Cloudflare 的域名。建议使用两个子域：

```text
qwen.example.com      网页
qwen-api.example.com  OpenAI 兼容 API
```

API 和网页分开，因为 Cloudflare Access 的浏览器登录会干扰 `pi` 等普通 OpenAI 客户端。

## 1. 创建 Named Tunnel

在 Cloudflare Dashboard：

1. 进入 **Networking → Tunnels**（Zero Trust 侧栏里也可能显示为 **Networks → Tunnels**）。
2. 新建 **Cloudflared** tunnel，命名例如 `qwen-colab`。
3. 复制 token。
4. 添加两个 Public Hostname：

```text
qwen.example.com      -> http://127.0.0.1:8081
qwen-api.example.com  -> http://127.0.0.1:8081
```

tunnel token 会授权连接器代表这个 tunnel 接流量，按密码处理。泄露后在 Dashboard 轮换。

## 2. 在 Colab Secrets 保存密钥

左侧钥匙图标 → Secrets，添加并勾选 notebook access：

```text
QWEN_API_KEY=用下面的命令生成
CLOUDFLARE_TUNNEL_TOKEN=Dashboard 给出的 token
```

生成 API key：

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

不要把真实值写进 notebook、Git、聊天记录或截图。

## 3. 启动带鉴权的 llama-server

从 Colab Secret 读取 key，写入仅当前用户可读的临时文件：

```python
import os
from pathlib import Path
from google.colab import userdata

api_key_file = Path("/content/qwen27b-pi/.llama-api-key")
api_key_file.write_text(userdata.get("QWEN_API_KEY") + "\n", encoding="utf-8")
api_key_file.chmod(0o600)

command += ["--api-key-file", str(api_key_file)]
```

`llama-server` 也支持 `--api-key`，但文件形式更适合避免密钥出现在进程参数里。

## 4. 启动 Named Tunnel

```python
import os
import subprocess

tunnel_env = os.environ.copy()
tunnel_env["TUNNEL_TOKEN"] = userdata.get("CLOUDFLARE_TUNNEL_TOKEN")
tunnel = subprocess.Popen(
    ["/content/qwen27b-pi/cloudflared", "tunnel", "--no-autoupdate", "run"],
    stdout=tunnel_log,
    stderr=tunnel_log,
    env=tunnel_env,
)
```

固定入口：

```text
网页：https://qwen.example.com
API： https://qwen-api.example.com/v1
```

运行时重启后 hostname 不变；tunnel token 和 API key 也不变。

## 5. 保护网页

Cloudflare Dashboard → **Zero Trust → Access controls → Applications → Add an application → Self-hosted**：

```text
Application: qwen web
Hostname: qwen.example.com
Policy: Allow
Include: Emails = 你的邮箱
```

不要把这个 Access application 套到 `qwen-api.example.com`。API 侧由 Bearer API key 保护；如果还要 Access Service Token，需要确认客户端能发送
`CF-Access-Client-Id` 和 `CF-Access-Client-Secret` 两个 header，否则先只用应用层 API key。

## 6. 验证

未带 key 必须失败：

```powershell
curl.exe -i https://qwen-api.example.com/v1/models
```

带 key 必须成功：

```powershell
$headers = @{ Authorization = "Bearer $env:QWEN_API_KEY" }
Invoke-RestMethod https://qwen-api.example.com/v1/models -Headers $headers |
  ConvertTo-Json -Depth 5
```

然后发一条最小聊天请求，并让 `pi` 完成一次真实工具调用。只验证 `/v1/models` 不够，它不能证明聊天模板和工具调用正常。

## 7. `pi` 配置

```powershell
pwsh -NoProfile -File scripts/update-pi-provider.ps1 `
  -BaseUrl https://qwen-api.example.com `
  -ApiKey $env:QWEN_API_KEY `
  -ServiceName qwen38-colab `
  -ModelId Qwen3.8-27B-pi-Q4_K_M
```

固定域名之后，这个配置不再需要每次启动重跑。

## 没有自有域名时

- 继续 Quick Tunnel，但必须加 API key；地址每次启动变化，用更新脚本改 provider。
- 用 ngrok 免费计划的账号级 dev domain（形如 `name.ngrok-free.app`）：免费计划只送这一个固定域名，`ngrok http 8081 --url https://name.ngrok-free.app` 启动后地址不再变化。免费计划有流量上限，HTML 首次访问会显示 ngrok 提示页，API 客户端需带 `ngrok-skip-browser-warning` header 或改用付费计划。
- 用 Tailscale Funnel：地址是固定的 `<machine>.<tailnet>.ts.net`，仍需应用层 API key。
- 用服务商提供的保留域名（例如付费/免费保留的 ngrok 域名），同样必须鉴权。

这些替代方案都只改变入口，不替代 API key。

## 免费域名的可行性（2026-10 核实）

Named Tunnel 需要一个能作为独立 zone 加进 Cloudflare 的域名。判断标准是它是否在 Public Suffix List 里：在册的 `foo.pp.ua`、`foo.eu.org`、`foo.dpdns.org` 能被 Cloudflare 当作可注册域接受；只在别人 zone 下的普通子域（`foo.us.kg` 之类被挂起的、或纯 CNAME 服务）不能。

| 方案 | 成本 | 交付周期 | 与 Cloudflare 配合 | 风险 |
|---|---|---|---|---|
| GitHub 学生包的 Namecheap `.me` | 首年免费，续费约 $5/年 | 几分钟 | 可改 NS，正常做 Named Tunnel | 需要在校学生身份，一年后付费 |
| GitHub 学生包的 Name.com 后缀 | 首年免费 | 几分钟 | 同上 | 同上 |
| `.pp.ua`（nic.ua） | 免费，每年续 | 手机激活后即时 | 在 PSL 内，可托管到 Cloudflare | WHOIS 强制公开真实信息，30 天内最多 3 个 |
| EU.org | 永久免费 | 人工审核，通常数天，偶有数周 | 在 PSL 内，可托管到 Cloudflare | 注册前必须先有可用 NS；非商业用途为主 |
| `dpdns.org`（DigitalPlat） | 免费 | 即时 | 在 PSL 内 | 与被称为 abuse 温床的 `us.kg` 同一套基础设施，曾被注册局停解析 |
| ngrok 免费 dev domain | 免费 | 即时 | 不能用 Named Tunnel，直接当入口 | 1 GB/月、2 万请求/月，HTML 有提示页 |
| Tailscale Funnel | 免费（全套餐） | 即时 | 不能用自有域名 | 仅 443/8443/10000，仅 `*.ts.net`，带宽受限且非可配 |

不要用 Freenom 的 `.tk/.ml/.ga/.cf/.gq`：2023 年起已停止免费注册，存量域名在被回收。

学生包的代价是实名：GitHub Education 会把账户姓名与在校材料对齐，不愿改真名时不必强求。付费替代很便宜，按 2026-10 的价格：

| 渠道 | 首年 | 续费 | 说明 |
|---|---|---|---|
| Spaceship `.com` | $8.88 | $9.98 | 续费不跳价，改 NS 后可直接做 Named Tunnel |
| Cloudflare Registrar `.com` | 约 $10.44 | 同价 | 按成本价卖，域名天然在 Cloudflare，少一步迁移 |
| IONOS `.com` | $1 | 约 $20 | 只适合短期，长期不划算 |

每年十美元量级的 `.com` 在稳定性和可信度上优于任何免费后缀，个人服务用一年也就一顿饭的钱。

## Tailscale Funnel 的实测边界

- 官方定位是 beta，所有套餐都能用；要求 MagicDNS、tailnet HTTPS 证书，以及 policy 文件里的 funnel node attribute。
- 公网端口只有 443、8443、10000；地址只能是 `<device>.<tailnet>.ts.net`，**不支持自有域名**。
- 流量经过 Tailscale 的 relay，官方写明有"不可配置的带宽限制"；第三方实测中位数约 7 Mbps（厂商对比博客，非中立基准），对 token 流式聊天够用，传大文件不合适。
- Colab 容器没有 `/dev/net/tun`、也没有 `CAP_NET_ADMIN`（tailscale#634），必须跑 `tailscaled --tun=userspace-networking`；该模式下 funnel 能否稳定拿到证书尚未在本机验证，用前先在免费 CPU 运行时试一次。
- Funnel 是公开入口，仍需 `--api-key-file`；它没有 Cloudflare Access 那种身份层。
