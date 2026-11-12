# Tailscale Service：固定地址、仅 tailnet 内可见

适合“只有自己和自己的设备要访问”的场景：笔记本上的 `pi`、手机浏览器、家里另一台机器。
地址固定为 `qwen.<tailnet>.ts.net`（`svc:qwen` 这个 Service 的域名），只有登录同一 tailnet 的设备能连，
不用买域名、不用 Cloudflare。域名指向 Service 的虚拟地址，和每次新建运行时拿到的节点名彻底解耦。

Service 名默认是 `qwen`。想用别的名字，在 `.secrets.json` 里设 `service_name`（环境变量 `TS_SERVICE_NAME`），并让 Tailscale 里建的 Service、ACL 里的 `svc:...` 和 `tailnet_fqdn` 第一段同名；下文的 `qwen` / `svc:qwen` 都按这个名字替换。

本文的“运行时里的流程”就是 `scripts/colab_kernel.py` 里 `prepare_tailscale()` 实际执行的步骤，
2026-10 在 Colab 运行时上实测通过（Tailscale 1.104.1）。旧的“主机名方案”已废弃，留在文末附录供排查参考。

## 一次性配置（管理台，四步）

> Tailscale Services 目前是 beta 功能：先在 Admin Console 的 Services 页面（`admin/services`）启用它，才能建 `svc:qwen`。

1. **DNS 设置里开两项**（`admin/dns`）：开启 **MagicDNS**；在 HTTPS Certificates 区块点 **Enable HTTPS**，
   并确认 tailnet 域名（`<tailnet>.ts.net`）已显示。少了这两项，Service 域名解析不到或 HTTPS 证书拿不到。

   `tailscale serve` 首次运行如果打印
   `Serve is not enabled on your tailnet. To enable, visit: https://login.tailscale.com/f/serve?node=<nodeid>`，
   打开该链接，取消勾选可选的 **Tailscale Funnel**（不需要公网入口），再点 **Enable HTTPS**，
   页面出现 *Tailscale Serve is ready to use* 即可。

2. ACL 加标签与自动审批（`admin/acls/file`）：

   ```json
   "tagOwners": { "tag:colab": ["autogroup:admin"] },
   "autoApprovers": {
     "services": { "svc:qwen": ["tag:colab"] }
   }
   ```

   `autoApprovers.services` 是服务主机的自动审批；没有它，每个新节点的
   `tailscale serve --service` 都要在管理台点一次 Approve。

3. `admin/services` 新建 Service：名字 `qwen`、端口 `tcp:443`、标签 `tag:colab`。
   域名固定为 `qwen.<tailnet>.ts.net`（`<tailnet>` 换成你自己的 tailnet 名）。

4. `admin/settings/trust-credentials` 建 OAuth 客户端：scope 选 **Auth Keys: Write**，
   标签填 `tag:colab`。拿到 `client_id` 和 `tskey-client-...` 的 secret，填进 `.secrets.json`
   的 `ts_client_id` / `ts_client_secret`。secret 不过期；换掉/删掉这个客户端即可吊销。

客户端（Windows/Mac/手机）需要 1.78+ 才能解析 Service VIP。

## 运行时里的流程（`colab_kernel.py` 已写进笔记本「一键启动」）

先用 OAuth 客户端现签一把一次性 key：

```python
token = requests.post('https://api.tailscale.com/api/v2/oauth/token',
                      data={'client_id': CID, 'client_secret': CSEC}, timeout=30).json()['access_token']
key = requests.post('https://api.tailscale.com/api/v2/tailnet/-/keys',
                    headers={'Authorization': 'Bearer ' + token},
                    json={'capabilities': {'devices': {'create': {
                        'reusable': False, 'ephemeral': True, 'preauthorized': True,
                        'tags': ['tag:colab']}}}}, timeout=30).json()['key']
```

再起 daemon 并发布 Service（内核里的真实参数；`tailscale` 命令都带 `--socket=/var/run/tailscale/tailscaled.sock`，下面省略）：

```bash
tailscaled --tun=userspace-networking --socks5-server=localhost:1055 \
           --state=/content/tailscaled.state --statedir=/content/tsstatedir \
           --socket=/var/run/tailscale/tailscaled.sock          # 必须 start_new_session=True，见“两个必踩的坑”
tailscale up --auth-key=<key> --hostname=<模型的 ts_hostname> --advertise-tags=tag:colab --accept-dns=false --timeout=60s
tailscale serve reset
tailscale serve --bg --service=svc:qwen --tcp=443 tcp://127.0.0.1:8081    # Service 的 :443 是原始 TCP，直通 llama-server（HTTP）
tailscale serve --bg --https=443 http://127.0.0.1:8081                     # 节点自己的 https 名，仅调试用
```

要点：

- Service 的 `:443` 是**原始 TCP 转发到 llama-server 的明文 HTTP**，所以访问地址是 `http://qwen.<tailnet>.ts.net:443/`，
  不是 `https://`。对这个名字发 HTTPS 永远不会完成握手。
- `ephemeral` 让节点下线后自动从 tailnet 回收，不会越攒越多，也不会再抢名字。
- 每次运行现场签发一把一次性 key，长期 secret 只有 OAuth 客户端那一份。
- `serve` 首次运行会打印 “approval from an admin is required”，随后被
  `autoApprovers` 自动放行；**域名在审批生效后几秒到十几秒内可用**，
  切换瞬间的一次请求可能超时，重试即可。
- 若配置了 `local_tailscale_ip`，内核另起一个线程把运行时 `127.0.0.1:3426` 经 `--socks5-server=localhost:1055`
  反连到你本机的 `local_novel_mcp.py`；没配则不起这个线程。

## 两个必踩的坑

1. **Colab“中断执行”会杀掉 daemon。**
   中断会向整个进程组发 SIGINT，日志里是
   `tailscaled got signal interrupt; shutting down`。
   凡是 `Popen` 起的常驻进程（`tailscaled`、`tailscale up`、`llama-server`）都加
   `start_new_session=True`，并显式传 `stdin=subprocess.DEVNULL`。

2. **缺 `--statedir` 会导致 HTTPS 握手失败。**
   只给 `--state` 时，`tailscale cert` 报
   `500 Internal Server Error: no TailscaleVarRoot`，
   外部访问表现为 `TLS alert: InternalError`，看日志很像网络问题。
   补上 `--statedir=/content/tsstatedir` 重启 daemon，`tailscale cert <域名>` 应返回
   `Wrote public cert to ...`。

另外：`serve` 在 HTTPS 尚未开启时不会报错退出，而是一直挂起等待你在管理台点确认
（脚本里表现为 `rc=124` / 超时）。这属于预期行为，回到“一次性配置”第 1 步。

## 验证

**必须从另一台 tailnet 设备验证**。Colab 容器是 userspace 网络，没有 `tailscale0`，
在容器里 `curl` 该域名只会拿到 `000`，不能当作失败。

```powershell
Invoke-WebRequest -Uri 'http://qwen.<tailnet>.ts.net:443/' -TimeoutSec 30      # 200
Invoke-RestMethod -Uri 'http://qwen.<tailnet>.ts.net:443/v1/models' -Headers @{Authorization="Bearer $key"}
```

模型起来后同一地址就是 OpenAI 兼容接口：`http://qwen.<tailnet>.ts.net:443/v1`。
各模型笔记本不再给 `llama-server` 传 `--api-key-file`：`svc:qwen` 只在 tailnet 内可见（未开 Funnel），网页端无需粘贴 key，只剩 tailnet 身份这一层。也就是说 `.secrets.json` 的 `api_key` 在服务端不校验：tailnet 内任何设备都能免密调用 `/v1`，配了文件桥时还能经 `--ui-mcp-proxy` 的 MCP 代理读写本机 `novel_root`。把 tailnet 共享给别人之前先收紧 ACL（只放行自己的设备）或加回 `--api-key-file`。任何要上 Funnel、Quick Tunnel 或其他公网入口的场景必须重新加回 `--api-key-file`。

实测（2026-10-20）：新节点自带标签、服务主机自动 Online，
Web 200、`/v1/models` 1.3 s、chat 1.66 s。

## 用完清理

```python
tailscale serve reset
tailscale down
pkill -f 'tailscaled --tun=userspace-networking'
```

`tailscale down` 只让当前临时节点下线，不是 logout；节点是 ephemeral，会自动从管理台回收，固定 Service 不受影响。
通常不需要手动做这些：释放运行时（`stop --all` 或停机按钮）就一并清掉了。

## 改完笔记本要刷新页面

用 Drive API 直接替换 `.ipynb` 内容后，打开着的 Colab 标签页仍是旧副本，必须
`reload`；刷新后右上角会变成「连接」，点一次即可接回同一个运行时（模型不会重启）。
刷新前先确认 `http://qwen.<tailnet>.ts.net:443/v1/models` 还能通，能通说明 VM 还活着。

---

# 附录：已废弃的“主机名方案”（仅供排查旧笔记本）

早期做法是让每个运行时自己登录成一个固定名字的节点，访问 `https://<节点名>.<tailnet>.ts.net/`。
根因问题：Tailscale 身份存在 `/content/tailscaled.state` + `tsstatedir`，Colab 每个运行时都清掉 `/content`，
新运行时会注册成**新节点**；旧离线节点还占着名字，新节点就只能拿到 `qwen-colab-1`、`-2`。
只还原 state 文件试过，缺 profile 数据会报
`cannot start backend when state store is unhealthy`，所以不要走“备份登录态”这条路。
现行方案用上面的 Service 取代它，下面几节只在你碰到旧笔记本或旧节点时有用。

## A. 首次登录与指向本地端口（旧）

```python
Popen([... , 'up', '--hostname=qwen-colab', '--accept-dns=false'],
      stdout=open('/content/ts-up.log', 'w'), stderr=STDOUT, start_new_session=True)
# 从 /content/ts-up.log 里取 https://login.tailscale.com/a/... 交给本人点一次
```

登录后 `BackendState` 应为 `Running`，`tailscale status` 里能看到本机节点名与 100.x 地址。

```python
tailscale serve --bg 8081        # https://<host>.<tailnet>.ts.net/ → http://127.0.0.1:8081
tailscale serve status
```

期望输出：

```
https://qwen-colab.<tailnet>.ts.net (tailnet only)
|-- / proxy http://127.0.0.1:8081
```

验证（旧）：

```powershell
Invoke-WebRequest -Uri 'https://qwen-colab.<tailnet>.ts.net/verify' -TimeoutSec 30
```

返回 200 且回显里有 `Tailscale-User-Login: <本人账号>`、`X-Forwarded-For: 100.x.x.x`，说明 Serve 链路与身份头都正常。

## B. 新运行时要重新登录（2026-10-20 实测，旧）

`/content` 是每个运行时独立的盘，`tailscaled.state` 和 `tsstatedir` 随运行时删除一起消失，
所以**只把 state 放在 `/content` 时，每次新建运行时都要重新登录一次**。管理台看到的旧节点
仍是 offline，浏览器登录一次即可（账号已记住，`Connect device` 一点就过）。

要做成免登录，得把状态持久化：

1. 启动前从 Drive 恢复 `tailscaled.state` + `tsstatedir`（打包成 `tailscale-state.tar.gz`）。
2. `serve` 设好后把它打回去；约 7–10 KB。

注意这段依赖 `google.colab.auth.authenticate_user()`，即 Colab 的 Drive 授权弹窗。

## C. 设备名后缀与证书（旧）

旧节点没删时，新节点会拿到 `qwen-colab-1`、`-2` 这类后缀，固定域名跟着变。处理顺序：

1. 管理台把 offline 的旧节点改名（`qwen-colab-old`、`-old2`），腾出名字。
2. 把当前节点改成 `qwen-colab`，`tailscale status --json` 的 `Self.DNSName` 会立刻变。
3. **重跑 `serve --bg --https=443 ...` 并执行 `tailscale cert qwen-colab.<tailnet>.ts.net`**。

第 3 步不能省：改了名但证书还是旧域名的，本机访问会直接超时，看着像网络不通。
`tailscale cert` 只接受域名参数，1.104.1 不支持 `--cert-file/--key-file`。

## D. Colab Drive 授权弹窗点不动的退路（旧，与现行笔记本无关）

2026-10-20 遇到：新运行时里 `authenticate_user()` 弹出 `您正在重新登录` 页面，
该 popup 窗口在浏览器自动化下 `Input.dispatchMouseEvent` 与截图都超时，只能导航不能点击，
于是单元格一直卡在 `read_reply_from_input`。此时不要反复重跑单元格。
现行笔记本不依赖 Drive 授权，直接从 Hugging Face 固定 revision 下权重；这一节只记录当时绕开 Drive 的做法：

- llama.cpp 用同版本 release 资产重装，`llama-server` 路径保持 `llama-dist/llama-b11560/llama-server`：

```bash
cd /content/qwen27b-pi/llama-dist
curl -fL -O https://github.com/ggml-org/llama.cpp/releases/download/b11560/llama-b11560-bin-ubuntu-cuda-12.8-x64.tar.gz
curl -fL -O https://github.com/ggml-org/llama.cpp/releases/download/b11560/cudart-llama-b11560-bin-ubuntu-cuda-12.8-x64.tar.gz
tar -xzf llama-b11560-bin-ubuntu-cuda-12.8-x64.tar.gz && tar -xzf cudart-llama-b11560-bin-ubuntu-cuda-12.8-x64.tar.gz
```

- 两个 GGUF 从 Hugging Face 按 revision 直接下，公开仓库无需 token：

```bash
REV=97caacb971d9246520fa8db5282a06bd36b103df
curl -fsSL -O https://huggingface.co/bytkim/Qwen3.8-27B-pi-GGUF/resolve/$REV/Qwen3.8-27B-pi-Q4_K_M.gguf
curl -fsSL -O https://huggingface.co/bytkim/Qwen3.8-27B-pi-GGUF/resolve/$REV/mtp/mtp-Qwen3.8-27B-pi-Q4_0.gguf
```

实测这四件合计约 19 GB，A100 运行时里从开始下载到 `/health` 200 约 6 分钟，
比 Drive 恢复（71 秒读取）慢，但完全不依赖授权，适合授权坏掉时的应急。
