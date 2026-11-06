---
name: colab-gpu-llm-service
description: 在 Google Colab 付费 GPU 上持久化并运行 GGUF 大模型，配置 llama.cpp、MTP 投机解码和推理档位，接入 pi 或 OpenAI 兼容客户端，并用固定域名与鉴权安全暴露。用于 Colab A100 启动、模型调参、冷启动异常、CU 预算、Drive 持久化、隧道地址每次变化、无鉴权公网入口、pi provider 或 llama-server 排障时。
---

# Colab GPU LLM Service

把一个有限时、可丢失的 Colab 运行时做成可重复启动的模型服务。模型和运行环境放 Drive；GPU 只承载当前会话；公网入口必须有应用层鉴权。

## 1. 先定边界

确认模型文件、量化、GPU 型号、上下文、使用时长和是否要公网访问。状态类信息按当天实际页面或命令回答；Colab 套餐、CU 单价和可用 GPU 会变化。

完成条件：写下目标模型 ID、启动参数、预算上限和预期使用场景。

## 2. 持久化模型和运行环境

首次迁移单独做，并优先使用 CPU 运行时：

1. 固定 Hugging Face repo、revision、文件名、字节数和 SHA-256。
2. 下载后用大小和哈希验收，再上传到 Drive。
3. 同时保存 `llama-server`、CUDA 运行库和 manifest。
4. 日常启动从 Drive 并发读取，按文件大小跳过已有副本；完整哈希只在首次迁移和怀疑损坏时做。

完成条件：Drive 中存在主权重、配套 draft/MTP 文件、`llama-server`、CUDA 库和 manifest；每项的大小与记录一致。

## 3. 启动与预热

### 官方 Colab CLI：只适合 20 分钟内的短任务、查账与释放

**长时间用模型仍走浏览器笔记本。** CLI 创建的 G4 会话在创建后约 24 分钟被 Colab 回收（2026-11-06 用 CLI 0.7.5 实测三次，三种保活都无效：每 10 分钟推理请求、内核持续执行阻塞单元、每 60 秒 `GET /tun/m/<endpoint>/keep-alive/`（`X-Colab-Tunnel: Google`，22 次 0 错误））。CPU 运行时闲 28 分钟仍在，不能拿 CPU 推 G4。CLI 升级后先用 `start` 测一次能否活过 30 分钟，过了再把它改回日常入口。

统一内核登记的 4 个 G4 模型（`qwen38-fn`、`qwen35-122b`、`nemotron-120b`、`gemma4-31b`）用 [`scripts/colab_cli.py`](scripts/colab_cli.py) 一条命令起停，背后是 Google 官方 `google-colab-cli`（`uv tool install google-colab-cli`，`colab.exe`）：

```bash
python scripts/colab_cli.py start qwen38-fn   # 开 G4 → 跑首格同一份代码 → Tailscale 验 /v1/models → 全链路验收
python scripts/colab_cli.py status            # 余额 / 燃烧率 / 会话 / 空闲守护状态
python scripts/colab_cli.py stop --all        # 释放并核验 0 active / 0 CU/h
```

- 它复用 `build_notebook_dict` 生成的首格代码，只切掉需要前端的尾巴（横幅、弹 WebUI、停机按钮）。
- `start` 会用 CLI 自己的解释器后台起 [`scripts/colab_keepalive.py`](scripts/colab_keepalive.py)：每 60 秒 ping 一次、经 Tailscale 查 `/slots`，连续 `--idle-min`（默认 45）分钟无推理就 `colab stop`；状态与日志在 `~/.config/colab-cli/keepalive-<slug>.json|.log`。电脑睡眠或关机时 ping 停了，Colab 20 多分钟后自己回收，忘关也烧不了多久。
- `status` 显示「NO keep-alive running」就是守护进程不在：再跑一次 `start <slug>`（已有会话会复用，首格对健康服务是空操作）或直接 `stop`。
- 首次要授权一次：跑 `colab sessions`，打开它打印的 Google 链接点「允许」，把授权码贴回终端；token 存 `~/.config/colab-cli/token.json`，之后不再弹窗。
- CLI 会话里 **Colab Secrets（`userdata`）和 `drive.mount()` 都不可用**：内核从 Hugging Face 固定 revision 下载、凭据在生成时写入，不依赖二者；本机有 `HF_TOKEN` 环境变量或 `~/.cache/huggingface/token` 时会自动带上。
- 浏览器同步（`--sync-colab --run-cell1`）保留为备用入口。

实测（2026-11-06，G4）：开机到 `/v1/models` 可用 130 s；`exec` 结束后 llama-server 等后台进程照常存活，内核变量跨次保留。

启动顺序固定为：恢复文件 → 启动 `llama-server` → 等 `/health` → 调 `/v1/models` 确认别名 → 发两个短请求预热。

冷启动会触发权重首次触页、CUDA/cuBLAS 初始化和图构建。第一次请求明显变慢时，先做连续请求对比，再判断真实吞吐。把冷启动开销留在启动单元里，用户的第一次提问才稳定。

常用结构：

```python
server = subprocess.Popen(command, stdout=log, stderr=log, env=env)
for _ in range(180):
    if requests.get("http://127.0.0.1:8081/health", timeout=2).ok:
        break
    time.sleep(2)
else:
    raise RuntimeError("llama-server did not become healthy")
```

统一内核（`scripts/colab_kernel.py`）在预热后还做三件事：

- **拼写探针**：新启动时考 12 道生僻词和 DOM/SVG 标识符题，结果写进 `<root>/spell-probe.json`。查找表（如 PLE）或量化出问题时，跑分和推理可能看不出来，最先坏的是拼写，所以错 2 题以上要先排查权重文件和分片摆放，再拿去写长篇。
- **内存回收**：SHA-256 通过后，删掉 `/dev/shm` 里只给显存用的分片副本（运行期由 CPU 读取的 PLE 分片保留）。`--cache-ram` 设为 16 GiB 加上可回收内存的一半，最多 64 GiB。删掉了哪些文件、对应哪个服务 PID 记在 `released-copies.json`，服务不重启时重跑首格不会重新下载；服务挂了或配置变了，就会重新下载这几片。
- **热切换**：「热切换模型」格不释放 G4：先停掉当前服务、清掉其他模型的文件（`llama-dist` 留着复用），再执行目标模型的首格。Tailscale 与 `svc:qwen` 地址不变，客户端只需换模型名。覆盖内核登记的全部 4 个 G4 模型。

完成条件：`/health` 返回 200，`/v1/models` 返回预期模型 ID，预热请求成功，拼写探针至少 11/12。

## 4. 调参与验收

先按模型卡和模板约束选参数，再测。不要只凭一次生成判断速度或 MTP 是否生效。

- 上下文、KV cache 类型和 `--parallel` 一起决定显存余量。
- MTP 需要匹配的 draft 文件；检查启动日志中的 draft model、acceptance 和 mean accepted length。
- 推理档位由聊天模板决定。模板只接受某些值时，在客户端做映射。
- 用同一提示至少做一次冷请求和两次连续请求，记录 prompt/output tokens、耗时、`predicted_per_second`。
- 工具型客户端另做一次真实工具调用；文本聊天成功不代表工具调用兼容。

具体模型的已验证参数、文件哈希、笔记本位置与启动细节（统一由 [`scripts/colab_kernel.py`](scripts/colab_kernel.py) 内核维护与一键分发：`python scripts/colab_kernel.py --all --sync-colab --sync-magpie`，WebUI 本机小说目录读写桥为 [`scripts/local_novel_mcp.py`](scripts/local_novel_mcp.py)）：
- G4 96GB（综合 1 · 125B 小说共创调优版：`Swift-1.5-Qwen3.8-Flash-Next-IQ4_XS`）：见 [`references/qwen38-flash-next-g4.md`](references/qwen38-flash-next-g4.md)，笔记本在 [`notebooks/qwen3.8-flash-next-g4-colab.ipynb`](notebooks/qwen3.8-flash-next-g4-colab.ipynb)（薄入口 [`scripts/make_g4_fn_notebook.py`](scripts/make_g4_fn_notebook.py)）。
- G4 96GB（笔记本 2 · 122B Kimi+Opus 蒸馏破限版：`Qwopus3.5-122B-A10B-Kimi-Q4_K_M`）：见 [`references/qwen35-122b-g4.md`](references/qwen35-122b-g4.md)，笔记本在 [`notebooks/qwen3.5-122b-g4-colab.ipynb`](notebooks/qwen3.5-122b-g4-colab.ipynb)（生成脚本 [`scripts/make_122b_qwen35_notebook.py`](scripts/make_122b_qwen35_notebook.py)）。
- G4 96GB（笔记本 3 · 120B 异构混合破限版：`Nemotron-3.5-Super-VL-120B-Heretic`）：见 [`references/nemotron-120b-g4.md`](references/nemotron-120b-g4.md)，笔记本在 [`notebooks/nemotron-120b-g4-colab.ipynb`](notebooks/nemotron-120b-g4-colab.ipynb)（生成脚本 [`scripts/make_120b_nemotron_notebook.py`](scripts/make_120b_nemotron_notebook.py)）。
- G4 96GB / A100 40GB（笔记本 4 · 31B 人感破限版：`Gemma-4-31B-Ortenzya-Heretic-Q8_0`）：见 [`references/gemma4-31b-g4.md`](references/gemma4-31b-g4.md)，笔记本在 [`notebooks/gemma4-31b-g4-colab.ipynb`](notebooks/gemma4-31b-g4-colab.ipynb)（生成脚本 [`scripts/make_31b_gemma4_notebook.py`](scripts/make_31b_gemma4_notebook.py)）。

完成条件：模型 ID、思考字段、至少两次正常生成和一次工具调用均有可复核结果。

## 5. 选择公网入口

先选威胁模型：

| 需求 | 入口 | 必须补上的保护 |
|---|---|---|
| 只在 Colab 内用 | 无隧道 | 无 |
| 临时给本机或朋友用 | Quick Tunnel | `llama-server --api-key-file`；地址短期有效 |
| 固定地址给网页和 API | Cloudflare Named Tunnel + 自有域名 | Web 走 Access；API 走独立 hostname + API key |
| 只有自己的设备要访问 | Tailscale Service + Serve（固定 `svc.<tailnet>.ts.net:443`，仅 tailnet 内可见） | 依赖 WireGuard 私网身份隔离（免 `--api-key-file` 以便 WebUI 免密即开即用）；若上 Funnel 公网则必加 `--api-key-file` |
| 不想买域名但要公网 | Tailscale Funnel 或服务商保留域名 | 应用层 API key |

固定域名本身不提供安全。每次变化的随机 Quick Tunnel 地址只是隐蔽，不等于鉴权。任何可写 API 的公网入口都先加 `llama-server` API key。

推荐的 Cloudflare 结构：

- `qwen.example.com` → Web 聊天界面；用 Cloudflare Access 限制到本人邮箱。
- `qwen-api.example.com` → OpenAI 兼容 API；不套浏览器登录，由 `--api-key-file` 校验 Bearer key。
- 两个 hostname 都路由到 Colab 内的 `http://127.0.0.1:8081`。
- Tunnel token 和模型 API key 放 Colab Secrets，勾选 notebook access；写入临时文件时设 `0600`。

完整命令、Secret 名称、Dashboard 配置、轮换方法和 `pi` 配置见
[`references/secure-remote-service.md`](references/secure-remote-service.md)。

完成条件：未带 key 的 API 请求被拒绝；带 key 的 `/v1/models` 和聊天请求成功；Web 只有授权账号可进；固定地址在运行时重启后仍不变。

只用自己设备时，优先 Tailscale：不暴露公网、请求自带 `Tailscale-User-Login` 身份头。
Colab 每换一次运行时 `/content` 就清空、节点改名成 `qwen-colab-1/2`，所以对外地址要用
**Tailscale Service**（`svc:qwen` → `qwen.<tailnet>.ts.net`）而不是节点域名；新运行时用
OAuth 客户端现场签发带 `tag:colab` 的一次性 `ephemeral` auth key 加入，ACL 的
`autoApprovers.services` 自动审批，全程不需要人工登录或点 Approve。安装、`serve --bg`、
两个必踩的坑（中断执行杀 daemon、缺 `--statedir` 导致 HTTPS `InternalError`）、ACL 片段和
验证方法见 [`references/tailscale-serve.md`](references/tailscale-serve.md)。

## 6. 接入 pi

本机已有 Magpie 时，优先在 Magpie 原生注册 custom provider，上游填 Tailscale Service 的完整 HTTPS `/v1` 地址与模型 key；pi 复用已有本机网关 `http://127.0.0.1:3425/v1`。用 `magpie model context/output/efforts` 管理模型声明并下发，不同时手改 Magpie 生成的 pi 条目。本机已在 Magpie `qwen-colab` 下注册内核登记的 4 个模型（由 `colab_kernel.py --sync-magpie` 按 `MODELS` 下发），分时复用 `http://qwen.<YOUR_TAILNET>.ts.net:443/v1`，pi 调用名加 `magpie/` 前缀；具体命令及边界见各模型参考文档。Magpie 负责模型请求和 token 用量，Colab 仍负责运行时与 CU；不上路由组即可按需手选，离线时不自动换供应商。

没有 Magpie、需要独立直连时才使用下面的脚本。

把 base URL 指向固定的 `https://qwen-api.example.com/v1`，把同一 API key 写入 provider。模型 ID 必须与服务端 alias 完全一致。

```powershell
pwsh -NoProfile -File scripts/update-pi-provider.ps1 `
  -BaseUrl https://qwen-api.example.com `
  -ApiKey '<same-secret>' `
  -ServiceName qwen38-colab `
  -ModelId Qwen3.8-27B-pi-Q4_K_M
```

脚本先备份 `~/.pi/agent/models.json`，只替换目标 provider，不改其他 provider 或默认模型。之后用：

```powershell
pi --model qwen38-colab/Qwen3.8-27B-pi-Q4_K_M
```

完成条件：`pi` 能列出一个非流式回答，并在需要时成功调用一个本地工具。

## 7. 用完释放

Colab 按运行时被分配的时间计费，空闲也扣 CU。CLI 启动的用 `python scripts/colab_cli.py stop --all`（或 `colab stop -s <slug>`），它会核验 `0 active / 0 CU/h`；CLI 叫不出名字的会话（`colab sessions` 里显示 `[?]`）自动转交 `colab_sync.py --release-all` 强杀。笔记本启动的在笔记本里终止服务后调用 `google.colab.runtime.unassign()`，再用 `python scripts/colab_meter.py stop --all` 兜底。

完成条件：页面右上角显示“重新连接”；`代码执行程序 → 管理会话` 中没有正在进行的会话。旧输出保留不算运行凭据。

### 零 OAuth 实时查账、双轨对账与幽灵实例强杀

Colab 网页自带的 `/tun/m` 接口（`ccu-info`、`assignments`、`unassign/<endpoint>`）在已登录的 `colab.research.google.com` 标签页里直接可用，不需要任何 OAuth 或 API key。脚本按 BrowserMCP（`live-browser` 技能的 `browsermcp-bridge.mjs`）→ Playwriter → `playwright-cli -s=edge` 的顺序降级，在页面上下文里调用：已有任意 Colab 标签页就原样复用（不刷新、不跳转）；一个都没有就后台开临时页，`evaluate` 后立刻关闭，不留标签页。

```bash
python scripts/colab_meter.py status [--json]      # Active assignments: N | Burn rate: X.XX CU/h | Balance: YYY.YY CU | Eligible GPUs: ...
python scripts/colab_meter.py start -s lit -g G4   # 记录开机余额与时间（.colab-meter-state.json）
python scripts/colab_meter.py stop  -s lit         # 强杀 + 二次核验 + 双轨对账单；--all 清场全部
python scripts/colab_sync.py --status              # 同 status，不依赖 meter 状态文件
python scripts/colab_sync.py --release-all         # 强杀全部活跃 assignment 并核验
```

- **强杀**：对每个 `endpoint` 先 `GET /tun/m/unassign/<endpoint>` 取 `token`，再 `POST` 同一地址并带 `X-Goog-Colab-Token`。释放后等 1.5 s 重查，最多 3 次，直到 `assignmentsCount == 0` 且 `consumptionRateHourly == 0`。这条路径能清掉笔记本页面已关闭、`runtime.unassign()` 没机会执行的**幽灵会话**，它们仍按小时扣 CU。
- **双轨对账**：轨道 1 为云端账单实扣差值 `max(0, start_balance - end_balance)`；轨道 2 为墙钟时长折算 `dur_sec / 3600 * rate_hourly`。云端余额可能滞后几分钟，两条并列比单看一条可靠。
- **与同步脚本联动**：`make_*_notebook.py --sync-colab --run-cell1` 成功后自动打点起始余额；`--release-runtime` 除了 `nb.kernel.unassignCurrentVm`，还会清理残留 assignment、核验 `0 active / 0.00 CU/h` 并打印对账单。

完成条件：`status` 输出 `Active assignments: 0 | Burn rate: 0.00 CU/h`。这比页面“重新连接”更可靠，因为它直接读云端账本。

## 凭据与开源脱敏

真实配置放仓库根目录的 `.secrets.json`（已被 `.gitignore` 忽略，字段见 `.secrets.example.json`），也可用环境变量 `LLAMA_API_KEY`、`TS_CLIENT_ID`、`TS_CLIENT_SECRET`、`TAILNET_FQDN` 与各笔记本的 `*_DRIVE_ID`（字段见 `scripts/secrets_config.py`）提供。`make_*_notebook.py` 默认读取它们生成自用笔记本；`--sanitized` 一律填占位符，`--output-dir <dir>` 改输出目录。`--sanitized` 与 `--sync-colab` 互斥，避免把占位符推进 Colab。

## 快速排障

- `colab new --gpu G4` 报 "Backend rejected accelerator"：G4 只有一种 176 GiB 机型，带 `--high-mem` 会被拒；`colab_cli.py` 已对 G4/L4 不加该参数。
- `colab exec` 报 `Timeout waiting for reply`：它默认 30 秒超时，长任务必须带 `--timeout`（`colab_cli.py` 已带）。
- `colab sessions` 出现 `[?]`：本地状态里没有这个会话名（例如同名会话被重复创建），`colab stop` 停不掉，用 `colab_cli.py stop --all` 或 `colab_sync.py --release-all`。
- 首格下载报 404：先跑 HF 核对（逐个 HEAD `resolve/<rev>/<path>`，比对 `X-Linked-Size` 与 `X-Linked-ETag`=sha256），registry 里的 rev、路径或哈希与参考文档不一致时以 HF 实际值为准改 `MODELS`。
- 首条请求只有约 5 tok/s：先连续发两条；若恢复到几十 tok/s，属于冷启动，已由预热解决。
- 首格报「需要重新启动服务，但缺少上次为扩大提示缓存而删掉的分片副本」：配置变了或旧服务不健康，旧服务已经停掉。再运行一次首格，会重新下载被删的那几片。只改缓存上限不会触发重启，新上限等下次启动才生效。
- 内存回收提示「实际回收不到预计的一半」：说明这些分片仍被 mmap 映射着（mmap 模式下 CPU 侧张量所在的分片），要等服务重启才会真正归还；这时提示缓存别用满。
- `pi` 返回模板错误：检查 `thinkingLevelMap`，把客户端不支持的档位映射到服务端接受的档位。
- Cloudflare 地址每次变化：当前用的是 Quick Tunnel；换成 Named Tunnel，不要只做本地地址替换。
- 固定域名能打开但不能聊天：确认 API hostname 没有误套 Cloudflare Access，并确认 `pi` 的 `apiKey` 与服务端一致。
- 服务健康但工具调用失败：测试 `/v1/chat/completions` 的 tool calling，检查 `--jinja` 和模型聊天模板。
- 新运行时上“日常启动 1/2”一直没输出、`/content` 里没有模型目录：它卡在 Google Drive 授权。
  Colab 会弹一个 `accounts.google.com` 的“重新登录 … 继续”窗口，浏览器自动化点不动（CDP 派发鼠标事件超时），
  必须本人点一次“继续”。长期解法是把 `auth.authenticate_user()` 之外的重活改成直接从 Hugging Face 拉取，
  或先把 Drive 授权跑通再切 GPU，避免在 A100 上等授权烧 CU。
- 切换运行时类型后，`/content` 清空、tailscaled 也一起消失：Tailscale Serve 单元要在新运行时里重跑一次，
  管理台里该节点显示 offline 属正常，不是运行时已死。
