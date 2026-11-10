# colab-gpu-llm-service

中文 | [English](README.en.md)

在 Google Colab 租用的 G4 GPU（RTX PRO 6000 Blackwell，96 GB 显存）上跑百 GB 级 GGUF 大模型：冷启动快，无头保活，闲置自动释放，经 Tailscale 私网固定地址访问。

整套服务可复现、私有、花费可控。单一统一内核（`scripts/colab_kernel.py`）同时驱动**两条并存轨道**：

- **主力轨道（无头 Colab CLI + 30 秒内核心跳）** — `colab_cli.py` + `colab_keepalive.py` 无需打开浏览器标签页即可分配 G4 并运行内核启动代码，通过每 30 秒一次的 `colab exec` 内核心跳（移植自 Wakeloom 本地模式）跨过 Colab CLI 的 21–35 分钟闲置回收窗口，并在连续闲置 45 分钟后自动释放实例。
- **后备轨道（浏览器 Notebook 同步 + 零 OAuth `/tun/m` 算力计费与强杀开关）** — `make_*_notebook.py --sync-colab --run-cell1`、`colab_sync.py` 与 `colab_meter.py` 在已登录的 Edge 标签页内（`BrowserMCP → Playwriter → playwright-cli`）驱动同一份 Notebook，提供交互式 WebUI / LoRA / 模型热切换控制台、运行归档的多模型 Notebook、核对双轨 CU 账单，并在 `colab_cli.py stop --all` 移交时强制清理无名 `[?]` 幽灵会话。

Colab 运行时是一次性的，每次新会话都会清空 `/content`，因此下文的全部设计都围绕两件事展开：**快速、确定性的冷启动**，以及**绝不为没在使用的 GPU 付一分钱**。

## 核心组件一览

| 组件 | 解决的问题 |
|---|---|
| **主力轨道：** 无头 Colab CLI + 30 秒内核心跳（`colab_cli.py`、`colab_keepalive.py`） | 在终端直接启停 G4，无需打开浏览器标签页；30 秒 `colab exec` 心跳防止 CLI 0.7.5 的 21–35 分钟闲置回收，配合 45 分钟 `/slots` 无推理自动关停 |
| 生产级极速启动内核与加载后 `tmpfs` 内存回收（`colab_kernel.py`） | `DEBUG_VERIFY = False` 跳过启动探针，每 0.5 秒轮询 `/health`，后台异步预热，并在加载完成后释放仅驻留 GPU 的 `/dev/shm` 分片副本，将 `--cache-ram` 扩大至 16–64 GiB |
| DDR5 `tmpfs` 内存盘 + 32 路并发直写 | Colab 持久盘写入上限约 200–350 MiB/s；写一个 95 GB 的模型会平白多花好几分钟 |
| PLE 嵌入表 `MAP_POPULATE` 零拷贝映射 | 消除推理期间的二次内存拷贝与缺页中断（page fault） |
| `llama.cpp` MTP 投机解码 | 针对代码、工具调用与长文写作调优的 MTP 草稿头投机解码 |
| Tailscale Service（`svc:qwen`）+ 可选本机 MCP 文件桥（`local_novel_mcp.py`） | 跨运行时固定不变的 `qwen.<YOUR_TAILNET>.ts.net` 入口地址，搭配反向 SOCKS5 桥让 WebUI 无需弹窗确认即可读写本地目录 |
| **后备轨道：** 浏览器标签页同步、零 OAuth CU 计费与幽灵会话强杀（`colab_sync.py`、`colab_meter.py`） | 在已登录浏览器标签页推送/运行 Notebook、免 OAuth 读取实时余额与燃烧率、打印双轨 CU 账单，并强杀关掉页面后残留的无名 `[?]` 会话 |
| 可选配套 [Colab-yoga](https://github.com/incredible404-rgb/Colab-yoga) 托盘工作台（`colab_cli.py yoga`） | 与本仓库共用同一台 G4；`colab_kernel.py` 把现役模型的无头首格导出为它的 `colab_kernel_runners.json` |
| `.secrets.json` + `--sanitized` | Notebook 由私有配置生成；脱敏构建（sanitized build）不含任何真实密钥、Tailnet 名称或 Google Drive 文件 ID |

## 架构全景

```mermaid
flowchart LR
    subgraph LOCAL["本地机器"]
        direction TB
        subgraph PRI["主力轨道：无头 CLI"]
            CLI["colab_cli.py<br/>start · launch · status · stop"]
            KA["colab_keepalive.py<br/>30s 内核心跳 · 60s TFE ping<br/>45 分钟 /slots 闲置自动关停"]
            CLI --> KA
        end
        subgraph FB["后备轨道：已登录 Edge（__ANY__ 标签页）"]
            NB["make_*_notebook.py<br/>--sync-colab --run-cell1"]
            SY["colab_sync.py<br/>BrowserMCP → Playwriter → PW"]
            ME["colab_meter.py<br/>双轨账单 · unassign 强杀"]
        end
        CLI -. "残留无名 [?] 会话时回退" .-> SY
    end

    subgraph VM["Colab 运行时 · G4"]
        K["统一内核 cell 1<br/>colab_kernel.py<br/>① GPU 检查 ≥ 85 GB<br/>② tmpfs 内存盘<br/>③ 32 路并发直写<br/>④ llama-server · MTP<br/>⑤ 回收 tmpfs → cache-ram<br/>⑥ Serve → svc:qwen<br/>⑦ 横幅 · 热切换（仅浏览器）"]
    end

    CLI -- "colab exec" --> K
    KA -- "心跳" --> K
    NB --> K
    SY --> K
    ME <-- "/tun/m/*" --> K
    K --> TS["qwen.‹YOUR_TAILNET›.ts.net<br/>仅 Tailnet 私网可见<br/>WebUI · /v1 · 本地文件桥"]
```

## 极速冷启动

1. **绕开持久盘写入瓶颈。** Notebook 先扩容 `/dev/shm` 与 `/var/colab`，再通过软链接映射回运行时目录。模型权重直接落入物理 DDR5 内存，热路径上零磁盘 I/O。
2. **32 路并发流下载。** 独立下载器启动 8 进程 × 4 线程，每个线程用 `os.pwrite` 按各自偏移量直接写入文件。多进程架构避开了单进程 Python GIL 与 TLS 解密瓶颈；签名 CDN 链接只解析一次并跨工作进程复用。
3. **最大张量采用 `MAP_POPULATE` 预填充。** 第二分片是单块 26.8 GiB 的逐层嵌入表（PLE table），优先写入 `/dev/shm` 并由 `llama-server` 以 `--lazy-mode off` 映射、预先填充页表，推理时既无缺页中断也无二次拷贝。
4. **单单元格控制台。** 启动完成后，第一个单元格直接展示就绪横幅、WebUI 入口链接以及一键释放 GPU 的停止按钮。

## 投机解码与采样参数

`llama.cpp` 以 `--spec-type draft-mtp` 配合对应的 MTP 草稿头运行。G4 Notebook 专为代码与长程 Agent 任务调优：单槽 262,144 token 上下文、严格采用模型卡推荐采样参数（不额外加惩罚项或 DRY）、16 GiB 提示词缓存（支持槽位持久化）以及 65,536 token 最大输出上限。具体吞吐可按自己的提示词实测：先发一次冷请求、再连发两次热请求，从服务端日志读取 `predicted_per_second` 与草稿接受率。

## Tailscale Service 私网固定入口

运行时使用由 OAuth 客户端动态签发的单次、临时（ephemeral）、带 `tag:colab` 标签的认证密钥加入你的 Tailnet，并发布 `svc:qwen` 服务。在 Tailscale ACL 中配置 `autoApprovers.services` 后，全程无需人工登录或点击批准，跨运行时的访问地址始终固定为 `qwen.<YOUR_TAILNET>.ts.net`。ACL 配置片段与两种常见故障模式（中断单元格杀掉守护进程、漏传 `--statedir` 导致 HTTPS 失效）见 [`references/tailscale-serve.md`](references/tailscale-serve.md)。

该服务仅在你的 Tailnet 私网内可见，因此打开 WebUI 无需粘贴 API Key。若你需要通过 Funnel 或任何公网入口暴露服务，请务必先加上 `--api-key-file`。

## 主力轨道：官方 Colab CLI 无头启动 + 30 秒内核心跳

自 CLI 0.7.5 起，由 CLI 创建的运行时（无论是 T4 还是 G4）若内核没有收到新的代码执行请求，会在创建后 21–35 分钟内被强制回收；单纯的 TFE ping、单个长跑单元格或正在进行的模型推理都不算数，且旧版 `RuntimeService/KeepAliveAssignment` RPC 会返回 `403 USER_PROJECT_DENIED`。每隔 30 秒执行一行轻量 `colab exec` 即可让 VM 持续存活（与 Wakeloom 本地模式同理）。2026-11-07 实测（见 `debug-runs/2026-11-07-*kernel-beat*`）：T4 对照组在 21.0 分钟被回收，而心跳组完整存活满 42 分钟；运行 `qwen38-fn` 的 G4 实例在 77 次心跳间持续对外服务满 39 分钟（仅在 Cell 1 启动期间出现 1 次预期的队列超时，启动完成后 0 失败），直至闲置守卫干净地将其关停。

`scripts/colab_cli.py` 封装并驱动官方 [`google-colab-cli`](https://github.com/googlecolab/google-colab-cli)（通过 `uv tool install "google-colab-cli==0.7.5"` 安装，首次使用运行 `colab sessions` 完成一次授权）。它执行由 `colab_kernel.py` 生成的完全相同的 Cell 1 代码（`colab_kernel.py --list` 列出的任一现役或归档模型），仅剥离了纯前端尾部（横幅、WebUI 弹窗、停止按钮），全程无需打开浏览器标签页。

CLI 0.7 移除了自带的保活守护进程。`start` 与 `launch` 会使用 CLI 自身的 Python 解释器在后台独立拉起 `scripts/colab_keepalive.py`：每 30 秒发送一次内核心跳，每 60 秒请求一次 `GET /tun/m/<endpoint>/keep-alive/`（带 `X-Colab-Tunnel: Google` 头），同时通过 Tailscale 轮询 `/slots`，在连续 `--idle-min`（默认 45）分钟无推理活动后自动执行 `colab stop`。若本地电脑休眠，心跳随之停止，Colab 会在约 20 分钟内自动回收 VM。

```bash
python scripts/colab_cli.py start  [<slug>]    # 分配 G4 -> 跑 cell 1 -> 检查 Tailscale /v1/models -> 跑验证 cell
python scripts/colab_cli.py launch [<slug>]    # 后台分离启动 cell 1 (LAUNCH_DETACHED=1) -> 轮询 /health
python scripts/colab_cli.py status                # 查看余额、燃烧率、活动会话、心跳与闲置守卫状态
python scripts/colab_cli.py stop   --all          # 释放全部实例；若残留无名会话则自动回退调用 colab_sync --release-all
```

- **模型代号（slug）：** 现役 `qwen38-huihui-v2`（默认）、`gemma4-31b`、`qwen38-27b-rvn`、`glm53-flash`；归档 `qwen38-fn`、`qwen38-fn-nl` 等可按 slug 直接启动。清单以 `python scripts/colab_kernel.py --list` 为准。
- **停止时自动回退：** 若 `colab sessions` 显示存在无名 `[?]` 分配（例如从浏览器标签页启动的会话或孤立的 CLI 会话），`colab_cli.py stop --all` 会自动调用后备轨道的 `colab_sync.py --release-all`，通过 `/tun/m/unassign` 强制释放并核验 `0 active / 0.00 CU/h`。
- CLI 会话无法读取 Colab Secrets 或挂载 Google Drive；内核直接从锁定的 Hugging Face commit 下载权重，并在本地存在 `HF_TOKEN` 时自动转发。此外 G4 不接受 `--high-mem` 参数，且 `colab exec` 默认 30 秒超时（需显式传 `--timeout`），脚本已内置处理这两项细节。

## 后备轨道：浏览器标签页同步、零 OAuth CU 计费与幽灵会话强杀

在以下场景下，可随时使用浏览器工作流作为后备：
1. 需要交互式 Cell 1 控制台（一键切换 LoRA、网页停止按钮）或 Cell 2 **不释放 G4 VM 的实例内模型热切换**；
2. 运行已归档冻结的多模型 Notebook（`qwen3.5-122b`、`nemotron-120b`，位于 `archive/multi-model-kernel/`）或实验性的 Strata Notebook；
3. CLI OAuth 令牌过期或未配置，希望直接在已登录的 Edge 标签页内（`BrowserMCP → Playwriter → playwright-cli`）零 OAuth 操作；
4. 需要打印双轨 CU 账单，或清扫 CLI 无法按名称寻址的 `[?]` 幽灵会话。

Colab 官方网页前端通过 `/tun/m/*` 接口通信。本轨道脚本直接在已登录的 `colab.research.google.com` 标签页上下文内调用这些接口，因此不需要额外的 OAuth 客户端或 API Key。

```bash
python scripts/colab_meter.py status
# Active assignments: 0 | Burn rate: 0.00 CU/h | Balance: 988.85 CU | Eligible GPUs: H100, G4, A100, L4, T4

python scripts/colab_meter.py start -s g4 -g G4     # 记录初始算力余额与启动时间
python scripts/colab_meter.py stop  -s g4           # 强杀实例、核验释放结果并打印双轨账单
python scripts/colab_meter.py stop  --all           # 清空所有活动分配

python scripts/colab_sync.py --status               # 同样的实时状态读取，无需本地计费状态文件
python scripts/colab_sync.py --release-all          # 强杀全部活动实例并核验
```

**标签页免打扰。** 使用虚拟 Drive ID `__ANY__` 时，若浏览器里已有打开的 Colab 标签页，脚本直接原位借用（绝不刷新或跳转页面）；若当前没有打开的 Colab 标签页，则在后台静默新建一个临时标签页，调用完成后立即关闭。

**强杀握手（Kill handshake）。** 对每个活动 `endpoint`：先请求 `GET /tun/m/unassign/<endpoint>` 获取 `token`，再带 `X-Goog-Colab-Token: <token>` 头向同一 URL 发送 `POST` 完成释放。等待 1.5 秒后重新查询 `ccu-info` 与 `assignments`（最多重试 3 次），直到确认 `assignmentsCount == 0` 且 `consumptionRateHourly == 0`。这能彻底清掉还没来得及执行 `runtime.unassign()` 就被关掉页面的幽灵会话。

**双轨账单（Two-track bill）。**

```text
轨道 1（账户总账差额）  max(0, start_balance - end_balance)
轨道 2（挂钟时长核算）  duration_seconds / 3600 x rate_hourly
```

轨道 1 是账户级余额差值：比实际消耗滞后几分钟，包含该时间窗内运行过的所有实例，遇到充值或退费甚至可能上升（此时脚本会截断为 0 并打印说明）。轨道 2 按单会话精确计时计费。同时展示两轨能让未释放的卡死会话一目了然；多实例并发时以轨道 2 作为单会话消耗为准。

**安全停止。** `colab_meter.py stop -s NAME` 只释放该会话名下记录的 endpoint；未知会话名或尚未绑定 endpoint 的会话会以退出码 1 安全退出、不误杀任何实例（需全清时显式用 `--all`）。若 `--run-cell1` 记录时尚处于 VM 分配中（费率为 0、无 endpoint），下一次运行 `status` 或 `stop` 时会自动回填真实 endpoint 与燃烧率。

**同步集成。** `make_*_notebook.py --sync-colab --run-cell1` 会在 Cell 1 启动后自动记录初始余额；`--release-runtime` 会依次调用 `unassignCurrentVm`、清扫残留分配、核验 `0 active / 0.00 CU/h` 并打印账单。

`/tun/m` 属于未公开内部接口，未来可能变动；日常作为便捷工具使用即可，若数据有出入以 Colab 网页的“管理会话（Manage sessions）”为准。

## 快速上手

### 前置条件

| 项 | 要求 |
|---|---|
| Colab 付费额度 | Colab Pro / Pro+ / Ultra 订阅或按量购买的计算单元（CU），账号需能选 G4；免费档不行。费率与额度会变，估算见 [`references/cu-budget.md`](references/cu-budget.md) |
| 主机 | Windows 10/11、Python 3.10+、Git；`pip install requests`（其余依赖均为标准库）。脚本用计划任务与 `taskkill` 清理保活进程，macOS/Linux 未验证 |
| 官方 Colab CLI | `uv tool install "google-colab-cli==0.7.5"`，随后运行一次 `colab sessions` 完成 OAuth 授权。保活机制按 0.7.5 的回收行为设计，请固定此版本 |
| Tailscale | 本机客户端 ≥ 1.78 并已登录；tailnet 里建好 `svc:qwen` Service、`tag:colab` 标签与 OAuth 客户端，步骤见 [`references/tailscale-serve.md`](references/tailscale-serve.md) |
| 可选：Colab-yoga | Windows 托盘多模态工作台（[Colab-yoga](https://github.com/incredible404-rgb/Colab-yoga)），装到 `%LOCALAPPDATA%\Programs\Colab-yoga` 后与本仓库共用同一台 G4；不装不影响任何核心命令 |
| 可选：后备轨道 | Node.js，以及已登录 Google 账号并安装 BrowserMCP 扩展的 Edge（Playwriter 或 `@playwright/cli` 作后备） |

### 从克隆到第一次开机

```bash
git clone https://github.com/incredible404-rgb/colab-gpu-llm-service.git
cd colab-gpu-llm-service
pip install requests
cp .secrets.example.json .secrets.json          # 按下表填写；该文件已在 .gitignore 中忽略

python scripts/colab_kernel.py --list           # 现役/归档模型 slug；start 不带 slug 时用现役表第一个
python scripts/colab_cli.py status              # 只读自检：CLI 已授权、读得到 CU 余额，不分配 GPU
python scripts/colab_cli.py start               # 分配 G4 -> 跑 cell 1 -> 检查 Tailscale /v1/models -> 跑验证 cell
python scripts/colab_cli.py stop --all          # 用完立即释放，核验末行 0 active / 0.00 CU/h
```

`start` 成功后会输出 `[colab_cli] start OK in <秒>s`；之后 `http://qwen.<YOUR_TAILNET>.ts.net:443/v1` 是 OpenAI 兼容端点（Bearer 密钥为 `.secrets.json` 的 `api_key`），根路径是 WebUI，二者仅 tailnet 内可见。

### `.secrets.json` 字段说明

前四项是 `start` 的必填项，缺一项会直接报 `missing ... in .secrets.json / environment`。

| 字段 | 含义 |
|---|---|
| `api_key` | 你自定义的任意长随机串，写入运行时的 llama-server 作 Bearer 密钥，也是你访问 `/v1` 的密钥 |
| `ts_client_id`、`ts_client_secret` | Tailscale OAuth 客户端凭据（仅 Auth Keys: Write，标签 `tag:colab`），运行时用它现签一次性临时认证密钥 |
| `tailnet_fqdn` | 服务访问域名，形如 `qwen.<YOUR_TAILNET>.ts.net`（`qwen` 是你建的 Service 名） |
| `local_tailscale_ip` | 可选。运行 `local_novel_mcp.py`（WebUI 本地文件桥）那台主机的 Tailscale IP |
| `*_drive_id`（每个现役模型一项） | 可选。该模型 Notebook 在你 Google Drive 里的文件 ID，仅 `--sync-colab` 需要；`colab_kernel.py` 生成时会从本机 Drive 客户端数据库自动解析并回填 |

每个字段也可通过环境变量传入：`LLAMA_API_KEY`、`TS_CLIENT_ID`、`TS_CLIENT_SECRET`、`TAILNET_FQDN`、`LOCAL_TAILSCALE_IP`，以及 `G4HUIHUI_DRIVE_ID`、`GEMMA31B_DRIVE_ID`、`RVN27B_DRIVE_ID`、`GLM53_DRIVE_ID`。读取优先级依次为：`.secrets.json` -> 环境变量 -> 占位符。其他可调环境变量：

| 环境变量 | 作用 |
|---|---|
| `COLAB_YOGA_DIR` | Colab-yoga 安装目录（默认 `%LOCALAPPDATA%\Programs\Colab-yoga`） |
| `COLAB_NOTEBOOKS_DIR` | 生成 Notebook 后镜像到的 Drive “Colab Notebooks”目录；不设则自动探测各盘符下的 `My Drive`，探测不到就不镜像 |
| `DRIVEFS_DB` | 指定 Drive 客户端元数据库路径 |
| `NOVEL_ROOT` | `local_novel_mcp.py` 向 WebUI 暴露的本机目录 |
| `HF_TOKEN` | 转发到运行时的 Hugging Face 令牌（受限或限速模型需要） |

### 常用命令速查

```bash
python scripts/colab_cli.py start  [<slug>] [--gpu G4] [--idle-min 45] [--no-verify] [--keep-on-fail]
python scripts/colab_cli.py launch [<slug>]       # 后台分离启动（读取 notebooks/ 里的私有构建，需先 colab_kernel.py <slug> 生成）
python scripts/colab_cli.py verify [<slug>]       # 只重跑验证 cell
python scripts/colab_cli.py status                # 余额、燃烧率、活动会话、心跳与闲置守卫
python scripts/colab_cli.py stop [<slug> | --all] # 释放；残留无名 [?] 会话时自动回退 colab_sync --release-all
python scripts/colab_cli.py yoga [gui | wake | status | stop]   # 可选：Colab-yoga 协同（旧名 easycolab 仍可用）

python scripts/colab_kernel.py --list             # 现役与归档模型
python scripts/colab_kernel.py --probe-hf <repo> [rev]   # 新模型 HF 预检
python scripts/colab_meter.py status              # 后备轨道：云端账本 + 当前分配
python scripts/colab_meter.py stop --all          # 后备轨道：强杀并打印双轨账单

cd scripts && python -X utf8 -m unittest discover -p "test_*.py"   # 全部单元测试
```

后备轨道（浏览器同步）：`python scripts/make_g4_huihui_v2_notebook.py --sync-colab --run-cell1`，再 `python scripts/colab_meter.py stop --all` 释放并结算。

### 可选：Colab-yoga 托盘工作台

[Colab-yoga](https://github.com/incredible404-rgb/Colab-yoga) 是配套的 Windows 托盘管理台，安装目录为 `%LOCALAPPDATA%\Programs\Colab-yoga`（`Colab-yoga.exe`、`runtime\python.exe`、`bridge\colab_yoga_bridge.py`）。`colab_kernel.py` 生成 Notebook 时会把各现役模型的无头首格导出到其目录下的 `colab_kernel_runners.json`（目录不存在则跳过并提示）；`colab_cli.py yoga ...` 找不到安装时会报出安装位置，也可用 `COLAB_YOGA_DIR` 指向自定义目录。先 `colab_cli.py start` 再在 Colab-yoga 里唤醒，或反过来，都会经 `~/.config/colab-cli/sessions.json` 复用同一台 G4。

### 故障排查

| 现象 | 处理 |
|---|---|
| `missing ... in .secrets.json / environment` | 必填四项仍是占位符，回到上表填写 |
| `colab` 找不到，或要求授权 | `uv tool install "google-colab-cli==0.7.5"`，再运行一次 `colab sessions` |
| `Colab-yoga not found at ...` | 可选组件未安装：按提示装 Colab-yoga，或设 `COLAB_YOGA_DIR`；不影响核心命令 |
| `Backend rejected accelerator` | G4 只有单一规格，`--high-mem` 会被拒（脚本已跳过）；也可能是账号无 G4 额度或当日无空闲卡 |
| 实例 20～35 分钟后消失 | 心跳进程没在跑（电脑休眠或被杀）；`status` 查看，`start`/`launch` 才会拉起保活 |
| `stop --all` 后仍有 active 或 `[?]` | `python scripts/colab_sync.py --release-all`（需已登录 Google 的 Edge 标签页），再用 `colab_meter.py status` 核验 |
| 域名超时 | Service 审批刚生效需几秒到十几秒，重试；确认 Tailscale ≥ 1.78 且 ACL 含 `autoApprovers.services` |

更完整的排障与模型接入流程见 [`SKILL.md`](SKILL.md) 第 7 节。

公开仓库 `notebooks/` 目录下提交的均为**脱敏构建（sanitized builds）**，仅含占位符。在本地直接运行 `make_*_notebook.py` 会把带私有配置的 Notebook 写到 `notebooks/*.ipynb`（与已提交的脱敏版同名）；向公开远端推送前，请务必用 `--sanitized --output-dir` 重新生成脱敏副本，并用 `git status` 确认没有暂存私有构建。

### 生成脱敏构建

```bash
python scripts/make_g4_huihui_v2_notebook.py --sanitized --output-dir dist/
```

`--sanitized` 会将所有敏感字段替换为占位符，并忽略本地 `.secrets.json` 与环境变量；同时它禁止与 `--sync-colab` 混用，防止误把占位符 Notebook 推送到 Colab。

## 仓库目录结构

| 路径 | 作用 |
|---|---|
| `SKILL.md` | 面向 Agent 的操作手册（前置条件、首次开机、命令速查、故障排查；CLI 主力轨道 + 浏览器后备轨道） |
| `scripts/colab_kernel.py` | 统一 G4 内核与分发器（现役 `MODELS`：`qwen38-huihui-v2`、`gemma4-31b`、`qwen38-27b-rvn`、`glm53-flash`），并导出 Colab-yoga 的 `colab_kernel_runners.json` |
| `scripts/archived_models.py` | 归档模型声明（`ARCHIVED_MODELS`），共用最新内核，可按 slug 直接启动 |
| `scripts/colab_cli.py` | 主力轨道：无头 Colab CLI 启动、状态查询与停止（自动回退至 colab_sync），含 `yoga` 子命令 |
| `scripts/colab_keepalive.py` | 主力轨道：后台分离的 30s 内核心跳 + 60s TFE ping + 45m /slots 闲置守卫 |
| `scripts/make_g4_huihui_v2_notebook.py`、`make_31b_gemma4_notebook.py`、`make_27b_rvn_notebook.py`、`make_glm53_flash_notebook.py` | 各现役模型的 Notebook 生成入口（7 行薄包装） |
| `scripts/make_g4_fn_notebook.py`、`make_g4_fn_nl_notebook.py` | 归档模型 `qwen38-fn` / `qwen38-fn-nl` 的生成入口 |
| `scripts/make_strata_fn_notebook.py` | 实验性 Strata 引擎（Unsloth UD-IQ4_XS-v31 + DFlash2）Notebook 生成器 |
| `scripts/boot_strata_trial.py` | 实验性 Strata Notebook 的分离式 CLI 启动辅助脚本 |
| `scripts/bench_ab.py` | 对在跑的服务做 A/B 基准（decode / recall / tools / compare） |
| `scripts/colab_sync.py` | 后备轨道：Colab 网页推送与单元格执行（BrowserMCP→Playwriter→PW）及 /tun/m 释放 |
| `scripts/colab_meter.py` | 后备轨道：零 OAuth status / start / stop CLI 与双轨 CU 账单 |
| `scripts/colab_pw.mjs` | Playwriter 后备后端（支持 __ANY__ 标签页复用模式） |
| `scripts/local_novel_mcp.py` | 桥接进 Colab WebUI 的本地 Windows 目录 MCP 服务 |
| `scripts/secrets_config.py` | .secrets.json / 环境变量 / 占位符加载器 |
| `scripts/parallel_download.py` | 多进程并发下载器参考实现 |
| `scripts/webui_console.py` | 首单元格控制台与停止按钮 |
| `scripts/stop-runtime.py` | 运行时释放辅助脚本 |
| `scripts/update-pi-provider.ps1` | 将服务端点注册进本地 pi provider 列表 |
| `scripts/test_*.py` | 单元测试：保活与 CLI、内核与启动指纹、Colab-yoga 对接、本地 MCP、计费、基准 |
| `references/` | 各模型说明、CU 预算核算、Tailscale 与远程接入指南 |
| `archive/multi-model-kernel/` | 已归档冻结的多模型内核、生成器与说明（122B、Nemotron、Gemma） |
| `debug-runs/` | 2026-11-07 T4 对照/心跳 A/B 实验与 G4 长稳测试日志 |
| `notebooks/` | 脱敏后的生成版 Notebook |
| `.secrets.example.json` | 配置模板 |

## 安全说明

- `.secrets.json` 与 `.colab-meter-state.json` 已列入 `.gitignore`；首次推送前请用 `git status` 确认未被暂存。
- 本地生成的非脱敏 Notebook 内嵌了模型 API Key 与 Tailscale OAuth 密钥（供运行时读取），切勿提交或分享未加 `--sanitized` 生成的 Notebook。
- Tailscale OAuth 客户端权限应严格限制为仅签发带 `tag:colab` 的认证密钥；若非脱敏 Notebook 曾离开本机，请立即轮换密钥。
- 仅限 Tailnet 内部访问的服务依赖 WireGuard 身份完成隔离；若对公网暴露，必须额外启用应用层 API Key。

## 许可证

MIT，详见 [`LICENSE`](LICENSE)。各模型权重遵循其各自的许可证，使用前请查阅对应模型卡。
