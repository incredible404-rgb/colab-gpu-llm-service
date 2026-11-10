---
name: colab-gpu-llm-service
description: 在 Google Colab 付费 GPU（默认 G4 RTX PRO 6000 96GB）上部署并管理 GGUF 大模型服务（llama.cpp、MTP 投机解码），经 Tailscale 固定域名提供 OpenAI 兼容接口，无头 CLI + 30 秒内核心跳保活。用于首次安装与第一次开机、启动/停止/续命 Colab GPU、查 CU 余额与燃烧率、新增/切换/归档模型、显存与分片核算、清理 [?] 幽灵实例、排障，以及对接可选的 Colab-yoga 托盘工作台。Deploy and run GGUF LLMs on a paid Colab G4 via google-colab-cli (kernel heartbeat, Tailscale Service, CU accounting).
---

# Colab GPU LLM Service

把有限时、每局清空 `/content` 的 Colab G4（RTX PRO 6000 Blackwell 96GB 显存 + 176 GiB DDR5 内存）做成可重复冷启动的 GGUF 模型服务。权重每次从 Hugging Face 固定 revision 经 32 路并发直写 `/dev/shm` 内存盘，对外通过 Tailscale Service 暴露固定地址 `http://qwen.<你的tailnet>.ts.net:443`（OpenAI 兼容 `/v1`，另带 WebUI）。

**给 agent 的硬规则**

1. 开机会烧付费 CU：起 GPU 前先向用户确认；任务结束必须 `stop --all` 并核验 `0 active`。
2. `.secrets.json` 与不带 `--sanitized` 生成的 notebooks 含真实密钥，不回显、不提交、不外传。
3. 状态以命令实时输出为准（`status`、`colab_kernel.py --list`），不要凭本文档回答余额、费率或模型清单。

---

## 0. 前置条件（先逐项自检）

| 项 | 要求 | 自检 |
|---|---|---|
| Colab 付费额度 | 付费计算单元（CU）：Colab Pro / Pro+ / Ultra 订阅或按量购买，账号能选 G4。免费档不行。费率与额度会变，估算见 [`references/cu-budget.md`](references/cu-budget.md) | `colab usage`（装好 CLI 后） |
| 主机 | Windows 10/11 + Python 3.10+ + Git。脚本用计划任务与 `taskkill` 清理保活进程，按 Windows 编写；macOS/Linux 未验证 | `python --version` |
| Python 依赖 | 仅 `requests`，其余为标准库 | `pip install requests` |
| 官方 Colab CLI | **google-colab-cli 0.7.5**，固定此版本：保活机制按它 21–35 分钟回收实例的行为设计 | 见下 |
| Tailscale | 本机已安装并登录；tailnet 里建好 `svc:qwen` Service、`tag:colab` 标签、OAuth 客户端；客户端 ≥ 1.78 | `tailscale version` |
| HF 令牌（可选） | 受限/限速模型需要；本机 `HF_TOKEN` 环境变量或 `~/.cache/huggingface/token` 会被自动转发到运行时 | |
| Colab-yoga（可选） | Windows 托盘多模态工作台（视频/生图/ComfyUI），与本技能共用同一台 G4。不装不影响任何核心命令 | `python scripts/colab_cli.py yoga status` |
| Edge + BrowserMCP、Node.js（可选） | 仅后备轨道（网页同步、零 OAuth 查账与强杀）需要 | |

安装官方 CLI 并授权一次（先装 [`uv`](https://docs.astral.sh/uv/)，例如 `winget install astral-sh.uv` 或 `pip install uv`）：

```bash
uv tool install "google-colab-cli==0.7.5"
colab sessions          # 首次打印 OAuth 链接，浏览器授权后令牌存于 ~/.config/colab-cli/token.json
colab usage             # 应显示 CU 余额；报授权错误就重跑 colab sessions
```

Tailscale 一次性配置（ACL 的 `tagOwners` + `autoApprovers.services`、定义名为 `qwen` 的 Service、建只签发 `tag:colab` 认证密钥的 OAuth 客户端）按 [`references/tailscale-serve.md`](references/tailscale-serve.md) 的“一次性配置”三步做，拿到的 `client_id` / `client_secret` 填进 `.secrets.json`。

---

## 1. 从克隆到第一次开机（最短路径）

```bash
git clone https://github.com/incredible404-rgb/colab-gpu-llm-service.git
cd colab-gpu-llm-service
pip install requests
cp .secrets.example.json .secrets.json      # 按第 2 节填写；该文件已被 .gitignore 忽略

python scripts/colab_kernel.py --list       # 看现役/归档模型 slug；start 不带 slug 时用现役表第一个
python scripts/colab_cli.py status          # 只读自检：CLI 已授权、读得到 CU 余额，不分配 GPU
python scripts/colab_cli.py start           # 分配 G4 → 跑首格 → 检查 Tailscale /v1/models → 跑验证格
```

`start` 成功的标志是出现 `[colab_cli] start OK in <秒>s`，之后可用：

```bash
curl -H "Authorization: Bearer <api_key>" http://qwen.<你的tailnet>.ts.net:443/v1/models
# 或浏览器打开 http://qwen.<你的tailnet>.ts.net:443/ 使用 WebUI（仅 tailnet 内可见，免密）
```

用完立即释放：

```bash
python scripts/colab_cli.py stop --all      # 核验末行 Active assignments: 0 | Burn rate: 0.00 CU/h
```

冷启动耗时取决于模型体量（百 GB 级权重要下载数分钟）。`start` 会在后台拉起保活进程，45 分钟无推理流量自动 `colab stop`。`launch` 是后台脱离拉起版本，它读取 `notebooks/` 里的私有构建，需先 `python scripts/colab_kernel.py <slug>` 生成；这个文件含真实密钥且与仓库里已提交的脱敏版同名，提交前先 `git status` 确认没被暂存。

---

## 2. `.secrets.json`：由 `.secrets.example.json` 复制并填写

前四项是 `start` 的必填项，缺一项直接报 `missing ... in .secrets.json / environment`。每个字段也可用同名环境变量代替，优先级 `.secrets.json` → 环境变量 → 占位符。

| 字段 | 环境变量 | 含义 |
|---|---|---|
| `api_key` | `LLAMA_API_KEY` | 你自定义的任意长随机串，写入运行时的 llama-server 作 Bearer 密钥，也是你访问 `/v1` 的密钥 |
| `ts_client_id` | `TS_CLIENT_ID` | Tailscale OAuth 客户端 ID（权限仅 Auth Keys: Write，标签 `tag:colab`） |
| `ts_client_secret` | `TS_CLIENT_SECRET` | 上述 OAuth 客户端的 secret，运行时用它现签一次性、临时的节点认证密钥 |
| `tailnet_fqdn` | `TAILNET_FQDN` | 服务域名，形如 `qwen.<你的tailnet>.ts.net`（`qwen` 是你在 Tailscale 里建的 Service 名） |
| `local_tailscale_ip` | `LOCAL_TAILSCALE_IP` | 可选。运行 `local_novel_mcp.py` 那台主机的 Tailscale IP，WebUI 本机文件桥用；不用文件桥可不填 |
| `*_drive_id`（每个现役模型一项） | 见 `scripts/secrets_config.py` | 可选。该模型 notebook 在你 Google Drive 里的文件 ID，仅 `--sync-colab`（网页后备轨道）需要；`colab_kernel.py` 生成时会从本机 Drive 客户端数据库自动解析并回填 |

示例文件里的 `<YOUR_...>` 都是占位符，不是可用值。其余可调环境变量：

| 环境变量 | 作用 |
|---|---|
| `COLAB_YOGA_DIR` | Colab-yoga 安装目录（默认 `%LOCALAPPDATA%\Programs\Colab-yoga`） |
| `COLAB_NOTEBOOKS_DIR` | 生成 notebook 后镜像到的 Drive “Colab Notebooks”目录；不设则自动探测各盘符下的 `My Drive`，探测不到就不镜像 |
| `DRIVEFS_DB` | 指定 Drive 客户端元数据库路径（自动解析 Drive 文件 ID 用） |
| `NOVEL_ROOT` | `local_novel_mcp.py` 向 WebUI 暴露的本机目录 |
| `HF_TOKEN` | 转发到运行时的 Hugging Face 令牌 |

---

## 3. 命令速查

```bash
# 启停与状态（日常主力：官方 CLI + 30s 内核心跳）
python scripts/colab_cli.py start  [<slug>] [--gpu G4] [--idle-min 45] [--no-verify] [--keep-on-fail]
python scripts/colab_cli.py launch [<slug>]          # 后台脱离拉起（LAUNCH_DETACHED=1），轮询 /health
python scripts/colab_cli.py verify [<slug>]          # 只重跑验证格
python scripts/colab_cli.py status                   # CU 余额、燃烧率、活跃会话、心跳计数
python scripts/colab_cli.py stop [<slug> | --all]    # 释放 GPU；遇 [?] 幽灵实例自动回调 colab_sync 强杀

# 模型清单 / 探测 / 生成
python scripts/colab_kernel.py --list                # 现役 MODELS 与归档 ARCHIVED_MODELS
python scripts/colab_kernel.py --probe-hf <repo> [rev]  # 新模型 HF 预检：revision、字节数、sha256、张量数、PLE 表、聊天模板
python scripts/colab_kernel.py --all                 # 生成全部现役 notebook（会话在跑时别重跑）

# 可选：Colab-yoga 协同（旧子命令名 easycolab 仍可用）
python scripts/colab_cli.py yoga [gui | wake | status | stop]

# 后备轨道与账单（浏览器标签页，零 OAuth）
python scripts/colab_meter.py status                 # 云端账本 + 当前分配
python scripts/colab_meter.py stop --all             # 强杀并打印双轨账单
python scripts/colab_sync.py --release-all           # 强杀全部活动实例并核验

# 测试
cd scripts && python -X utf8 -m unittest discover -p "test_*.py"
```

---

## 4. 同核双轨、保活与 Colab-yoga 协同

`scripts/colab_kernel.py` 生成同一份首格代码，供无头 CLI（Track A）与网页版（Track B）共用。

| 轨道 | 角色 | 适用场景 |
|---|---|---|
| **Track A · 官方 Colab CLI + 30s 内核心跳** | **主力（日常默认）** | 终端无头起停任意现役或归档模型；自带心跳防 21–35 分钟回收 + 45 分钟空闲自停 |
| **Colab-yoga 协同工作台**（可选） | 多模态共驻 | 本地托盘 App 与 Track A 双向复用同一台 G4（共用 `~/.config/colab-cli/` 的令牌与会话） |
| **Track B · 浏览器网页版 + 零 OAuth `/tun/m`** | 备用与兜底 | ① 网页首格按钮与第 4 格免释放热切换模型；② CLI OAuth 异常时复用已登录 Edge；③ 强杀 CLI 无法按名停止的 `[?]` 幽灵会话 |

- **30 秒内核心跳（`colab_keepalive.py`）**：`google-colab-cli` 0.7.5 创建的实例若无新代码执行，21～35 分钟会被回收（推理流量、长阻塞单元、60s HTTP ping 均不算数）。`start` / `launch` / `yoga wake` 自动在后台每 30 秒执行一行 `colab exec` 心跳，每 60 秒 ping TFE，并轮询 `/slots`，满 45 分钟无推理自动 `colab stop`。电脑休眠心跳即停，约 20 分钟后云端自动回收。
- **Colab-yoga**（[github.com/incredible404-rgb/Colab-yoga](https://github.com/incredible404-rgb/Colab-yoga)）：安装到 `%LOCALAPPDATA%\Programs\Colab-yoga`（含 `Colab-yoga.exe`、`runtime\python.exe`、`bridge\colab_yoga_bridge.py`）。找不到时 `colab_cli.py yoga ...` 会报出安装位置与下载地址。`colab_kernel.py` 生成时把各现役模型的无头首格导出为 `Colab-yoga\colab_kernel_runners.json`（目录不存在则跳过并提示）。先 `colab_cli.py start` 再在 Colab-yoga 里唤醒，或反过来，都经 `~/.config/colab-cli/sessions.json` 复用同一台 G4，不重复开卡；Colab-yoga 只清理 `18080..18999` 端口进程，不会误杀 `:8081` 的 `llama-server`。

---

## 5. 模型清单与新增/归档流程

模型声明字典分置两处，清单以 `python scripts/colab_kernel.py --list` 为准：

- **现役表 `MODELS`（[`scripts/colab_kernel.py`](scripts/colab_kernel.py)）**：`--all` 生成、热切换下拉框与 `--sync-magpie` 暴露的唯一来源。
- **归档表 `ARCHIVED_MODELS`（[`scripts/archived_models.py`](scripts/archived_models.py)）**：保留完整声明，共用最新内核；不进暴露列表，但可按 slug 单独生成或启动。

现役模型（规格、显存布局与实测见各自的参考文档）：

| slug | 服务端 `alias` | 参考文档 |
|---|---|---|
| `qwen38-huihui-v2`（默认） | `Huihui-Qwen3.8-Flash-Next-abliterated-v2-UD-Q4_K_XL` | [`qwen38-flash-next-huihui-v2-g4.md`](references/qwen38-flash-next-huihui-v2-g4.md) |
| `gemma4-31b` | `Gemma-4-31B-it-ARA-Heretic-Q8_0` | [`gemma4-31b-g4.md`](references/gemma4-31b-g4.md) |
| `qwen38-27b-rvn` | `Qwen3.8-27B-RVN-Heretic-BF16` | [`qwen38-27b-rvn-g4.md`](references/qwen38-27b-rvn-g4.md) |
| `glm53-flash` | `GLM-5.3-Flash-Uncensored-GSQ-RCO-Q4` | [`glm53-flash-g4.md`](references/glm53-flash-g4.md) |

归档：`qwen38-fn`、`qwen38-fn-nl`、`qwen35-122b`、`nemotron-120b`、`gemma4-31b-ortenzya`。

### A. 新增模型四步法（先探测 → 声明 → 单测 → 分发）

1. **HF 预检（必做，零盲猜）**：`python scripts/colab_kernel.py --probe-hf <hf_repo_id> [revision]`。它查询 HF API 并 Range 读取每个 `.gguf` 前 16 MiB，输出固定 `rev`、精确字节数、LFS `sha256`、张量数 `n_tensors`、是否含 `per_layer_token_embd.weight`（PLE 表）与 `tokenizer.chat_template`。核对三条硬边界：
   - **显存红线（G4 物理显存 `95.6 GiB` / `97,887 MiB`）**：含 PLE 表的模型用总权重减去 PLE 分片（CPU mmap 常驻 `/dev/shm`）得**纯 GPU 主干**；纯 GPU 主干 + `mmproj` + `MTP` + 262k KV 须 `≤ 92 GiB`。含 PLE 的分片设为 `ple_shard_name` 且 `priority: 1`。超大 MoE 用 `extra_server_flags: ['--n-cpu-moe', '<层数>']` 把前 N 层路由专家分流到 176 GiB 主机内存，使进显存部分控制在 `80–85 GiB`。
   - **MTP 草稿头红线**：`llama-server --model-draft` 把草稿文件当独立模型加载，草稿 GGUF 必须带齐 `token_embd.weight` 与 `output_norm.weight`（用 `--probe-hf` 看 `n_tensors`，不要用缺张量的精简草稿头）。
   - **模板族（`template_family`）与转义红线**：`'qwen38_fn'`（Qwen3.8-Flash-Next 与 27B，自动剥离 Unsloth 注入的 `high -> xhigh` 折合再打四档中文思考补丁）、`'gemma4'`（Gemma 4 Canonical 模板，`enable_thinking` 默认改 true）、`'glm5_next'` / `'qwen35'` / `'nemotron'`。在 `colab_kernel.py` 的 `oneclick_code = f"""..."""` 里新增模板匹配字符串时，Jinja 里的 `\n` 必须写成 **4 个反斜杠 `\\\\n`**。
2. **登记进 `MODELS`**：在 `colab_kernel.py` 的 `MODELS` 新增 `<slug>` 条目，按需新增 7 行薄入口 `scripts/make_<name>_notebook.py` 与 `references/<name>.md`。`drive_key` 由 `generate_one` 自动注册，并从本机 Drive 客户端数据库提取 `cloud_id` 写入 `.secrets.json`，无需手改 `secrets_config.py`。
3. **跑本地单测**：`cd scripts && python -X utf8 -m unittest discover -p "test_*.py" -v`。套件对 `MODELS` 与 `ARCHIVED_MODELS` 全部条目做 AST 语法检查、生产路径零探针检查、启动指纹自检与模板补丁实跑。
4. **一键生成并分发**：`python scripts/colab_kernel.py --all`；用 Magpie 网关的再加 `--sync-magpie`（其他人省略）。Colab 会话正在运行时不要重跑生成，以免覆盖云端正在使用的笔记本。

### B. 归档与恢复（单点移动）

- **归档**：把条目从 `MODELS` 剪到 `scripts/archived_models.py` 的 `ARCHIVED_MODELS`，移走对应 `make_*_notebook.py`，再跑 `colab_kernel.py --all`（用 Magpie 则带 `--sync-magpie`，它会把该模型从暴露列表移除并 reset 其配置）。
- **临时运行归档模型**：不改文件，直接 `python scripts/colab_cli.py start <archived-slug>`。
- **恢复为现役**：把条目移回 `MODELS`，重跑 `colab_kernel.py --all`。

---

## 6. 评测、网络入口与账单

- **A/B 基准（[`scripts/bench_ab.py`](scripts/bench_ab.py)）**：改参数（MTP 步数、KV 类型、量化、ubatch）时对已在跑的服务执行 `decode`、`recall`（8k/32k/128k/216k 大海捞针）、`tools`（48 次工具调用）与 `compare a.json b.json`。上线门槛：`recall` 全过、`tools` 0 失败。
- **网络入口**：日常自用走 **Tailscale Service**（`svc:qwen`，仅 tailnet 私网可见，免 `--api-key-file` 以便 WebUI 免密即开即用；经 SOCKS5 反连本机 `:3426` 的 `local_novel_mcp.py` 读写 `$NOVEL_ROOT`），详见 [`references/tailscale-serve.md`](references/tailscale-serve.md)。要暴露公网（Cloudflare Named Tunnel / Funnel）必须启用 `--api-key-file`，详见 [`references/secure-remote-service.md`](references/secure-remote-service.md)。
- **零 OAuth 查账与幽灵实例强杀**：`colab sessions` 出现 `[?]`（无本地会话名的云端实例）时，`colab_cli.py stop --all` 会自动调用 `colab_sync.py --release-all`，经已登录 Edge 的 `/tun/m/unassign/<endpoint>` 强杀并核验 `0 active / 0.00 CU/h`；也可直接 `colab_meter.py status` / `stop --all` 打印云端账本差值 + 墙钟折算双轨账单。

---

## 7. 故障排查

| 现象 | 原因与处理 |
|---|---|
| `missing ... in .secrets.json / environment` | 必填四项（`api_key`、`ts_client_id`、`ts_client_secret`、`tailnet_fqdn`）仍是占位符；回第 2 节填写 |
| `colab` 找不到，或 `colab sessions` 要求授权 | 重做第 0 节：`uv tool install "google-colab-cli==0.7.5"`，再跑一次 `colab sessions` 完成 OAuth |
| `Colab-yoga not found at ...` | 未装可选的 Colab-yoga：从其仓库安装，或设 `COLAB_YOGA_DIR` 指向现有安装；核心命令不受影响 |
| `colab new --gpu G4` 报 `Backend rejected accelerator` | G4 只有单一 176 GiB 规格，传 `--high-mem` 会被拒（`colab_cli.py` 已自动跳过）；也可能是账号无 G4 额度或当日无空闲卡，稍后重试 |
| 实例 20～35 分钟后无故消失 | 心跳进程没在跑（电脑休眠或进程被杀）；`status` 看心跳计数，`start` / `launch` 才会拉起保活 |
| `stop --all` 后仍有 active 或 `[?]` | 走第 6 节的强杀：`colab_sync.py --release-all`（需已登录 Google 的 Edge 标签页），再 `colab_meter.py status` 核验 |
| 域名超时或解析不到 | Service 审批刚生效需几秒到十几秒，重试；确认本机 Tailscale ≥ 1.78 且在同一 tailnet，ACL 含 `autoApprovers.services`（见 `references/tailscale-serve.md`） |
| 首格报「需要重新启动服务，但缺少上次为扩大提示缓存而删掉的分片副本」 | 配置变更触发重启，再跑一次首格即自动重下被回收的显存专用分片 |
| 测试失败 | 在 `scripts/` 下跑 `python -X utf8 -m unittest discover -p "test_*.py"`，按失败的模块名定位；改过 `MODELS` 声明多半是 AST/指纹检查在拦 |
