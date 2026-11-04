# Qwen3.8-27B-Antislop-ZH-Q8_0 (文学 1) on Colab G4 / A100

主权重是原生 Heretic ARA 破限的 `darrellbest/Hemmingway-1-Heretic-GGUF` `Q8_0`；首格把 `Moeblack/Qwen3.8-27B-chinese-taste-sft-lora`（8000 万字中文网文 SFT）按 scale 1.0 原地烘焙进主权重，`llama-server` 不挂任何 LoRA。默认 **Colab G4 (RTX PRO 6000 Blackwell 96GB)**，**单槽 × 262,144 上下文**（`--parallel 1`，整张卡只服务一条流）；A100 40GB 回退为单槽 131,072。模型别名 `Qwen3.8-27B-Antislop-ZH-Q8_0`。

## 1. 模型与文件清单（固定 Revision；新下载的文件在加载权重时并行校验 SHA-256）

| 组件 | Hugging Face 仓库 @ Revision | 文件名 | 精确字节数 | SHA-256 |
|---|---|---|---:|---|
| **主权重 (Q8_0，原生 Heretic ARA 破限)** | `darrellbest/Hemmingway-1-Heretic-GGUF` @ `d15d84a09627b54b70f77b26af46524c7dda438f` | `Hemmingway-1-Heretic-Q8_0.gguf` | `29_047_084_288` | `1e76479ae4f05530035d788453b27fc76bcb7a46faa7a6bcee6e2b69b0c3bb18` |
| **视觉投影器 (BF16)** | `darrellbest/Hemmingway-1-Heretic-GGUF` @ `d15d84a09627b54b70f77b26af46524c7dda438f` | `Hemmingway-1-Heretic-mmproj-BF16.gguf` | `931_145_792` | `64a7db66b2cb3031c5fd9e008a78cabc5b4043c83244d8a011edea3f8239a935` |
| **MTP 投机头 (Q4_0)** | `bytkim/Qwen3.8-27B-pi-GGUF` @ `97caacb971d9246520fa8db5282a06bd36b103df` | `mtp/mtp-Qwen3.8-27B-pi-Q4_0.gguf` | `2_008_054_464` | `6ce59430911676c54217db078ed51ffa0408831a15e2acf4436302c4e8643381` |
| **中文网文 SFT LoRA（启动时烘焙进主权重）** | `Moeblack/Qwen3.8-27B-chinese-taste-sft-lora` @ `6c8950c79ade177b6fcf74c6c921369f5715b225` | `qwen3.8-27b-chinese-taste-sft-lora-q8_0.gguf` | `15_995_264` | `02211ccec9e44a0c753ea37682105b4f0faf39d37512ac8f931d9dd3164b3c3d` |

四个文件共 `32_002_279_808` B（29.80 GiB），全部直写 `/dev/shm` 内存盘并软链回 `/content/qwen27b-lit`。

聊天模板：`tokenizer.chat_template` 在 GGUF 文件偏移约 10.93 MB 处（本模型 10,934,615 B，长度 8,952 B），**64 KiB Range 读不到**，脚本读前 16 MiB。Hemmingway 与综合 1 的 Heretic 分片 1 模板逐字节相同（SHA-256 前缀 `c3cf9e34abf4f9e3`），脚本里 `reasoning_effort` 与 `preserve_thinking` 两处替换断言全部通过，并用 jinja2 对 `low/medium/high/xhigh` 与非法档位渲染过。

## 2. 启动参数与控制台

- **模型别名**：`Qwen3.8-27B-Antislop-ZH-Q8_0`（Magpie 条目：`qwen-colab/Qwen3.8-27B-Antislop-ZH-Q8_0`）
- **运行时目录**：`/content/qwen27b-lit`
- **长程与多轮润色**：`--parallel 1 -c 262144`（A100 回退 `-c 131072 --ubatch-size 512`）、`--cache-ram 16384`（16 GiB 主机内存 Prompt Cache，网页端回退分支与重新生成直接恢复 KV 与 Gated DeltaNet 状态）、`--dry-multiplier 0.8 --dry-base 1.75 --dry-allowed-length 4 --dry-penalty-last-n 1024`
- **采样与思考**：`--reasoning on --reasoning-effort low --temp 0.85 --top-p 0.92 --top-k 40 --min-p 0.05`，MTP `draft-mtp` draft 3 / p_min 0.45，Q8 KV，`--batch-size 4096`
- **中文 LoRA 烘焙**：首格用 GPU 把 LoRA 的 256 个矩阵合进 Q8_0 主权重并写 `.zh_lora_baked` 标记，同一运行时重跑首格时跳过。运行期挂 LoRA 有约 31% 的吞吐损失（见第 3 节），烘焙后没有。
- **不带 `--api-key-file`**：`svc:qwen` 只在 tailnet 内可见（未开 Funnel），网页端打开即可聊天，客户端带不带 `Authorization` 都行。接 Funnel 或 Quick Tunnel 时必须加回 `--api-key-file`。
- **不带 `--no-context-shift`**：b12032 默认关闭 context shift，挂 `--mmproj` 或记忆无法平移（Gated DeltaNet 混合记忆）时服务端还会强制关闭，所以超过上下文上限的会话会被拒，不会自动滚动。
- **首格控制台**：「🚀 服务已就绪」横幅（含 WebUI 链接）和「🛑 结束 GPU 运行时 (省 CU)」按钮。WebUI 标签页由首格输出帧打开，浏览器只允许打开者关闭它，所以只有这个按钮会一并关掉标签页；末格停机、45 分钟空闲自停或 Colab 回收运行时后需手动关。实现在 `scripts/webui_console.py`。

## 3. 选参依据（G4 实测；测量时为 MTP draft 6、LoRA 运行期挂载，请求不带 `Authorization`）

- **MTP 接受率决定 draft 长度**：中文创作文本接受率只有约 19–31%，稳态 63–83 tok/s；只有高度可预测的文本（「从 1 数到 200」）接受率 94.7%、184.63 tok/s。创作场景接受率低，所以 draft 取 3。
- **LoRA 运行期开销**：同一 400 字提示、T=0，scale 0.0 为 69.88–70.77 tok/s，scale 1.0 降到 48.14–48.36 tok/s（约 −31%）。这是改为启动时烘焙的原因。
- **验收单元**：两次正常生成、流式（ttft 0.67s）与 `get_weather` 工具调用均成功。

## 4. 文件与运行入口

- 生成脚本：[`scripts/make_27b_lit_notebook.py`](../scripts/make_27b_lit_notebook.py)（共用 [`scripts/colab_sync.py`](../scripts/colab_sync.py)）
- 本机权威笔记本：[`notebooks/qwen3.8-27b-pi-colab.ipynb`](../notebooks/qwen3.8-27b-pi-colab.ipynb)
- Colab 云端副本：<https://colab.research.google.com/drive/<YOUR_LIT_NOTEBOOK_DRIVE_ID>>
- 固定入口：网页 `http://qwen.<YOUR_TAILNET>.ts.net:443/`，接口 `http://qwen.<YOUR_TAILNET>.ts.net:443/v1`
- 在旧模型的运行时上原地升级：先停掉旧 `llama-server`，清空 `/dev/shm/qwen27b-lit` 与指向它的软链，再跑首格；`--sync-colab --run-cell1` 默认经 Playwriter 在后台 Colab 标签里执行（后台标签照常运行）；Playwriter 不可用而降级到 `playwright-cli -s=edge` 时，Colab 标签页会被留在前台，因为后台标签会被浏览器冻结、`eval` 挂住。
- 在 `pi` 中调用：
  ```powershell
  pi --model magpie/qwen-colab/Qwen3.8-27B-Antislop-ZH-Q8_0 --thinking low
  ```
