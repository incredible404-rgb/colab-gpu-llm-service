# Strata · Qwen3.8-Flash-Next UD-IQ4_XS-v31（实验）

未在本机 G4 上跑过。现役 llama.cpp 笔记本（Swift-1.5 Heretic IQ4_XS-plus / IQ4_NL）不要用这本替换。

| 项目 | 值 |
|---|---|
| 笔记本 | [`notebooks/qwen3.8-flash-next-strata-g4-colab.ipynb`](../notebooks/qwen3.8-flash-next-strata-g4-colab.ipynb) |
| 生成 | `python scripts/make_strata_fn_notebook.py` |
| 引擎 | Strata Terminal，`install.sh` 取 GitHub latest；模型目录 `/content/strata-qwen` |
| 权重 | `--family unsloth --model UD-IQ4_XS-v31`（plan 失败才退回 `UD-IQ4_XS`） |
| 启动 | `--context 256k --parallel 1 --drafting dflash2 --kv int8 --vision on --access local --port 8081`，另加 `--set spec=6 --set spec-min-p=0.7`（该版本的 `--help` 里没有的旗标会省掉） |
| 入口 | 现有 `svc:qwen`（`http://<tailnet fqdn>:443/v1`）。不注册 Magpie |
| 不做什么 | 不进 `colab_kernel.py`，不热切换，不接小说 MCP，不由 `colab_cli.py` 启动 |

## 为什么是这一档

同卡社区报告（RTX PRO 6000 Blackwell 96GB，253GB 内存，Windows，Strata 0.2.11 / engine 0.1.57，UD-IQ4_XS-v31，DFlash2 `--spec 6 --spec-min-p 0.7`，KV 全驻留）：聊天解码中位数约 276–278 tok/s，agent 约 263–267，16K 预填约 6700 tok/s。专家约 56 GiB，能放进 96GB 显存；那次对比里 KV 是否全驻留不改变解码。本机 G4 是同型号显存、主机内存大约 176GB，解码有机会接近，但不能把那份数当成已经在 Colab 上测到。

Strata 文档：UD-IQ4_XS 相对 Q8_0 的首 token 一致率 88.9%，困惑度 4.120 对 4.052。下载约 94GB。这不是 Swift-1.5 Heretic。

DFlash2 多槽位不可用，所以 `--parallel 1`。不要和 llama.cpp 那本同时开。

## 首格会拒绝的情况

- 不是 RTX PRO 6000 或显存 < 90GB
- 驱动主版本 < 580（CUDA 13 引擎）
- `/content` 剩余 < 150 GiB

`strata launch` 退出码 0 不代表在服务（没有合适引擎时它可能只下完权重就结束）。首格还要看到 `/health` 或 `/v1/models`。

## 验收

首格打印 GPU、驱动、剩余磁盘、`strata --version`、plan 选中的型号、`/v1/models` 和 Tailscale 地址。探针格对同一句短提示跑两次，打印墙钟 tok/s。用完跑停机格，再用 `python scripts/colab_meter.py status` 确认 0 个活动会话。
