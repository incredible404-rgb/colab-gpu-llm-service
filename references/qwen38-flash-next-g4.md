# Swift-1.5-Flash-Next-Heretic-V2 (综合 1 · 原生破限版) on Colab G4

当前为 **原生 Heretic v2 破限版**（`jackjusko/Swift-1.5-Flash-Next-Heretic-ReverseEngineer` `IQ4_XS-plus` 94.70 GB 6 分片 + `ukisai` `mmproj-BF16` 视觉头 + `unsloth` `MTP Q4_K_M` 投机解码头，不挂 LoRA）。**本职定位：高智商全能黄金配置**，多轮改稿最优参数（`--parallel 1 -c 262144`、`--cache-ram` 16–64 GiB 动态上限（PLE 分片常驻，其余内存盘分片加载后删副本）、`--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0` 模型卡默认采样、`--spec-draft-n-max 3 --spec-draft-p-min 0.48`、`--batch-size 4096 --predict 65536`、`--slot-save-path`）以及启动后免密自动弹出 WebUI（`http://qwen.<YOUR_TAILNET>.ts.net:443/`）。

## 0. 思考档位与采样参数

采样按 Qwen3.8 模型卡「思考模式」默认值：`--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0`，不加 repeat/presence/frequency 惩罚与 DRY（Swift-1.5 仓库评测用的也是这组）。

档位为 `low / medium / high / xhigh（默认，同官方）`，其余值由模板 `raise_exception` 拦下。生成起手用官方模板原样：只有 `enable_thinking=false` 才关思考，low 照常思考、只要求想得短（同官方）；high 是补的一档。四档都不预填，system 顶部只加一句中文引导（low「越短越好」、medium「简短些」、high「用中文思考」、xhigh「尽量深入」）。四档文本必须两两不同，llama.cpp 靠渲染差异探测档位。历史轮次只保留当前工具链里非空的思考。

**为什么 `n_max=3 / p_min=0.48`**：创意写作是高温柔性分布，`n_max=4 / p_min=0.55` 在接受率下降时频繁回滚，反而拖慢长章节生成；3 步 + 0.48 在 G4 上更稳。

## 1. 笔记本与入口

| 项目 | 值 |
|---|---|
| 笔记本文件 | [`notebooks/qwen3.8-flash-next-g4-colab.ipynb`](../notebooks/qwen3.8-flash-next-g4-colab.ipynb)（[Colab 链接](https://colab.research.google.com/drive/<YOUR_G4_DRIVE_ID>)） |
| 生成脚本 | [`scripts/make_g4_fn_notebook.py`](../scripts/make_g4_fn_notebook.py) |
| 主模型权重 | `jackjusko/Swift-1.5-Flash-Next-Heretic-ReverseEngineer` (`IQ4_XS-plus`, 94.70 GB, 原生 Heretic v2) |
| 固定 WebUI / API | `http://qwen.<YOUR_TAILNET>.ts.net:443/`（启动自动弹窗，首格停机按钮一并关闭，免密直连） |
| Magpie 模型 ID | `Swift-1.5-Qwen3.8-Flash-Next-IQ4_XS` |

## 2. 综合 1 精确文件清单（固定 Revision）

- **主权重**：`jackjusko/Swift-1.5-Flash-Next-Heretic-ReverseEngineer` @ `76208ff7308ccebd887eec5293d3cab305091514`
  - `IQ4_XS/Swift-1.5-Qwen3.8-Flash-Next-Heretic-V2-IQ4_XS-plus-00001-of-00006.gguf` (`539_422_592` B, sha256 `b963118b880fe3b7413e53b474c1d8d3faefc1ae0d2d6bae7a7c3e9f68217b46`)
  - `IQ4_XS/Swift-1.5-Qwen3.8-Flash-Next-Heretic-V2-IQ4_XS-plus-00002-of-00006.gguf` (`28_800_138_432` B, sha256 `0e9027aa7de4817c91b5954039b563484466392e29a6b051b096f9c47a71a1c6`，含 26.8 GiB PLE 表，第一优先落 `/dev/shm` `tmpfs`)
  - `IQ4_XS/Swift-1.5-Qwen3.8-Flash-Next-Heretic-V2-IQ4_XS-plus-00003-of-00006.gguf` (`16_681_220_224` B, sha256 `95c25e4ada4fa9ce4380683e55fb9f55aa8e411dcca13bec789f3cc106ffd403`)
  - `IQ4_XS/Swift-1.5-Qwen3.8-Flash-Next-Heretic-V2-IQ4_XS-plus-00004-of-00006.gguf` (`16_984_320_096` B, sha256 `5458253ef234558243c522c8d5289b923fe28601ec6aca26239de14f03ed1951`)
  - `IQ4_XS/Swift-1.5-Qwen3.8-Flash-Next-Heretic-V2-IQ4_XS-plus-00005-of-00006.gguf` (`16_662_985_152` B, sha256 `ae9f3f684493f0107f0aa01dd603844266ee8b9a53bad53475761f2acc31202a`)
  - `IQ4_XS/Swift-1.5-Qwen3.8-Flash-Next-Heretic-V2-IQ4_XS-plus-00006-of-00006.gguf` (`15_036_859_296` B, sha256 `bcebb4fecfaaf887c2a763c252ecca17749efe175e6c9c1c45d555c0e68906c0`)
- **视觉头**：`ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF` @ `0142554c963a8653912c0c8283f05f13f63653f3` → `mmproj-Swift-1.5-Qwen3.8-Flash-Next-BF16.gguf` (`907_543_520` B)
- **MTP 投机解码头**：`unsloth/Qwen3.8-Flash-Next-GGUF` @ `766911a6b7369840a91dbcd95f9f997acaab6cd6` → `MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` (`2_786_204_800` B)

## 3. 生成与双写

`python scripts/make_g4_fn_notebook.py` 会写两份**完全相同**的 JSON（同一字符串，`newline=''`，UTF-8 + LF）：

1. `notebooks/qwen3.8-flash-next-g4-colab.ipynb`（仓库内规范产物）；
2. 当 `not SANITIZED and OUTPUT_DIR == ROOT_DIR / 'notebooks'` 且 `G:/My Drive/Colab Notebooks` 存在时，同一份 JSON 再落 `G:/My Drive/Colab Notebooks/qwen3.8-flash-next-g4-colab.ipynb`，由 Google Drive 桌面版同步到云端笔记本 `<YOUR_G4_DRIVE_ID>`。

`--sanitized`（全占位符）与 `--output-dir <临时目录>` 都不触发第 2 份写入，避免把脱敏产物同步进云端覆盖真件。`--sync-colab` 仍走原有 API 推送路径。
