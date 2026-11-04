# Swift-1.5-Flash-Next-Heretic-V2 (综合 1 · 原生破限版) on Colab G4

当前为 **原生 Heretic v2 破限版**（`jackjusko/Swift-1.5-Flash-Next-Heretic-ReverseEngineer` `IQ4_XS-plus` 94.70 GB 6 分片 + `ukisai` `mmproj-BF16` 视觉头 + `unsloth` `MTP Q4_K_M` 投机解码头，不挂 LoRA）。多轮改稿与长程任务最优参数（`--parallel 1 -c 262144`、`--cache-ram 16384`、`--temp 0.25 --top-p 0.85 --top-k 20 --min-p 0.05` 代码高确定性采样、`--dry-multiplier 0.5 --dry-allowed-length 6 --dry-penalty-last-n 2048` 工程防死循环、`--spec-draft-n-max 4 --spec-draft-p-min 0.55`、`--batch-size 4096 --predict 65536`、`--slot-save-path`）以及启动后免密自动弹出 WebUI（`http://qwen.<YOUR_TAILNET>.ts.net:443/`）。

## 1. 两套 Colab 笔记本分工

| 项目 | 文学 1（27B 原生破限文学精修版） | 综合 1（125B-A6B MoE 原生破限旗舰版） |
|---|---|---|
| 笔记本文件 | [`notebooks/qwen3.8-27b-pi-colab.ipynb`](../notebooks/qwen3.8-27b-pi-colab.ipynb)（[Colab 链接](https://colab.research.google.com/drive/<YOUR_LIT_NOTEBOOK_DRIVE_ID>)） | [`notebooks/qwen3.8-flash-next-g4-colab.ipynb`](../notebooks/qwen3.8-flash-next-g4-colab.ipynb)（[Colab 链接](https://colab.research.google.com/drive/<YOUR_G4_NOTEBOOK_DRIVE_ID>)） |
| 生成脚本 | [`scripts/make_27b_lit_notebook.py`](../scripts/make_27b_lit_notebook.py) | [`scripts/make_g4_fn_notebook.py`](../scripts/make_g4_fn_notebook.py) |
| 主模型权重 | `darrellbest/Hemmingway-1-Heretic-GGUF` (`Q8_0`, 29.05 GB, 0/100 拒答) + GPU 原地永久固化 `Moeblack` 8000万字中文小说 LoRA（零运行期减速） | `jackjusko/Swift-1.5-Flash-Next-Heretic-ReverseEngineer` (`IQ4_XS-plus`, 94.70 GB, 原生 Heretic v2) |
| 固定 WebUI / API | `http://qwen.<YOUR_TAILNET>.ts.net:443/`（启动自动弹窗，首格停机按钮一并关闭，免密直连） | `http://qwen.<YOUR_TAILNET>.ts.net:443/`（启动自动弹窗，首格停机按钮一并关闭，免密直连） |
| Magpie 模型 ID | `Qwen3.8-27B-Antislop-ZH-Q8_0` | `Swift-1.5-Qwen3.8-Flash-Next-IQ4_XS` |

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
