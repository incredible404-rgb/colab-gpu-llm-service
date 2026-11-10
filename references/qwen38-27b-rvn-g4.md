# Qwen3.8-27B-RVN-Heretic-BF16 · Colab G4

基于 [`0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF`](https://huggingface.co/0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF) 的 `RVN-BF16.gguf`（53.81 GB / 50.11 GiB 无损 BF16 原精度，在 `trohrbaugh/Qwen3.8-27B-heretic-ara` 基础上再做两轮全权重 ARA 破限，KL 损伤低至 `0.0085`，拒答率 `0–1/100`）。由 `colab_kernel.py` 统一生成，支持 `colab_cli.py start qwen38-27b-rvn` 无头启动以及与 `qwen38-huihui-v2` / `gemma4-31b` 免释放热切换。

| 项目 | 值 |
|---|---|
| slug / 生成脚本 | `qwen38-27b-rvn` / [`scripts/make_27b_rvn_notebook.py`](../scripts/make_27b_rvn_notebook.py) |
| 笔记本 | [`notebooks/qwen3.8-27b-rvn-g4-colab.ipynb`](../notebooks/qwen3.8-27b-rvn-g4-colab.ipynb)（Drive ID 存 `.secrets.json` 的 `rvn27b_drive_id`） |
| CLI 启动 | `python scripts/colab_cli.py start qwen38-27b-rvn`（或 `launch qwen38-27b-rvn`） |
| Magpie 模型 ID | `qwen-colab/Qwen3.8-27B-RVN-Heretic-BF16` |

## 权重与组件

| 角色 | 仓库 @ revision | 文件 | 字节 | SHA-256 |
|---|---|---|---:|---|
| model | `0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF` @ `ab8d7bd81d03117501e4da60cccdff4c80f0eea0` | `RVN-BF16.gguf` | `53_808_281_888` | `461fa227cb3d0bb49a3e8e45ec2b650f9a2e593c51a92144d5ac819f0ee8f37d` |
| mmproj | `unsloth/Qwen3.8-27B-GGUF` @ `4ca720788d1e01f1bff70c033e0d0028fd02e502` | `mmproj-BF16.gguf` | `931_146_432` | `83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53` |
| draft | `unsloth/Qwen3.8-27B-GGUF` @ `4ca720788d1e01f1bff70c033e0d0028fd02e502` | `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` | `1_369_590_656` | `50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e` |

## 运行与模板补丁

- 无 PLE 表（27B 全稠密 Gated DeltaNet 混合架构，64 层），3 个文件合计 `56.11 GB`（`52.26 GiB`），全部直写 `/dev/shm` `tmpfs` 并加载进 G4 96GB 显存，加载后自动删除内存盘副本回收给 Prompt Cache。
- 采样按 Qwen3.8-27B 模型卡思考模式默认：`--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0`。
- 聊天模板：内嵌官方 8,952 字节 Qwen3.8 模板，由 `qwen38_fn` 模板族注入四档中文思考引导（`low` / `medium` / `high` / `xhigh`，默认 `high`）。
- MTP 投机解码：`--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.45`（选用 Unsloth 含完整 18 张量的 `mtp-Qwen3.8-27B-Q4_0.gguf`；同仓 `mtp-RVN.gguf` 仅 16 张量缺 `token_embd` 与 `output_norm`，不能直接传 `--model-draft`）。
