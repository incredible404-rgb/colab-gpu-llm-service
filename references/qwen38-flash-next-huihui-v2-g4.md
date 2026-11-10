# Huihui-Qwen3.8-Flash-Next-abliterated-v2 UD-Q4_K_XL-V2 on Colab G4

基于 [`huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF`](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF) 的最新 `unsloth-UD-Q4_K_XL-V2`（作者用 `calibration3.txt` / `imatrix_micro3.gguf` 重校，并在 v2 中不消融路由专家层，专为小说与编程调整）。采样、4 档中文思考补丁、MTP 投机解码、262k 上下文与 WebUI 本机小说目录桥全部由 `colab_kernel.py` 统一生成，可与 `qwen38-fn` / `qwen38-fn-nl` 免释放热切换。

| 项目 | 值 |
|---|---|
| slug / 生成脚本 | `qwen38-huihui-v2` / [`scripts/make_g4_huihui_v2_notebook.py`](../scripts/make_g4_huihui_v2_notebook.py) |
| 笔记本 | [`notebooks/qwen3.8-flash-next-huihui-v2-g4-colab.ipynb`](../notebooks/qwen3.8-flash-next-huihui-v2-g4-colab.ipynb)（Drive ID 存 `.secrets.json` 的 `g4huihui_drive_id`） |
| CLI 启动 | `python -X utf8 scripts/colab_cli.py start qwen38-huihui-v2`（或 `launch`） |
| Magpie 模型 ID | `qwen-colab/Huihui-Qwen3.8-Flash-Next-abliterated-v2-UD-Q4_K_XL` |
| 主权重 | `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF` @ `cbc7fba13f1939ac3fa42b18d77ccf43847d74ef`，`unsloth-UD-Q4_K_XL-V2/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{1..4}-of-00004.gguf`，共 111.33 GB（103.69 GiB） |
| 视觉头 | 同仓 `mmproj-Qwen3.8-Flash-Next-BF16.gguf`（`907_543_008` B，sha256 `b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0`） |
| MTP 草稿头 | `unsloth/Qwen3.8-Flash-Next-GGUF` `MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`（`2_786_204_800` B，与 `qwen38-fn` 共用） |

## 为什么选 `unsloth-UD-Q4_K_XL-V2` 而不是 `UD-Q5_K_XL`

- GGUF 张量表实测：`UD-Q5_K_XL`（6 分片共 158.29 GB）除掉第 3 片 50.66 GiB 的 `Q8_0` PLE 表后，纯 GPU 主干仍有 **96.75 GiB（103.87 GB）**，还没算视觉头、MTP 和 KV 缓存就已经超出单张 G4（RTX PRO 6000 96GB，实测 95.6 GiB）的物理显存上限。
- `unsloth-UD-Q4_K_XL-V2`（4 分片共 111.33 GB / 103.69 GiB）：
  - 第 1 片 `00001-of-00004.gguf`（10.95 MB）：纯元数据与词表（`n_tensors=0`），含 Unsloth 版 `tokenizer.chat_template`（内核自动剥掉 Unsloth 注入的 3 行 `high -> xhigh` 折合后再打 4 档中文思考补丁）。
  - 第 2 片 `00002-of-00004.gguf`（49.86 GB / 46.44 GiB）：含 `output.weight`、`per_layer_token_embd.weight`（30.53 GiB，`IQ4_NL`）及前 11 层（约 15.9 GiB）。标记为 `ple_shard_name`（`priority=1`），第一优先落 `/dev/shm` `tmpfs` 并常驻。
  - 第 3、4 片（49.38 GB + 12.09 GB）：第 11–47 层，加载进显存后如落在内存盘则自动删除副本回收给 Prompt Cache。
  - 纯 GPU 主干共 **78.55 GB（73.16 GiB）**，加上 `mmproj`（0.85 GiB）、`MTP`（2.60 GiB）与 262k KV 缓存后，G4 实测显存占用 **92,337 / 97,887 MiB（剩余 4,914 MiB）**，刚好把单张 G4 吃满。

## 实测（2026-11-09，G4，CLI 启动）

| 指标 | 实测值 |
|---|---|
| 显存已用 / 总量 / 剩余 | **92,337 / 97,887 / 4,914 MiB** |
| 主机内存（已释放显存专用分片副本后） | `total 176 GiB, used 65 GiB, available 111 GiB`（第 2 片常驻 `/dev/shm`，第 1/4 片、mmproj、MTP 内存盘副本已删） |
| 冷启动耗时（CLI，含下载） | **233.5 s**（并发下载 107.13 GiB 用 212.8 s，均速 515 MB/s；内存盘 61.1 GiB、本地盘 46.0 GiB） |
| `/props` 门禁 | `reasoning_effort_levels: ['low', 'medium', 'high', 'xhigh']`，`n_ctx: 262144` |
| 长篇小说开篇试写（3,159 prompt tokens → 2,996 output tokens，`high` 思考档） | 预填 **3,446.1 tok/s**（0.92 s）｜解码 **104.4 tok/s**（28.67 s）｜MTP 草稿接受 **1,635 / 2,909（56.2%）** |

## 分片大小与 SHA-256

- `unsloth-UD-Q4_K_XL-V2/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`：`10_946_624` B，`4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082`
- `unsloth-UD-Q4_K_XL-V2/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf`：`49_859_583_136` B，`ab283aba22daee321944dad65b09180810140fc75502c0791200cf01b85f9c75`
- `unsloth-UD-Q4_K_XL-V2/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf`：`49_376_141_504` B，`2a8d2e69ba47fef6b42c953c07a24b5d4c59c1c2cdc89b2bc17e13e000a62fa0`
- `unsloth-UD-Q4_K_XL-V2/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf`：`12_087_983_520` B，`0d626450754c554235f93653b8f7c874c7c2d35380b9efab5b0852d616b84702`
