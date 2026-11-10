# GLM-5.3-Flash-Uncensored-GSQ-RCO-Q4 · Colab G4

基于 [`GCSA-AiLab/GLM-5.3-Flash-Uncensored-RCO-GSQ-GGUF`](https://huggingface.co/GCSA-AiLab/GLM-5.3-Flash-Uncensored-RCO-GSQ-GGUF) 的 `Q4/GLM-5.3-Flash-Uncensored-GSQ-RCO-Q4.gguf`（137.07 GB / 127.66 GiB，已将同作者 [`GCSA-AiLab/GLM-5.3-Flash-Abliterated-LoRA-GGUF`](https://huggingface.co/GCSA-AiLab/GLM-5.3-Flash-Abliterated-LoRA-GGUF) 的 `LoRA-v2` 秩 1 消融增量预合并进底座 [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) 320B-A18B 权重并做 GSQ-RCO 混合量化：KL 散度 `0.011077`、StrongREJECT rubric `0.972222`、拒答率 `1.67%`）。由 `colab_kernel.py` 统一生成，支持 `colab_cli.py start glm53-flash` 无头启动以及与 `qwen38-huihui-v2` / `gemma4-31b` / `qwen38-27b-rvn` 免释放热切换。

| 项目 | 值 |
|---|---|
| slug / 生成脚本 | `glm53-flash` / [`scripts/make_glm53_flash_notebook.py`](../scripts/make_glm53_flash_notebook.py) |
| 笔记本 | [`notebooks/glm5.3-flash-abliterated-g4-colab.ipynb`](../notebooks/glm5.3-flash-abliterated-g4-colab.ipynb)（Drive ID 存 `.secrets.json` 的 `glm53_drive_id`） |
| CLI 启动 | `python -X utf8 scripts/colab_cli.py start glm53-flash`（或 `launch glm53-flash`） |
| Magpie 模型 ID | `qwen-colab/GLM-5.3-Flash-Uncensored-GSQ-RCO-Q4` |

## 权重与组件

| 角色 | 仓库 @ revision | 文件 | 字节 | SHA-256 |
|---|---|---|---:|---|
| model（已合入 LoRA-v2） | `GCSA-AiLab/GLM-5.3-Flash-Uncensored-RCO-GSQ-GGUF` @ `1827843362350fbc87ee6b478d9dd09e33d9fb50` | `Q4/GLM-5.3-Flash-Uncensored-GSQ-RCO-Q4.gguf` | `137_073_273_280` | `49f3ccd0f62b622ff6f66bfe12c79b71787a360c22449594ded0b75c0cf4f3b4` |
| mmproj | `unsloth/GLM-5.3-Flash-GGUF` @ `a38483c8cd5df544f53d70fb281afe97369d5ab6` | `mmproj-BF16.gguf` | `1_164_010_176` | `82b7adb44411dbb298b1c689567804e5ed0790d80f308ed8b5b6b8ac0527d2ac` |
| lora-v2（备查，本包已合入无需再挂） | `GCSA-AiLab/GLM-5.3-Flash-Abliterated-LoRA-GGUF` @ `1d8439bff3ef52fe237a86e1c422dbff4834d825` | `GLM-5.3-Flash-Ablitered2-LoRA-v2.gguf` | `277_259_072` | `58b189c71ff5539bbb9cc93adbe651c9f942414f60967de1527b8d8333f527fa` |
| lora-v1（备查） | `GCSA-AiLab/GLM-5.3-Flash-Abliterated-LoRA-GGUF` @ `1d8439bff3ef52fe237a86e1c422dbff4834d825` | `GLM-5.3-Flash-Ablitered-LoRA-v1.gguf` | `287_876_448` | `5ec8ea1a12a683edfa28fc7a1887fe5bddfb3e7a34ae644c152649565cd42e73` |

## 运行与显存/内存混合布局

- **架构与张量账本**：GLM-5.3-Flash 为 320B 总参数、18B 激活参数的混合注意力（KDA 线性注意力 + 512 维 nope MLA + k-pool 稀疏 Indexer + mHC 4 流残差）MoE 架构（共 45 层：前 3 层稠密层 `blk.0..2`，后 42 层 `blk.3..44` 每层 288 路由专家选 8 + 1 共享专家）。
  - 非路由专家权重（词表嵌入、3 层稠密层、全部 42 层注意力/Indexer/共享专家、输出头）：共 `8.95 GiB`。
  - 42 层路由专家权重（`ffn_{gate,up,down}_exps.weight`）：共 `118.07 GiB`（`blk.3` 占 `3.50 GiB`，`blk.4..37` 每层 `2.90 GiB`，`blk.38` 占 `2.67 GiB`，`blk.39..44` 每层 `2.22 GiB`）。
- **单张 G4（95.6 GiB 显存 + 176 GiB DDR5 主机内存）切分**：
  - 启动参数附加 `--n-cpu-moe 19`：将前 19 层中 `blk.3..18` 共 16 个 MoE 层的路由专家（`47.00 GiB`）驻留主机 DDR5 锁页内存；
  - 后 26 个 MoE 层 `blk.19..44` 的路由专家（`71.07 GiB`）+ 全部非专家层（`8.95 GiB`）+ `mmproj-BF16.gguf`（`1.08 GiB`）= **`81.10 GiB`** 常驻 G4 显存，预留约 `14.5 GiB` 显存给 262k 上下文缓存与计算图缓冲。
  - `ram_reserve_gib = 65`：`127.66 GiB` 主文件直写 `/content/glm53-flash/Q4/` 本地盘并以 `-lm none --lazy-mode off` 直接分流加载入 GPU 显存与 CPU 锁页内存，不额外占用 `/dev/shm`；`mmproj-BF16.gguf` 落 `/dev/shm` 并在加载后自动删除副本回收给 Prompt Cache。
- **采样与聊天模板**：
  - 采样按 `zai-org/GLM-5.3-Flash` 官方 `generation_config.json`：`--temp 1.0 --top-p 0.95 --top-k 0 --min-p 0.0`。
  - 聊天模板：由 `glm5_next` 模板族将官方 `low / high / max` 三档扩展为 `low / medium / high / xhigh` 四档中文思考引导（`xhigh -> Reasoning Effort: Max`，`high/medium -> Reasoning Effort: High`，`low -> Reasoning Effort: Low`），并按模型卡建议默认启用 `clear_thinking = true`。
- **独立 LoRA 适配器兼容补丁**：若改用未合并底模（如 `unsloth/GLM-5.3-Flash-GGUF`）挂载 `GCSA-AiLab/GLM-5.3-Flash-Abliterated-LoRA-GGUF`（`role: 'lora'`），首格会自动将该 LoRA GGUF 头部误写的 `general.architecture = glm5next` 原地修补为 `glm5-next`（消除 1 字节对齐填充，保持张量偏移与文件大小不变），避免 `llama-adapter.cpp` 报 `model arch and LoRA arch mismatch`。
