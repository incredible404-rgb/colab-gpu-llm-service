# Gemma-4-31B-it-ARA-Heretic-Q8_0 · Colab G4

基于 [`mradermacher/gemma-4-31B-it-ARA-heretic-GGUF`](https://huggingface.co/mradermacher/gemma-4-31B-it-ARA-heretic-GGUF)（底模 `richardyoung/gemma-4-31B-it-ARA-heretic`）的 `Q8_0` 满血量化。由 `colab_kernel.py` 统一生成，支持 `colab_cli.py start gemma4-31b` 无头启动以及与 `qwen38-huihui-v2` 免释放热切换。

| 项目 | 值 |
|---|---|
| slug / 生成脚本 | `gemma4-31b` / [`scripts/make_31b_gemma4_notebook.py`](../scripts/make_31b_gemma4_notebook.py) |
| 笔记本 | [`notebooks/gemma4-31b-g4-colab.ipynb`](../notebooks/gemma4-31b-g4-colab.ipynb)（Drive ID 存 `.secrets.json` 的 `gemma31b_drive_id`） |
| CLI 启动 | `python -X utf8 scripts/colab_cli.py start gemma4-31b`（或 `launch gemma4-31b`） |
| Magpie 模型 ID | `<magpie_provider>/Gemma-4-31B-it-ARA-Heretic-Q8_0` |

## 权重与组件

| 角色 | 仓库 @ revision | 文件 | 字节 | SHA-256 |
|---|---|---|---:|---|
| model | `mradermacher/gemma-4-31B-it-ARA-heretic-GGUF` @ `1b49bbdcb3dfaa67cc4daced7da7d16552959f97` | `gemma-4-31B-it-ARA-heretic.Q8_0.gguf` | `32_635_677_600` | `0312793c4b98e8ef6dd0d8ae46eade91db6c8c29723b2c58741ef30ef9e849a8` |
| mmproj | `mradermacher/gemma-4-31B-it-ARA-heretic-GGUF` @ `1b49bbdcb3dfaa67cc4daced7da7d16552959f97` | `gemma-4-31B-it-ARA-heretic.mmproj-f16.gguf` | `1_198_957_152` | `fff3ae7b364154cb835241ee5a436941bee24e982fe270fe9084c0eba7206310` |
| draft | `unsloth/gemma-4-31B-it-GGUF` @ `c1ac76e99d5513b141e8adde7288b85c3f9c32ec` | `MTP/mtp-gemma-4-31B-it-BF16.gguf` | `954_843_264` | `8e6211a54648eca5af909bddf9331b447203b1a7febae6831491367afb61da7f` |

## 运行与模板补丁

- 无 PLE 表，`ram_reserve_gib = 30`，全层 `CUDA0`，单槽 `262144` 上下文，`Q8_0` KV 缓存，最大输出 `65536`。
- 采样按 Gemma 4 模型卡：`--temp 1.0 --top-p 0.95 --top-k 64 --min-p 0.0`，不加惩罚项与 DRY。
- 模板补丁：将 Google Gemma 4 Canonical Chat Template 里的 `enable_thinking | default(false)` 改为 `default(true)`，并在 `<|think|>\n` 后注入四档中文引导（`low`「越短越好」、`medium`「简短些」、`high`「用中文思考。」、`xhigh`「尽量深入」；默认 `high`）。
- MTP 投机解码：`--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.50`。
