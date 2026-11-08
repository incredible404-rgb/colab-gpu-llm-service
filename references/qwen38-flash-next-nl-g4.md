# Swift-1.5-Flash-Next-Heretic-V2 IQ4_NL（综合 1 高精度版）on Colab G4

与 [`qwen38-flash-next-g4.md`](qwen38-flash-next-g4.md)（IQ4_XS-plus）是**同一份权重的另一档量化**：同作者、同 fused imatrix、同 Heretic V2。采样、模板、思考档位、MTP、上下文与 WebUI 配置全部继承 `qwen38-fn`（注册表里用 `**MODELS['qwen38-fn']` 派生），只换主权重，便于 A/B。

| 项目 | 值 |
|---|---|
| slug / 生成脚本 | `qwen38-fn-nl` / [`scripts/make_g4_fn_nl_notebook.py`](../scripts/make_g4_fn_nl_notebook.py) |
| 笔记本 | [`notebooks/qwen3.8-flash-next-nl-g4-colab.ipynb`](../notebooks/qwen3.8-flash-next-nl-g4-colab.ipynb)（Drive ID 存 `.secrets.json` 的 `g4nl_drive_id`） |
| Magpie 模型 ID | `qwen-colab/Swift-1.5-Qwen3.8-Flash-Next-IQ4_NL` |
| 主权重 | `jackjusko/Swift-1.5-Flash-Next-Heretic-ReverseEngineer` @ `bd3bca068a544d459cf42902bf21a0fac00ec853`，根目录 `Swift-1.5-Qwen3.8-Flash-Next-Heretic-V2-IQ4_NL-0000{1..8}-of-00008.gguf`，共 108.9 GB；大小与 SHA-256 见 `colab_kernel.py` 的 `_NL_SHARDS` |
| 视觉头 / MTP | 与 `qwen38-fn` 相同 |

## 与 IQ4_XS-plus 的差别

- 配方：热尾层（36–47）与前 5 层、15/20/23/30 层的 down 投影接近 8 bit，44–46 层 gate/up 5 bit；IQ4_XS-plus 的专家 gate/up 大多是 IQ3_S。
- 第 2 片两版都只含 `per_layer_token_embd`（N-gram 表，28.8 GB，IQ4_NL 类型），照旧常驻内存盘、运行期由 CPU 读。
- 显存部分 80.1 GB（IQ4_XS-plus 65.9 GB）。现版实测 73,587 MiB / 97,887 MiB，**推算** NL 约 87 GB、余约 10 GB，未实测。
- 下载 104.9 GiB：内存盘约 77 GiB，其余约 27 GiB 落本地盘（现版约 14 GiB），冷启动预计多 1–2 分钟。
- 参照 Unsloth 对原版 Flash-Next 的 KLD 表（UD-IQ4_XS 0.084 → UD-Q4_K_XL 0.047），预计平均 KLD 约 0.05–0.06；作者未公布本版实测。

## 实测（2026-11-06，G4，同一会话热切换对比）

| | IQ4_NL | 现版 IQ4_XS-plus |
|---|---|---|
| 显存已用 / 剩余 | 90,061 / 7,190 MiB | 76,477 / 20,774 MiB |
| 冷启动（CLI，含下载） | 210 s（下载 104.9 GiB 用 188 s；内存盘 71.5 GiB、本地盘 33.3 GiB） | 热切换 154 s |
| 3.9 万 token 长提示预填 | 4,295 tok/s（首 token 约 9.2 s） | 4,168 tok/s（约 9.5 s） |
| 长提示后解码 | 137.6 tok/s | 122.6 tok/s |
| 代码题解码（通过） | 192 tok/s | 191 tok/s |
| 约 800 字小说解码 | 100.0 tok/s | 99.5 tok/s |
| 3.9 万 token 中间藏一句的检索 | 答对 | 答对 |

- 速度两版基本相同：6B 激活下解码不卡在显存带宽，多出的 14 GB 权重没有拖慢。
- 创意写作的 MTP 接受率约 0.50–0.53（平均接受 2.2–2.3 个 token），代码类更高。
- 写作质量的盲比还没做，留给本人体感判断。

## 留存与退役

本人体感不如现版或更差，就从 `MODELS` 删掉本条目与 `make_g4_fn_nl_notebook.py`，Magpie 同步去掉该模型名。体感更好就可以把它定为默认，现版留作回退。
