# 冻结的多模型内核

2026-11-06 起，内核只为 3.8fn（`qwen38-fn` / `qwen38-fn-nl`）生成笔记本。这里保存移出前的整套文件：

- `scripts/colab_kernel.py`：移出前的统一内核，登记 `qwen38-fn`、`qwen38-fn-nl`、`qwen35-122b`、`nemotron-120b`、`gemma4-31b` 五个模型和 qwen35 / nemotron / gemma4 模板族。
- `scripts/make_*_notebook.py`、`scripts/secrets_config.py`：对应的生成入口和带三个 Drive ID 字段的配置加载器。
- `references/*.md`：三个模型的参数、文件哈希与启动说明。

三个模型最后一版笔记本仍在 `../../notebooks/` 和 Drive 的 `Colab Notebooks/`，可以在浏览器里照常跑，但不再更新。

恢复：把这里的 `scripts/` 与 `references/` 文件复制回仓库对应位置（会覆盖精简后的 `colab_kernel.py` 与 `secrets_config.py`），`.secrets.json` 里三个 Drive ID 一直保留着。
