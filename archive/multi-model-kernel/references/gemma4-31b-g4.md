# Gemma-4-31B-Ortenzya-Heretic-Q8_0 · Colab G4

运行 `python scripts/make_31b_gemma4_notebook.py` 生成并双写。支持 `--sanitized --output-dir DIR`，脱敏模式不写云端盘。

llama.cpp b12032 CUDA12.8，G4高内存。无PLE表，115G /dev/shm预留45GiB主机内存；8进程×4线程分块下载。加载时后台SHA-256验真，失败则终止服务并移除损坏文件。

全层CUDA0，单槽262144，prompt cache 上限 16–64GiB（加载后删内存盘里只给显存用的分片副本，回收内存的一半给缓存），Q8 KV，65536输出上限。Gemma A100 40GB回退65536上下文/ubatch512。采样按 Gemma 4 模型卡：temperature=1.0, top_p=0.95, top_k=64，min_p 显式设 0。不加惩罚项与 DRY。Gemma 的 low 档不关思考，所以四档都在 `<|think|>` 后加一句中文引导（low「越短越好」、medium「简短些」、high 只说「用中文思考」、xhigh「尽量深入」）。

MTP：draft-mtp / n_max3 / p_min0.50，真实接受率须GPU验证。

|角色|仓库 @ revision|文件|字节|SHA-256|
|---|---|---|---:|---|
|model|llmfan46/gemma-4-Ortenzya-The-Creative-Wordsmith-31B-it-uncensored-heretic-GGUF @ d7c96e816759b50ba7d0afe3a9e36deb7b096f16|gemma-4-Ortenzya-The-Creative-Wordsmith-31B-it-uncensored-heretic-Q8_0.gguf|32635675200|688170f6d6e03edc33ac86891eb3c37ec6cb1739f5db8625a84a1e4cdba17586|
|mmproj|llmfan46/gemma-4-Ortenzya-The-Creative-Wordsmith-31B-it-uncensored-heretic-GGUF @ d7c96e816759b50ba7d0afe3a9e36deb7b096f16|gemma-4-Ortenzya-The-Creative-Wordsmith-31B-it-uncensored-heretic-mmproj-BF16.gguf|1200726016|c040f82999134f7ba8dddf1c9e3451822c8ae9af51b60a01619df43a2a3a7128|
|draft|unsloth/gemma-4-31B-it-GGUF @ c1ac76e99d5513b141e8adde7288b85c3f9c32ec|MTP/mtp-gemma-4-31B-it-Q8_0.gguf|514687104|5ae8b0117bed601e8924c6305bd5b0585de361d51f0e77091bcb4252cf1f27de|

本地校验不代表GPU加载或速度已验收。启动首格后运行下一格记录health、模型名、中文推理、timing和usage。固定svc:qwen分时复用，同一时刻只开一份；用完点击首格停机按钮。

备选仓库（仅记录，不自动切换）：`zerofata/G4-MeroMero-v2-31B-GGUF @ 705d96bf79fb69ac7fb902e1149cc0ee868dda26`，gated=False。

- `G4-MeroMero-v2-31B-Q8_0.gguf`，32635675232 B，SHA-256 `94611336dad6bf459ce8fd2def1dfefe6321298d6cdf58ecd48b3c090d7bb01a`。

备选仓库（仅记录，不自动切换）：`kabachuha/G4-MeroMero-v2-31B-Heretic-ARA-LoRA-GGUF @ a1d7632c4f91f6b805e1f1e7cccbda567c919b64`，gated=False。

- `G4-MeroMero-v2-31B-Heretic-ARA-LoRA.gguf`，321144128 B，SHA-256 `4a2b486b536d90ad06222162c66c794520c2abe0dc1314e61e9ac0ceaf739340`。

备选仓库（仅记录，不自动切换）：`unsloth/gemma-4-31B-it-GGUF @ c1ac76e99d5513b141e8adde7288b85c3f9c32ec`，gated=False。

- `gemma-4-31B-it-Q8_0.gguf`，32635677632 B，SHA-256 `d5808e5874e660a85ab45b2da00c9e3b4a003621249a333772232d1a703e4d67`。
