# Nemotron-3.5-Super-VL-120B-Heretic · Colab G4

运行 `python scripts/make_120b_nemotron_notebook.py` 生成并双写。支持 `--sanitized --output-dir DIR`，脱敏模式不写云端盘。

llama.cpp b12032 CUDA12.8，G4高内存。无PLE表，115G /dev/shm预留45GiB主机内存；8进程×4线程分块下载。加载时后台SHA-256验真，失败则终止服务并移除损坏文件。

全层CUDA0，单槽262144，prompt cache 上限 16–64GiB（加载后删内存盘里只给显存用的分片副本，回收内存的一半给缓存），Q8 KV，65536输出上限。Gemma A100 40GB回退65536上下文/ubatch512。采样按 Nemotron 3.5 Super 模型卡（文本）：temperature=1.0, top_p=0.95；模型卡没给 top_k/min_p，显式设 top_k=0、min_p=0 关掉，免得落到 llama.cpp 默认 40/0.05。不加惩罚项与 DRY。low 关思考（`<think></think>`）；medium/high/xhigh 用 `<think>
` 起手、不预填，system 只加一句「用中文思考」。

MTP：draft-mtp / n_max3 / p_min0.50，真实接受率须GPU验证。

|角色|仓库 @ revision|文件|字节|SHA-256|
|---|---|---|---:|---|
|model|spiritfather/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-heretic-ara-GGUF @ 3608d16a78f65d04e28a9d2003061df7cc0f622d|Q4_0/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-heretic-ara.Q4_0-00001-of-00002.gguf|44443244992|10641657ae2344d997b295f9e667b49122a296eed204681190d915f1edda120c|
|shard|spiritfather/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-heretic-ara-GGUF @ 3608d16a78f65d04e28a9d2003061df7cc0f622d|Q4_0/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-heretic-ara.Q4_0-00002-of-00002.gguf|27360507712|63f849a3bd573dab01ee0814abc9814455790c5d4ca72823ce44bfdea3c4b3c5|
|mmproj|spiritfather/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-heretic-ara-GGUF @ 3608d16a78f65d04e28a9d2003061df7cc0f622d|mmproj-NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-heretic-ara-Q8_0.gguf|966925728|5c375aba723b64e5b328521be640f8945fde619bc8152d6bf8fc7b9f235be0d4|
|draft|spiritfather/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-heretic-ara-GGUF @ 3608d16a78f65d04e28a9d2003061df7cc0f622d|mtp-NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-heretic-ara-Q4_0.gguf|2270017376|6e115ff58b530741d7a9f6768ef9a8ac780eb3dcb372a13565019c7df19d964b|

本地校验不代表GPU加载或速度已验收。启动首格后运行下一格记录health、模型名、中文推理、timing和usage。固定svc:qwen分时复用，同一时刻只开一份；用完点击首格停机按钮。

备选仓库（仅记录，不自动切换）：`unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF @ 75a5829055f69a0359085ae6bbd0a81b60bc05c6`，gated=False。

- `LiLiCorr/lilicorr-NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4.gguf`，1290795104 B，SHA-256 `d5c0c5f22f8889ae0c0e068ea775cb641cd944a571aa65b00975cbdcdf96c806`。
- `NVFP4/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-00001-of-00003.gguf`，7874272 B，SHA-256 `95109d33bc080d3b705d083be31ab18610c619b4990a5c5a72e0bd2e85766b9d`。
- `NVFP4/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-00002-of-00003.gguf`，47786634752 B，SHA-256 `5f67e480e37a48832c9a868c1b4cc4b42530702fe08d6317017cf61dc95e7ce1`。
- `NVFP4/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-00003-of-00003.gguf`，29891544960 B，SHA-256 `24752e37634e07ebbca5baea706fac9b60b03ff818ee8b6febc74c6d87fe4ef9`。
- `mmproj-BF16.gguf`，1733695840 B，SHA-256 `9c09381a5744d0ea9a30048c94922f6f25ee0d63842de9c45779eca4fb5924d9`。
