# Contributing / 参与贡献

[English](#english) | [中文](#中文)

## English

Issues and pull requests are welcome. This is a small project with one maintainer, so a reply can take a few days.

### Before a large change

Open an issue first and say what you want to change. A new model, a new host platform or a change to the runners format that Colab-yoga reads is easier to agree on before the code exists.

### What belongs here

This repository is the general tool: deploying and running GGUF models on a paid Colab GPU. One person's use case stays out of it: personal prompts and presets, tools built for one kind of work, folders and account ids of one machine, debug logs. Keep those in your own private overlay and connect them through the settings in `.secrets.json` or environment variables (`webui_system_prompt`, `LOCAL_MCP_CWD`, `magpie_provider`, ...). Colab-yoga is a separate, optional project: the only link is the runners file this repository exports. `scripts/test_generic_repo.py` fails when a known personal marker comes back.

### Set up and test

Python 3.10+ and `pip install requests`; everything else is standard library.

```bash
git clone https://github.com/incredible404-rgb/colab-gpu-llm-service.git
cd colab-gpu-llm-service/scripts
python -X utf8 -m unittest discover -p "test_*.py"
```

The tests are offline: they must not call Google, Colab, Hugging Face or Tailscale, and they must pass on both Windows and macOS. A test that only makes sense on one of them is skipped on the other with `unittest.skipIf`.

### Pull requests

- One topic per pull request, branched from `main`.
- CI runs the tests on Windows and macOS. A pull request is merged when it is green and reviewed.
- Add or update tests for behavior you change.
- When behavior changes, update `README.md`, `README.en.md`, `SKILL.md` and `references/` in the same pull request.
- Commit subjects follow `feat: ...`, `fix: ...`, `docs: ...`, `test: ...`.
- Leave "Allow edits by maintainers" on, so small fixes can be pushed to your branch.

### Keys

`.secrets.json`, API keys, Tailscale OAuth secrets and private notebooks (`notebooks/private/`) never go into a commit, an issue or a log you paste. Notebooks under `notebooks/` are the sanitized builds (`--sanitized`).

### Security

Report vulnerabilities privately through [Security advisories](https://github.com/incredible404-rgb/colab-gpu-llm-service/security/advisories/new), not in a public issue.

### License

Contributions are licensed under the license in [LICENSE](LICENSE).

## 中文

欢迎提 issue 和 pull request。项目只有一位维护者，回复可能要几天。

### 改动较大时

先开 issue 说明想改什么。新增模型、新增本机平台、改 Colab-yoga 读取的 runners 格式，这类事在写代码之前对齐更省事。

### 什么该进这个仓库

本仓库是通用工具：在付费 Colab GPU 上部署和运行 GGUF 模型。某一个人的用法不进来：个人提示词和预设、为某一类工作特制的工具、某台机器的目录和账号 id、调试日志。这些放在你自己的私有目录或仓库里，通过 `.secrets.json` 或环境变量里的设置项接入（`webui_system_prompt`、`LOCAL_MCP_CWD`、`magpie_provider` 等）。Colab-yoga 是独立的可选项目，两边唯一的衔接是本仓库导出的 runners 文件。已知的个人化标记再出现时，`scripts/test_generic_repo.py` 会失败。

### 环境与测试

Python 3.10+，`pip install requests`，其余都是标准库。

```bash
git clone https://github.com/incredible404-rgb/colab-gpu-llm-service.git
cd colab-gpu-llm-service/scripts
python -X utf8 -m unittest discover -p "test_*.py"
```

测试是离线的：不得访问 Google、Colab、Hugging Face 或 Tailscale，并且要在 Windows 和 macOS 上都通过。只在其中一个系统上有意义的测试，在另一个上用 `unittest.skipIf` 跳过。

### Pull request

- 一个 PR 只做一件事，从 `main` 拉分支。
- CI 会在 Windows 和 macOS 上跑测试，通过并审过之后合并。
- 行为有变时补上或更新测试。
- 行为有变时，在同一个 PR 里更新 `README.md`、`README.en.md`、`SKILL.md` 和 `references/`。
- 提交标题用 `feat: ...`、`fix: ...`、`docs: ...`、`test: ...`。
- 保持勾选 “Allow edits by maintainers”，方便维护者直接往你的分支补小修改。

### 密钥

`.secrets.json`、API 密钥、Tailscale OAuth 密钥和私有 notebook（`notebooks/private/`）不能出现在提交、issue 或粘贴的日志里。`notebooks/` 下的是脱敏版（`--sanitized` 生成）。

### 安全问题

请通过 [Security advisories](https://github.com/incredible404-rgb/colab-gpu-llm-service/security/advisories/new) 私下报告，不要开公开 issue。

### 许可

贡献内容按 [LICENSE](LICENSE) 中的许可授权。
