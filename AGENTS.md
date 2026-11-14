# Notes for coding agents

- This repository is the general tool. Read "What belongs here" in [CONTRIBUTING.md](CONTRIBUTING.md) before adding anything: personal prompts and presets, tools built for one kind of work, machine paths, account ids and debug logs go to the owner's private overlay, connected through settings (`webui_system_prompt`, `LOCAL_MCP_CWD`, `magpie_provider`, ...). When a request mixes the two, split it: the general mechanism here, the personal content there.
- Colab-yoga is a separate project. The only contract between the two is the runners file this repository exports; change it in both repositories or not at all.
- Check before every commit: `cd scripts && python -X utf8 -m unittest discover -p "test_*.py"`. It must pass on Windows and macOS; tests stay offline.
- Changes reach `main` through a pull request with green CI. Secrets (`.secrets.json`, private notebooks) are never committed.
