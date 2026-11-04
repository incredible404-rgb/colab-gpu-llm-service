# colab-gpu-llm-service

Run a 100 GB-class GGUF model on a rented **Colab G4 (RTX PRO 6000 Blackwell, 96 GB)** as a repeatable, private, budget-aware service. One notebook cell boots the model, a fixed Tailscale address serves it, and one command tells you exactly what the session cost and proves the GPU was released.

The runtime is disposable and `/content` is wiped on every new session, so everything below is designed around fast, deterministic cold starts and around never paying for a GPU you are not using.

## What is in the box

| Piece | Problem it solves |
|---|---|
| DDR5 `tmpfs` RAM disk with 32-way direct writes | Colab's persistent disk caps writes at roughly 200-350 MiB/s; a 95 GB model would take minutes longer than it needs to |
| `MAP_POPULATE` zero-copy mapping of the PLE table | Removes a second memory copy and page faults during inference |
| In-place GPU LoRA baking | A LoRA is merged on the GPU at load time, so inference runs at base-model speed |
| `llama.cpp` MTP speculative decoding | Draft-head speculation tuned for code and tool calls |
| Tailscale Service (`svc:qwen`) | A fixed `qwen.<YOUR_TAILNET>.ts.net` address that survives runtime changes, with no login or manual approval |
| Playwriter zero-OAuth CU metering | Live balance, burn rate and active sessions without OAuth or API keys |
| Two-track CU bill and ghost-session kill switch | Reconciles what Colab billed against what the wall clock predicts, and releases sessions whose page was already closed |
| `.secrets.json` + `--sanitized` | Notebooks are generated from a private config; a sanitized build contains no real credential, tailnet or Drive ID |

## Architecture

```text
 Local machine (signed-in Edge)                       Colab runtime (G4 96 GB, ~180 GB RAM)
 ┌─────────────────────────────┐                      ┌───────────────────────────────────────────┐
 │ make_*_notebook.py          │  Playwriter / CLI    │ cell 1: one-click start                    │
 │   reads .secrets.json       │ ───────────────────▶ │  1. GPU check (>= 85 GB VRAM)              │
 │   writes .ipynb             │  push cells, run     │  2. /dev/shm + /var/colab tmpfs RAM disks  │
 │ colab_sync.py               │                      │  3. 8 procs x 4 threads -> os.pwrite       │
 │   /tun/m wrapper            │                      │  4. llama-server (MTP, MAP_POPULATE)       │
 │ colab_meter.py              │ ◀─── /tun/m ──────── │  5. tailscaled + Serve -> svc:qwen         │
 │   status / start / stop     │  balance, assignments│  6. stop button + idle auto-release        │
 └─────────────────────────────┘                      └───────────────────────────────────────────┘
            │                                                           ▲
            └──────────── http://qwen.<YOUR_TAILNET>.ts.net:443/ ───────┘
                          (tailnet only, WebUI + OpenAI-compatible /v1)
```

## Fast cold start

1. **Skip the persistent-disk write wall.** The notebook grows `/dev/shm` and `/var/colab` and symlinks them back to the runtime directory. Weights land in physical RAM, so there is no disk I/O on the hot path.
2. **32 concurrent streams.** A separate downloader runs 8 processes x 4 threads, each writing with `os.pwrite` at its own offset. Separate processes avoid the single-process GIL and TLS decryption bottleneck. Signed CDN links are resolved once and reused.
3. **`MAP_POPULATE` for the largest tensor.** The second shard is a single 26.8 GiB per-layer embedding table. It is written to `/dev/shm` first and `llama-server` maps it with `--lazy-mode off`, pre-populating the page tables. Inference then runs without page faults and without a second copy.
4. **One-cell console.** After startup the first cell shows a ready banner with the WebUI link and a stop button that releases the GPU.
5. **LoRA baked in place.** The 27B literary notebook merges its LoRA on the GPU, so there is no runtime slowdown.

## Speculative decoding and sampling

`llama.cpp` runs `--spec-type draft-mtp` with a matching MTP draft head. The G4 notebook is tuned for code and long-horizon agent work: single 262,144-token slot, low-temperature sampling, a DRY penalty that allows legitimate short code repeats but breaks failure loops, a 16 GiB prompt cache with slot persistence, and a 65,536-token output limit. Treat the exact numbers as a starting point and re-measure with your own prompts: run one cold request and two consecutive ones, and read `predicted_per_second` and the draft acceptance rate from the server log.

## Private access with Tailscale Service

The runtime joins your tailnet with a one-shot, ephemeral, `tag:colab` auth key minted from an OAuth client, then advertises `svc:qwen`. With `autoApprovers.services` in the ACL, no one has to log in or click Approve. The address stays `qwen.<YOUR_TAILNET>.ts.net` across runtimes. See [`references/tailscale-serve.md`](references/tailscale-serve.md) for the ACL snippet and the two classic failure modes (an interrupted cell kills the daemon; a missing `--statedir` breaks HTTPS).

The service is only visible inside your tailnet, so the WebUI works without pasting a key. If you ever expose it through Funnel or any public endpoint, add `--api-key-file` first.

## Zero-OAuth CU metering and ghost-session kill

Colab's own web app talks to `/tun/m/*`. The scripts call those endpoints from inside a signed-in `colab.research.google.com` tab, so no OAuth client or API key is needed.

```bash
python scripts/colab_meter.py status
# Active assignments: 0 | Burn rate: 0.00 CU/h | Balance: 988.85 CU | Eligible GPUs: H100, G4, A100, L4, T4

python scripts/colab_meter.py start -s g4 -g G4     # record opening balance and time
python scripts/colab_meter.py stop  -s g4           # kill, verify, print the two-track bill
python scripts/colab_meter.py stop  --all           # clear every active assignment

python scripts/colab_sync.py --status               # same read, no meter state needed
python scripts/colab_sync.py --release-all          # kill all and verify
```

**Tab hygiene.** With drive id `__ANY__`, an already open Colab tab is reused untouched (never reloaded or navigated). If none is open, a temporary tab is opened in the background and closed immediately after the call.

**Kill handshake.** For each active `endpoint`: `GET /tun/m/unassign/<endpoint>` returns a `token`; `POST` to the same URL with `X-Goog-Colab-Token: <token>` releases it. After 1.5 s the script re-queries `ccu-info` and `assignments`, up to 3 times, until `assignmentsCount == 0` and `consumptionRateHourly == 0`. This also clears ghost sessions whose notebook page was closed before `runtime.unassign()` could run.

**Two-track bill.**

```text
track 1 (account ledger delta)  max(0, start_balance - end_balance)
track 2 (wall clock)            duration_seconds / 3600 x rate_hourly
```

Track 1 is an account-wide balance delta: it lags a few minutes behind real consumption, includes every instance that ran in the window, and can even rise on a top-up or refund (it is clamped to 0 and a note is printed). Track 2 is per session. Showing both makes a stuck session obvious; with concurrent instances read track 2 as the per-session figure.

**Safe stop.** `colab_meter.py stop -s NAME` only releases the endpoints recorded for that session. An unknown name, or a session without bound endpoints, exits with 1 and releases nothing; use `--all` to force-release every active instance. A session recorded by `--run-cell1` while the VM was still allocating (rate 0, no endpoints) is backfilled with the real endpoint and burn rate by the next `status` or `stop`.

**Sync integration.** `make_*_notebook.py --sync-colab --run-cell1` records the opening balance once cell 1 starts. `--release-runtime` also calls `unassignCurrentVm`, sweeps leftover assignments, verifies `0 active / 0.00 CU/h` and prints the bill.

The `/tun/m` endpoints are undocumented and may change. Treat them as a convenience, and keep the Colab "Manage sessions" page as the ground truth if the two ever disagree.

## Quick start

Requirements: Python 3.10+, Node.js, a signed-in Edge with the Playwriter extension (or `@playwright/cli`), a Colab Pro/Pro+ plan with G4 access, and a Tailscale tailnet with an OAuth client.

```bash
git clone <this repo> && cd colab-gpu-llm-service-oss
cp .secrets.example.json .secrets.json         # fill in your own values; the file is git-ignored
python scripts/make_g4_fn_notebook.py          # writes notebooks/qwen3.8-flash-next-g4-colab.ipynb
python scripts/make_g4_fn_notebook.py --sync-colab --run-cell1
```

`.secrets.json` fields:

| Field | Meaning |
|---|---|
| `api_key` | Model API key written to the runtime |
| `ts_client_id`, `ts_client_secret` | Tailscale OAuth client used to mint ephemeral auth keys |
| `tailnet_fqdn` | Service address, `qwen.<YOUR_TAILNET>.ts.net` |
| `lit_drive_id`, `g4_drive_id` | Drive file IDs of the two notebooks (needed only for `--sync-colab`) |

Each field can also come from an environment variable: `LLAMA_API_KEY`, `TS_CLIENT_ID`, `TS_CLIENT_SECRET`, `TAILNET_FQDN`, `LIT_DRIVE_ID`, `G4_DRIVE_ID`. Resolution order is `.secrets.json`, then environment, then a placeholder.

The checked-in notebooks under `notebooks/` are **sanitized builds** and contain placeholders only. Generate your own with the commands above.

### Sanitized builds

```bash
python scripts/make_g4_fn_notebook.py --sanitized --output-dir dist/
```

`--sanitized` fills every sensitive field with a placeholder and ignores `.secrets.json` and the environment. It refuses to combine with `--sync-colab`, so a placeholder notebook can never be pushed to Colab.

## Repository layout

```text
SKILL.md                      agent-facing runbook
scripts/
  make_27b_lit_notebook.py    literary 27B notebook generator
  make_g4_fn_notebook.py      G4 code/agent notebook generator
  secrets_config.py           .secrets.json / env / placeholder loader
  colab_sync.py               push cells to Colab, /tun/m wrapper, --status / --release-all
  colab_meter.py              status / start / stop CLI with two-track bill
  colab_pw.mjs                Playwriter backend (supports the __ANY__ tab mode)
  parallel_download.py        multi-process downloader used by the notebooks
  webui_console.py            first-cell console and stop button
  stop-runtime.py             release helper
  update-pi-provider.ps1      register the endpoint in a local pi provider list
references/                   model notes, CU budget, Tailscale and remote-access guides
notebooks/                    sanitized generated notebooks
.secrets.example.json         config template
```

## Security notes

- `.secrets.json` and `.colab-meter-state.json` are listed in `.gitignore`. Verify with `git status` before the first push.
- Generated notebooks embed the API key and Tailscale OAuth secret, because the runtime needs them. Never commit or share a notebook built without `--sanitized`.
- Use an OAuth client scoped to auth-key creation with `tag:colab` only, and rotate it if a non-sanitized notebook ever leaves your machine.
- A tailnet-only service relies on WireGuard identity for isolation. Public exposure needs an application-layer API key.

## License

MIT. See [`LICENSE`](LICENSE). Model weights keep their own licenses; check each model card before use.
