## Summary

A CLI-created G4 assignment disappeared from the server after about 23 minutes while `llama-server` was actively serving sequential inference requests. A separate keep-alive loop received HTTP 200 every minute, including the last successful request one minute before the assignment disappeared.

This looks different from closed issue #14: there was no 80-second execution timeout and no keep-alive 403. The failure here is that the server-side assignment itself vanished while real GPU inference was in progress.

## Environment

- `google-colab-cli`: 0.7.5 (`uv tool install`)
- OS: Windows 11 10.0.26200
- Python: 3.14.6
- Auth: `--auth oauth2`
- Runtime: G4, Standard shape
- Account: paid Colab account; observed usage rate 10.05 CU/hour
- Test date: 2026-11-07 (timestamps below are UTC-05:00)

## Reproduction

1. Create a session with:

   ```bash
   colab new -s qwen38-fn --gpu G4
   ```

2. Through `colab exec`, start `llama-server` as a detached process in the runtime. The server loaded a GGUF model successfully and served OpenAI-compatible chat completions.

3. From the local machine, send sequential real chat-completion requests to that server through a Tailscale Service. Each request asked for 384 output tokens; the next request started as soon as the previous one returned. This kept the model actively generating throughout the test rather than merely checking `/health`.

4. Once per minute, refresh the assignment/proxy information from `list_assignments()` and send:

   ```http
   GET https://colab.research.google.com/tun/m/<endpoint>/keep-alive/
   X-Colab-Tunnel: Google
   Accept: application/json
   X-Colab-Client-Agent: colab-cli
   Authorization: <same user bearer token used by the CLI>
   ```

5. Poll `colab sessions` and `colab usage` to distinguish a local disconnect from an actual server-side assignment removal.

## Expected

The runtime remains assigned while the model is actively processing requests and the keep-alive endpoint continues returning success.

The README's "Activity-Based Liveness" wording also suggests that active work should keep the session alive.

## Actual

- Runtime assigned at approximately `2026-11-07T02:10:53-05:00`.
- Continuous-inference probe started at `2026-11-07T02:14:59-05:00`.
- **273 real inference requests completed**, producing **104,832 completion tokens**.
- **23 keep-alive requests completed; every one returned HTTP 200**.
- At `2026-11-07T02:33:29-05:00`, about **18m33s into continuous inference** and **22m37s after assignment**, `colab sessions` reported:

  ```text
  [colab] Pruned 1 stale local session(s).
  [colab] No active sessions found on server.
  ```

- At the same time, `colab usage` changed to:

  ```text
  Usage rate: 0.00/hr
  Active assignments: 0
  ```

- The next keep-alive cycle confirmed that this was not just stale local CLI state:

  ```text
  2026-11-07 02:31:53 keep_alive_ping n=22 result=http_200 endpoint=<redacted>
  2026-11-07 02:32:53 keep_alive_ping n=23 result=http_200 endpoint=<redacted>
  2026-11-07 02:33:53 keep_alive_stopped reason=assignment_gone pings=23 endpoint=<redacted>
  ```

The probe stopped at this point because the server no longer reported an assignment. No OAuth token, runtime-proxy token, endpoint, or Tailscale address is included here.

## Questions

1. Does backend liveness currently recognize only Jupyter kernel execution, rather than GPU/CPU work performed by a detached process such as `llama-server`?
2. Is `GET /tun/m/<endpoint>/keep-alive/` expected to extend CLI-created G4 assignments in 0.7.5? HTTP 200 appears not to do so here.
3. What is the supported way to keep a headless CLI-created runtime alive for long-running model serving or training?
4. If this is an intentional G4/headless-session lifetime limit rather than a bug, could the limitation be documented explicitly? A runtime limit of roughly 23 minutes makes long-running CLI/agent workflows unusable.

I stopped after this reproduction rather than adding more client-side retries, because the server removed the assignment despite both continuous real inference and successful keep-alive responses.
