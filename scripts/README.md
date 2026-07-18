# scripts/ — extraction + harvest tooling

| Script | Job |
|---|---|
| `cdp_eval.py` | Evaluate JS in a Solace Browser tab (CDP :9888, raw RFC6455 WS, no deps). `python3 cdp_eval.py <tab_id> <expr.js>` |
| `fetch_convo.js` | Page-context fetch of full ChatGPT thread via `/api/auth/session` accessToken + `/backend-api/conversation/<id>` (DOM only renders ~5 msgs) |
| `driver.py` | Autonomous "Next"-prompter: sends Next into the ChatGPT paper thread, polls for finished assistant turns, harvests `:::writing` paper blocks into `canon/extracted/paper-NN-extracted.md` |

Recipe provenance: chatgpt_extraction_via_backend_api memory (2026-07-09) + this session (2026-07-18).
Requires the Solace Browser running with `--remote-debugging-port=9888` and a logged-in chatgpt.com session.
