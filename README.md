# Venice browser bridge for Pi

An OpenAI Chat Completions facade over the **Venice web UI in a real Google
Chrome browser**. It does not use the paid Venice API. The public LiteLLM alias is
`venice-deepseek`; the bridge model ID is
`venice-agentic-deepseek-v4.1-flash-uncensored`.

This is a prompt-based adapter, not native Venice function calling. Pi executes
local tools; the bridge never executes their commands. Synthetic acceptance has
verified coding, saved-session replay, codemode, manual compaction and active
command cancellation. It has **not** certified unattended multi-hour operation.
The browser uses **Classic Chat**, not either Venice agent runtime. The previous
ad-hoc isolation was insufficient: a reproduced long audit caused its server-side
JSON/tool detector to report `Unrecognized` host tools even with empty tool lists.
Those errors were not local Pi results. The fix removes that agent loop entirely.

## Text-only browser transport

Prompts enter the real `/chat/classic` composer; replies come from the current
assistant message's exact raw content, before math/Markdown preprocessing. The regular browser request
uses `/api/inference/chat` on the expected first-party origin. No tokens are read
or exported, and no standalone HTTP inference client, official paid API or response
replay is used. Do not route to `/agents/venice` OR `/agents/ad-hoc`: empty tool
lists do not disable their server-side tool parser.

The replacement payload contains only current user text, explicit model, text-only
system instructions, request ID and fixed text-mode flags. Venice system prompt,
web search, scraping and X search are disabled. No browser history, media, native
tools or paid approval declarations are forwarded. Full Pi history remains inside
the host prompt. The UI's exact known `/think ` prefix may be removed before
forwarding; arbitrary prompt transformations are rejected. Exactly ONE matching
inference POST must pass per turn. Unknown POSTs, agent endpoints, wrong origin,
composer mismatch, repeats or missing interception fail closed with HTTP502 /
`upstream_agent_transport_error`, never native fallback. Routing is removed in
finally, including cancellation. Completion also requires the matching browser
requestfinished event; visual quiet alone cannot release a partial response.
Plain chat and compaction use the same Classic
transport with non-JSON instructions. Pi remains the ONLY host tool executor.

Classic's Markdown input is NOT a lossless protocol source: its math preprocessor
inserts a backslash before dollar signs, making valid shell-variable JSON invalid.
The extractor resolves the committed React tree to THIS assistant element, then
reads only its nearest assistant-message content/id/role. It does not inspect other
messages' contents or use network replay. Unknown/unmounted/oversized trees and
missing or non-assistant message shapes fail closed; no slash removal or repair.
This same raw source is used for plain replies and compaction summaries.

The private web-client schema and React source extraction remain maintenance
dependencies. This does not make an LLM infallible or certify unattended hours.
Direct use of Venice outside the bridge and host permissions are unchanged.

## Runtime and ownership

```text
Pi -> LiteLLM (virtual user key)
   -> bridge sidecar (private upstream key)
   -> pod-local CDP -> headful Google Chrome + persistent profile
   -> dedicated SOCKS egress -> Venice web UI
```

- `Dockerfile`: bridge only (`main.py`, `agent_protocol.py`, Python dependencies).
- `Dockerfile.browser`: separately released real Chrome/Xvfb/operator runtime;
  explicitly copies its three required `scripts/` files. Do not rebuild/upgrade
  the production Chrome image as a side effect of a bridge release.
- Canonical production manifests are in the sibling homelab repository:
  `charts/venice-api/`, `deploy/venice-api/values.yaml`, and
  `deploy/argocd/apps/venice-api.yaml`.
- Despite its historical name, Deployment `venice-browser-candidate` is production.
  Its Chrome uses `/data/profile` on PVC `venice-browser-profile`, not a profile
  in this checkout. One writer, Recreate, private profile directory0700. A local
  retained PVC is not backup or HA.
- Production bridge reads `API_KEY` from Secret `venice-api-env`. Egress and
  operator files are mounted from their separate native Secrets. LiteLLM's
  upstream credential is wired in homelab's `deploy/platform/litellm/values.yaml`
  through Secret `venice-bridge-upstream`. CI publishes an immutable image;
  charts and existing secret management, not the image, own secret wiring.
- Production connects to `http://127.0.0.1:9224` inside the SAME pod. CDP/VNC are
  privileged, loopback-only operator interfaces; never expose them publicly.
  There is no runtime dependency on Mac Chrome, Mac profiles or reverse tunnels.

Current operations/acceptance: `homelab/llm-wiki/venice-api.md`. The older
[initial infrastructure audit](docs/infrastructure-audit.md) is historical and
must not be used as the current deployment contract.

CI creates a dedicated runner-temp Docker auth config (directory0700/file0600)
from its existing Harbor secrets without printing them. BuildKit performs actual
registry authentication during push; a separate daemon login preflight is not
required. The config is removed with `always()`, is outside the build context,
and is never part of an image. Push, local image smoke and immutable-digest
GitOps deployment remain mandatory; registry failures never authorize deployment.

## Run and test

Python3.12 is used in CI. From an isolated virtual environment:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

For an authorized existing Chrome reachable in the bridge's network namespace,
set `API_KEY` using your private secret-management mechanism, then:

```sh
export CDP_URL=http://127.0.0.1:9224
export HOST=127.0.0.1
python main.py serve
```

This does NOT start Chrome or create its login. The endpoint must be reachable
from that process; container loopback does not refer to the host or your Mac.
Use the existing production chart/runbook rather than opening CDP to make it
reachable. Do not run a second Chrome on an in-use profile.

Pi uses `pi --provider litellm --model venice-deepseek` or `/model` in a new
session. It uses per-model `openai-completions`, not the provider's other API.
For usage through LiteLLM, Venice's `supportsUsageInStreaming` must be true:
LiteLLM only forwards the estimate when Pi requests `stream_options.include_usage`.
Usage is approximate, not native tokenizer or Venice-credit accounting.

## Environment contract

Unless explicitly listed as a browser-runtime setting, these configure `main.py`.

| Variable | Default | Meaning / applicable mode |
| --- | --- | --- |
| `API_KEY` | empty | Incoming bearer credential. **Empty disables authentication**; never expose an unauthenticated bridge. |
| `HOST` | `127.0.0.1` | Bind address. Chart sets `0.0.0.0` inside the pod; this is not authorization. |
| `PORT` | `8080` | API listening port. |
| `CDP_URL` | empty | Attach to an existing browser. Recommended production mode; bypasses standalone browser launch and cookie import. |
| `VENICE_BASE_URL` | `https://venice.ai` | Web UI origin, not paid API base URL. |
| `VENICE_AGENT_PATH` | `/chat/classic` | Required non-agent web route. |
| `MODEL_ID` | ID above | The only accepted bridge model ID. LiteLLM owns the public alias. |
| `AGENT_TEXT_MODEL_ID` | `deepseek-v4-1-flash-uncensored` | Explicit browser Classic text model; not the public bridge alias. Must match the intended UI model. |
| `AGENT_INFERENCE_ORIGIN` | `https://outerface.venice.ai` | Expected first-party web-client inference origin. Unknown origin/path/payload or repeated inference POST is blocked, never forwarded to the native agent. |
| `MODEL_MATCH` | exact DeepSeek V4.1 Flash Uncensored regex | UI model match; absent/unverified selector fails closed. |
| `AUTO_GOOGLE_LOGIN` | `true` | Reuse the already-authorized Google session for Venice SSO; never bypass password/MFA/consent. Google can revoke the session. |
| `NEW_CHAT_PER_REQUEST` | `true` | Start a new Venice chat for each API turn; full agent history is serialized into that turn. |
| `HEADLESS` | `true` | Standalone Playwright launch only. Does not alter attached Chrome. Production Chrome is separately headful under Xvfb. |
| `PROFILE_DIR` | `$XDG_DATA_HOME/venice-api/profile`, otherwise `~/.local/share/venice-api/profile` | Dedicated standalone profile, created/protected0700. Both image recipes explicitly use `/data/profile`; only the Chrome runtime owns the production profile/PVC. |
| `XDG_DATA_HOME` | `~/.local/share` | Base for the standalone default; empty/relative XDG values use the home-directory fallback. Existing `./venice_profile` is NOT migrated or deleted. |
| `PROXY_SERVER` | empty in standalone bridge | Standalone browser launch only; cannot change routing of an attached Chrome. In `scripts/browser_runtime.py`, a credential-free HTTP/SOCKS5 proxy is REQUIRED; production uses pod-local SOCKS. |
| `BROWSER_CHANNEL` | empty | Standalone Playwright channel; irrelevant with CDP. |
| `USER_AGENT` | empty | Optional standalone browser override; irrelevant with CDP. |
| `CHROME_PROFILE`, `CHROME_USER_DATA` | unset; legacy Chrome root fallback under `~/Library/Application Support/Google/Chrome` | Legacy opt-in existing-profile selector, not production. Prefer a dedicated `PROFILE_DIR`; do not use the owner's personal Chrome profile. |
| `COOKIES_FILE` | unset | **Legacy explicit standalone import only**. No home-directory cookie file is auto-read. Explicit missing file fails browser initialization; no cookie file is needed in CDP mode. No cookie secret is baked into either image. |
| `GEN_TIMEOUT` | `600` seconds | Generation deadline; error, not successful stop. |
| `START_TIMEOUT` | `90` seconds | Wait for generation to start. |
| `NAV_TIMEOUT` | `90` seconds | Browser navigation deadline. |
| `MODEL_SELECT_TIMEOUT` | `20` seconds | Pre-send model selector hydration wait; no blind retry of submitted prompts. |
| `STABLE_SEC` | `2` seconds | Settled-answer quiet period. |
| `MAX_PENDING_REQUESTS` | `4` | Active generation plus waiters. Only ONE browser generation runs. |
| `QUEUE_TIMEOUT` | `180` seconds | Maximum queue wait. Busy errors are retryable; tool execution is not retried by the bridge. |
| `DEBUG` | `false` | Screenshot diagnostics may contain private prompts and answers. Keep disabled normally. |
| `DEBUG_DIR` | `./debug` | Local diagnostic output. Prefer an external private directory; never commit or publish captures. |
| `CDP_PORT` | `9224` | Dedicated Chrome runtime only, loopback. |

The standalone Playwright path remains for development/legacy use. It is not a
replacement for the accepted server Chrome/SSO architecture. Sampling fields and
`max_tokens` are not precisely enforced through the web UI. The conservative
Pi32k/4k settings do not establish the real web model's limits.

## Agent protocol and history policy

[Detailed adapter contract](docs/pi-agent-protocol.md).

- Text function tools, one new call per step, schema-validated arguments;
  auto/none/required/named choices. New calls must use currently declared tools.
- Only a complete, valid assistant envelope is actionable. No regex extraction,
  guessing tool names from text, automatic repair or server-side shell runner.
- Assistant history can legally contain BOTH text and tool calls; `content:null`
  is not mandatory. Null assistant content requires calls. Tool IDs, function
  structure, argument JSON objects, call/result ordering and role placement are
  validated; malformed structure is rejected before browser submission.
- **Past schema-invalid attempts and their linked error results are retained.**
  Tools may also have changed/been removed since a past call. The adapter checks
  old arguments against the CURRENT schema when available and supplies safe
  `history_validation` annotations for mismatches/unavailable tools. This is not
  proof of the schema or execution outcome at the original time. It does not
  reinterpret error text, coerce values, discard results or execute old calls.
  New calls still undergo strict validation; a historical mismatch never grants
  execution permission.
- JSON duplicate keys at any nesting, NaN/Infinity and floating overflow such as
  `1e999` are rejected in new calls/history. Native nonfinite Python values in
  tool schemas/message extras are rejected too.
- Buffered SSE emits validated tool calls with `finish_reason=tool_calls`, or
  final content with `stop`, then `[DONE]`. Browser/model/parse errors are errors,
  not successful stops. Approximate usage is included for agent turns.

## Context budget errors

Agent prompts have a120000 UTF-8-byte adapter budget (including tool schemas and
history annotations); response envelopes have a65536-byte budget. These are
safety budgets, **not proven model token limits**. No messages/results are
silently truncated or removed.

Over-budget stream AND non-stream requests return HTTP400 with an OpenAI-style
`error.code=context_length_exceeded` BEFORE SSE/browser work. The error includes:

- `budget.limit_bytes`, `total_bytes`, `messages_bytes`, `non_message_bytes`,
  `fixed_prompt_bytes` (instructions/tools/choice with no history);
- message count and up to three largest messages' **index/role/UTF-8 size only**;
- guidance to compact in Pi or reduce tool output at its source while retaining
  call/result pairs; if tool-definition overhead alone exceeds the limit, guidance
  to reduce enabled tools instead.

No message text, argument values, file paths or tool-call IDs are echoed in budget
diagnostics. Pi can recognize the context-window error and attempt compaction;
this is not a guarantee that an arbitrarily large single file/tool output can be
made to fit. Limit command output/read ranges upstream. A separate queue-full
HTTP429 is not a context overflow. Approximate usage/compaction still need bounded
long-running verification.

## Build-context and credential hygiene

`.dockerignore` is a default-deny allowlist shared by BOTH image recipes. Only
the two Dockerfiles, requirements, two Python bridge modules and the three
browser runtime scripts enter the context. Git history, `.env*`, profile data,
operator helpers, cookies, debug artifacts and future unknown files stay out.
`.gitignore` is a separate Git safeguard and does not replace `.dockerignore`.

With Docker Buildx available:

```sh
bash scripts/check-build-context.sh
```

The check stages only allowed source files and synthetic secret/profile/debug
canaries, then uses real BuildKit matching plus a scratch-image context export.
It never stages your real browser profile or secrets for its audit. CI requires
exactly the eight allowed files before publishing, then imports the bridge
image from the SAME BuildKit result loaded locally (`--pull=never`) before GitOps
deployment of the published immutable digest. This avoids an extra registry
connection just to run the smoke test. Both recipes' COPY dependencies are also unit-tested.
Adding a runtime dependency requires updating the allowlist and tests deliberately.

Keep keys/cookies/profile archives and evidence in private operator custody,
not this repository. LiteLLM's existing prompt logging may store requests/responses;
this bridge does not promise all data stays on the client. No auth/session reset,
secret rotation, shared DNS/VPN change or broad cluster reconcile is part of a
bridge hardening release.

### Observable format recovery

A malformed agent reply gets at most ONE explicit regeneration, never JSON repair.
Pi sees a bridge notice with diagnostic ID. Only a newly validated envelope emits
a tool call; a second failure remains an error. Transport and source extraction
errors do not retry. Protected bounded rejected-output receipts live inside the
bridge at /tmp/venice-agent-rejections (0700/0600, latest8,64KiB each); the host prompt, auth headers, cookies and profiles are not separately captured.
Rejected model output can still echo sensitive user context; treat receipts as private. These ephemeral files disappear on pod
replacement. AGENT_REJECTION_DIR="" disables capture. Valid assistant prose is
not filtered, including false failure stories: use actual Pi events as evidence.
See docs/pi-agent-protocol.md for usage accounting and remaining reliability limits.
