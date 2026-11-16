# Venice bridge: infrastructure audit

## Scope

Read-only runtime inspection; no deployment, restart, credential creation or paid inference. Consulted homelab AGENTS.md, llm-wiki/context.md, README.md, network-access.md, ssh-user.md, docs/mac-private-ssh.md and vpn-mesh-runtime-20261110.md. Historical acceptance is not current health evidence.

## Observed runtime

- Deployment venice-api, one pod on socrat-5, hostNetwork, Recreate, immutable digest sha256:b0c9c6d9373f6e8a7a8c845bebba9c26930323525d200715fab23a565177646a.
- CDP_URL points to node loopback 9223. Browser reports Chrome/157.0.8103.51.
- Mac ssh PID 93108 provides reverse forwarding node 9223 -> Mac 9223.
- PROXY_SERVER points to node loopback 8899. Mac ssh PID 73886 provides reverse forwarding node 8899 -> Mac 8899.
- Mac Python PID 73861 runs ~/venice-auth/connect-proxy.py; network call uses socket.create_connection to the requested destination. This proxy is not evidence of a dedicated VPS egress. Effective Mac routing remains outside this audit.
- Consequently BOTH current browser access and the configured HTTP proxy depend on Mac and SSH sessions.
- Direct Python HTTPS from the pod to venice.ai and api.venice.ai timed out during TLS handshake (12 s timeout).
- Node socrat-5 via its loopback HTTP proxy: venice.ai 200 (~0.338 s TLS), public API model catalog 200 (~0.344 s TLS).
- Independent socrat-2 direct HTTPS: venice.ai 200 (~0.117 s TLS), API catalog 200 (~0.085 s TLS). No Mac proxy configured in these curl calls.
- socrat-5 owner account has no passwordless sudo. Google Chrome, Chromium and Xvfb were not found on its current PATH.
- Public bridge GET /v1/models without bearer returns 401.
- Profile volume is emptyDir; readiness probes /, not upstream readiness.

## Official API alternative

Live public GET https://api.venice.ai/api/v1/models returned 142 entries, including EXACT requested model name:

- id: deepseek-v4-1-flash-uncensored
- name: DeepSeek V4.1 Flash Uncensored
- offline: false; privacy: private; uncensored: true
- listed input USD 0.33 / output USD 1.32 (pricing units must be verified against pricing documentation before budgeting)
- also deepseek-v4-1-flash-1017-uncensored

Catalog availability is not proof of account entitlement, credits, authenticated inference, or equality with the web Agentic Chat orchestration. No API key created or used, no billing changed. Official API is a viable candidate, not a completed migration.

## Code findings

- CDP launch returns before applying PROXY_SERVER; proxy setting cannot change the attached Chrome routing.
- Model selection fails open if selector missing. Requested model is not validated and can be echoed falsely in response.
- _clean_answer removes only first 14 prompt words and strips potentially legitimate answer lines; whole-transcript slicing is not reliable message extraction.
- Streaming assumes append-only DOM content.
- temperature/top_p/max_tokens silently ignored; message roles flattened into plain text; usage estimated, generation timeout reported as normal stop.
- No bounded request queue. Existing asyncio lock serializes a single shared page.

## Decision and next gates

1. Prefer official API if separate API economics are acceptable: scoped key supplied through secret management, bounded authenticated smoke test, thin authenticated proxy on VPS or app-scoped tunnel from socrat-5, strict model aliases and streaming passthrough. Do not move workload blindly onto a control-plane/VPN node; assess placement/resources first.
2. If existing web subscription is a hard requirement: separate Linux headful Chrome proof with dedicated persistent profile, interactive login, independently proven app-scoped VPS egress. Do not touch current Mac profile or claim automated Google MFA recovery. Cloudflare acceptance remains unproven.
3. Only after candidate acceptance update code/chart through GitOps. No global VPN/DNS/routes/firewall changes, no shared relay restarts, no public CDP or open HTTP proxy.
4. Acceptance must include Mac-free dependencies, exact model, long prompt fidelity, stream/non-stream, session/browser/pod restart, unavailable upstream, unauthorized client and bounded concurrency.

This file records an inspection snapshot, not deployment acceptance. PID values are observational, not instructions to terminate processes.
