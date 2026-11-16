# Mac-independent browser migration

## Fixed requirement

Use Venice web UI through a browser, not the official API. Neither browser execution nor its egress may depend on Mac. Existing production bridge remains intact until independent candidate acceptance.

## Placement and observations

socrat-5 has substantial available memory; its owner user session manager is running. Existing application image has Xvfb and Playwright Chromium, but no installed Google Chrome on the node PATH. socrat-2 has only about 406 MiB available memory in the inspection snapshot: do NOT put Chrome there. TCP from socrat-5 to socrat-2:2222 works over both public and mesh addresses. socrat-5 currently has no private SSH key in ~/.ssh; never copy the owner's Mac private key there.

## Proposed candidate

- Chrome and Xvfb run on socrat-5, isolated from existing CDP and profile.
- App-scoped egress to socrat-2. Prefer dedicated, restricted SSH machine identity and loopback SOCKS listener using native OpenSSH (no additional server proxy exposed). Server key must be pinned; no StrictHostKeyChecking=no. Decide persistence/ownership in homelab source before deployment.
- SOCKS listener and browser share a network namespace, explicitly configured in Playwright. Kill/fail closed browser upstream if egress unavailable; verify this behavior, do not infer it from config.
- Dedicated persistent Linux profile, one writer, Recreate. No Mac profile export and no Google password in automation.
- Dedicated Chrome image installed from Google distribution with sandbox/dependency validation; Xvfb headful. No stealth modifications added to candidate.
- Private interactive operator UI for initial login and session repair. No public VNC/CDP. Mac may be an optional administration client, not an execution dependency.

## Ordered gates

1. Prepare image and candidate configuration offline, without switching production.
2. Establish a least-privilege machine transport and verify socrat-5 -> VPS -> Venice TLS with Mac proxy explicitly absent. Verify egress public IP and transport failure behavior.
3. Start candidate Chrome/Xvfb with fresh persistent profile and private operator access.
4. Operator logs in using the ordinary web flow. If Google or Venice refuses automated environment, capture the failure without bypassing challenges.
5. Verify exact model and multiple responses. Validate that a page load alone is not counted as inference success.
6. Restart candidate browser and candidate pod; verify profile persistence and repeat inference.
7. Fix response extraction/model validation and contract tests; no false OpenAI compatibility promises.
8. GitOps handoff only after accepted candidate. Verify operation without either Mac reverse tunnel. Preserve rollback until functional checks pass.

## Remaining unresolved items

Machine egress credential provisioning and its server-side restrictions; browser sandbox acceptance in this runtime; private interactive access implementation; actual Linux Google/Venice login and Cloudflare behavior; profile storage selection and retention. None of these is claimed implemented by this document.
