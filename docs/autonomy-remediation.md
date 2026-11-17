# Autonomy remediation acceptance

The browser changes have 122 passing unit tests at the independent review checkpoint. A real local Uvicorn/TCP disconnect test, with the browser mocked, demonstrated the original nonstream slot leak and verified cancellation, one stop click, pending zero and an unlocked slot after correction. This does not prove that an unresponsive remote Chrome can acknowledge stop.

The last-window lifecycle mechanism was independently reproduced on native amd64 with the exact production browser image, `venice-browser@sha256:10f6a5c13bfbaee22ae1648d34f454a9537e3f4da5004f7218b2035214f9708b`, Google Chrome `157.0.8103.78`, sandbox enabled and a fresh temporary profile. Closing the only page before creating its replacement exited Chrome with code 0 and raised `TargetClosedError`. Opening the replacement first preserved the process and CDP connection; the replacement evaluated JavaScript successfully.

The probe ran in a dedicated deny-all namespace without PVCs, mounted secrets, service account token, Service or Ingress. The existing registry credential was copied privately for kubelet image-pull authorization only, never mounted in the container. The namespace and its copied pull secret were deleted after saving private receipts. Production container starts and restart counts remained unchanged: bridge 0, Chrome 4, egress 0.

This reproduces a physical last-window failure mechanism; it does not establish the cause of every historical production restart. Earlier exits cannot be reconstructed from the latest termination record. Do not call a Completed/0 exit an OOM.

The changes preserve the persistent CDP context and never create an incognito fallback, restart remote Chrome, replay an inference transport failure, or execute model tools in the bridge. Weak lifecycle tracking avoids retaining discarded transport identities. The extraction deadline covers the whole asynchronous call, not only element attachment.

A valid artifact, clean agent completion, format error, provider error, limits stop and controller stop are independent outcomes. Passing this suite, a successful native lifecycle probe, or Argo Healthy does not establish clean multi-hour autonomous operation. Integration, immutable CI/GitOps release and repository-agent runtime acceptance remain separate gates.

Browser RPCs for page creation, CDP connection, route registration and cleanup now have outer deadlines. Prepare and generation phases also have finite budgets. Cleanup timeouts do not replace the primary error. A stop click acknowledgement is not proof that a disconnected remote model physically stopped. Local driver cleanup runs on both CDP connection failure and cancellation.
