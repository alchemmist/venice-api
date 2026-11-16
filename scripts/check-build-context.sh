#!/usr/bin/env bash
set -euo pipefail
umask 077
repo=$(cd "$(dirname "$0")/.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
python3 - "$repo" "$work/input" <<'PY'
from pathlib import Path
import shutil
import sys

repo, staged = map(Path, sys.argv[1:])
allowed = [
    '.dockerignore', 'Dockerfile', 'Dockerfile.browser', 'requirements.txt',
    'main.py', 'agent_protocol.py', 'scripts/browser-entrypoint.sh',
    'scripts/browser_runtime.py', 'scripts/browser_probe.py',
]
for name in allowed:
    target = staged / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repo / name, target)
for name in [
    '.env', '.env.production', '.git/config', '.venv/private',
    'venice_profile/Default/Cookies', 'debug/prompt.txt', '.debug/screen.png',
    'scripts/private.pem', 'scripts/nested/private.key', 'scripts/browser-operator.sh',
    'new-directory/credentials.json', 'cookies.json', 'operator-password', 'README.md',
]:
    target = staged / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('synthetic-not-a-secret\n')
PY
docker buildx build --progress=plain --no-cache --file - \
  --output "type=local,dest=$work/export" "$work/input" <<'DOCKERFILE'
FROM scratch
COPY . /context/
DOCKERFILE
python3 - "$work/export/context" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
expected = {
    'Dockerfile', 'Dockerfile.browser', 'requirements.txt', 'main.py',
    'agent_protocol.py', 'scripts/browser-entrypoint.sh',
    'scripts/browser_runtime.py', 'scripts/browser_probe.py',
}
actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() or p.is_symlink()}
if actual != expected:
    raise SystemExit('Build context allowlist mismatch; refusing to publish')
print('BuildKit context allowlist verified: exactly 8 runtime/build files, no private artifacts')
PY
