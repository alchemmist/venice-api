"""Dedicated persistent Google Chrome runtime; no cookie injection or stealth.

Initial Google login is interactive. Chrome owns session persistence. CDP is
loopback-only; deploy with a private operator display, never public CDP/VNC.
"""
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
from urllib.parse import urlsplit


def main():
    proxy = os.environ.get("PROXY_SERVER", "")
    parsed = urlsplit(proxy)
    if parsed.scheme not in {"http", "socks5"} or not parsed.hostname:
        raise SystemExit("Explicit PROXY_SERVER required; refusing direct egress")
    if parsed.username or parsed.password:
        raise SystemExit("Do not embed credentials in proxy URLs")
    profile = Path(os.environ.get("PROFILE_DIR", "/data/profile")).resolve()
    profile.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(profile, 0o700)
    lock = (profile.parent / (profile.name + ".runtime.lock")).open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Profile already owned by another browser runtime")
    chrome = shutil.which("google-chrome-stable") or shutil.which("google-chrome")
    if not chrome:
        raise SystemExit("Google Chrome is not installed")
    port = int(os.environ.get("CDP_PORT", "9224"))
    if not 1024 <= port <= 65535:
        raise SystemExit("CDP_PORT must be between 1024 and 65535")
    args = [chrome, "--user-data-dir=" + str(profile),
            "--profile-directory=Default", "--proxy-server=" + proxy,
            "--remote-debugging-address=127.0.0.1", "--remote-debugging-port=" + str(port),
            "--no-first-run", "--no-default-browser-check", "--disable-quic",
            "--window-size=1440,900", "https://venice.ai/chat/agent"]
    # No --no-sandbox, automation flags, profile imports, or password handling.
    process = subprocess.Popen(args)
    def stop(_signum, _frame):
        if process.poll() is None:
            process.terminate()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        return process.wait()
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
