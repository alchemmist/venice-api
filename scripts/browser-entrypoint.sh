#!/bin/sh
set -eu
export DISPLAY=:99
Xvfb "$DISPLAY" -screen 0 1440x900x24 -nolisten tcp &
XVFB_PID=$!
cleanup() { kill "$XVFB_PID" "${VNC_PID:-}" 2>/dev/null || true; }
trap cleanup EXIT
# VNC requires a secret even though it is only reachable via pod-local loopback.
i=0
until [ -S /tmp/.X11-unix/X99 ]; do
  i=$((i + 1)); [ "$i" -lt 50 ] || exit 1
  sleep 0.1
done
x11vnc -display "$DISPLAY" -localhost -rfbport 5900 -rfbauth /operator/vnc-password -forever -shared -quiet &
VNC_PID=$!
python /app/browser_runtime.py &
BROWSER_PID=$!
trap 'kill "$BROWSER_PID" 2>/dev/null || true' TERM INT
wait "$BROWSER_PID"
