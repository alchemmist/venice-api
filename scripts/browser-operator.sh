#!/bin/sh
# Keep this terminal open while operating the remote browser.
# This is operator access only; shutting it down does not stop the server browser.
set -eu
exec kubectl -n venice-api port-forward --address 127.0.0.1 deployment/venice-browser-candidate 5901:5900 9224:9224
