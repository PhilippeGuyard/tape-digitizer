#!/bin/bash
# One-time setup: creates a private Python environment with what the digitizer needs.
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
chmod +x tape
echo "Setup done. Try: ./tape devices"
