#!/bin/bash
# Double-click to open Tape Digitizer. Closing this Terminal window quits it.
cd "$(dirname "$0")"
[ -d .venv ] || bash setup.sh
exec .venv/bin/python -m tapedigitizer app
