#!/bin/bash
# Run the digitizer, e.g. ./tape record 42 A
cd "$(dirname "$0")"
exec .venv/bin/python -m tapedigitizer "$@"
