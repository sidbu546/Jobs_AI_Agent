#!/bin/bash
set -e
cd "$(dirname "$0")"
exec .venv/bin/streamlit run app/main.py "$@"
