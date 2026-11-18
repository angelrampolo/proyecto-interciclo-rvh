#!/bin/sh
cd "$(dirname "$0")"
exec python3 servir_dashboard.py "$@"
