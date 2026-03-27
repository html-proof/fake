#!/usr/bin/env sh
set -eu

PORT_VALUE="${PORT:-5000}"

exec gunicorn \
  --workers 1 \
  --timeout 120 \
  --preload \
  --max-requests 50 \
  --max-requests-jitter 10 \
  --bind "0.0.0.0:${PORT_VALUE}" \
  backend.app:app
