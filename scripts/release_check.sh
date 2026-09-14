#!/usr/bin/env bash
set -euo pipefail

python scripts/release_static.py
python -m compileall -q packages/core/src apps/api/src scripts
python -m pytest -q
python -m pytest -q \
  --cov=quant_os_core \
  --cov=quant_os_api \
  --cov-report=term \
  --cov-fail-under=94
npm --prefix apps/web run check
npm --prefix apps/web run build

echo "RELEASE CHECK: PASS"
