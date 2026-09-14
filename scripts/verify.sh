#!/usr/bin/env bash
set -euo pipefail
ruff check .
mypy packages/core/src apps/api/src
pytest -q
