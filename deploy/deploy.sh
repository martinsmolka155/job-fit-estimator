#!/usr/bin/env bash
# Update an already-provisioned Job Fit Estimator on prod (jobfit.lilithai.dev).
# Git-based, mirrors the hell-dashboard deploy model. Run ON the server:
#   ssh lilith@89.167.88.19 'cd /opt/jobfit && bash deploy/deploy.sh'
#
# First-time provisioning (clone, .env, ISPV data, certbot, nginx, systemd) is
# deploy/setup-prod.sh — run that once before this.
set -euo pipefail

cd /opt/jobfit

echo "[deploy] fetching origin/main"
git fetch origin
echo "[deploy] resetting tracked files to origin/main"
git reset --hard origin/main

echo "[deploy] installing deps"
python3 -m venv venv
./venv/bin/pip install -q -U pip
./venv/bin/pip install -q \
  "pymupdf4llm>=0.0.17" "python-docx>=1.1.2" "openai>=1.55.0" \
  "pydantic>=2.0" "pydantic-settings>=2.0" "rich>=13.0" \
  "reportlab>=4.2.0" "structlog>=24.0" "openpyxl>=3.1.0" \
  "httpx>=0.27" "fastapi>=0.115" "python-multipart>=0.0.9" "uvicorn>=0.32"

# .env and data/ispv_2025.xlsx are gitignored runtime state — they must already
# exist on the server (placed by setup-prod.sh). Fail loudly if missing.
[ -f .env ] || { echo "[deploy] ERROR: /opt/jobfit/.env missing — run setup-prod.sh first" >&2; exit 1; }
[ -f data/ispv_2025.xlsx ] || echo "[deploy] WARNING: data/ispv_2025.xlsx missing — /analyze will 503 until present"

echo "[deploy] restarting jobfit.service"
sudo -n systemctl restart jobfit.service
sleep 2
systemctl --no-pager --full status jobfit.service | head -8
echo "[deploy] local health check"
curl -fsS http://127.0.0.1:8099/health
echo
echo "[deploy] done"
