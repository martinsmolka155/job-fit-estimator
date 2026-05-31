#!/usr/bin/env bash
# One-time provisioning of jobfit.lilithai.dev on the prod server.
# PREREQUISITE: DNS A record  jobfit.lilithai.dev -> 89.167.88.19  must already
# resolve (certbot HTTP-01 needs it). Verify with: dig +short jobfit.lilithai.dev
#
# Run ON the server as the lilith user (passwordless sudo):
#   ssh lilith@89.167.88.19 'bash -s' < deploy/setup-prod.sh
# After this, ongoing updates use deploy/deploy.sh.
set -euo pipefail

APP_DIR="/opt/jobfit"
REPO="git@github.com:martinsmolka155/job-fit-estimator.git"
DOMAIN="jobfit.lilithai.dev"
PORT=8099

echo "[setup] 1/6 clone or update repo"
if [ ! -d "${APP_DIR}/.git" ]; then
  sudo mkdir -p "${APP_DIR}"
  sudo chown lilith:lilith "${APP_DIR}"
  git clone "${REPO}" "${APP_DIR}"
fi
cd "${APP_DIR}"
git fetch origin && git reset --hard origin/main

echo "[setup] 2/6 venv + deps"
python3 -m venv venv
./venv/bin/pip install -q -U pip
./venv/bin/pip install -q \
  "pymupdf4llm>=0.0.17" "python-docx>=1.1.2" "openai>=1.55.0" \
  "pydantic>=2.0" "pydantic-settings>=2.0" "rich>=13.0" \
  "reportlab>=4.2.0" "structlog>=24.0" "openpyxl>=3.1.0" \
  "httpx>=0.27" "fastapi>=0.115" "python-multipart>=0.0.9" "uvicorn>=0.32"

echo "[setup] 3/6 .env + ISPV data check"
if [ ! -f .env ]; then
  echo "  -> .env MISSING. Create /opt/jobfit/.env with OPENAI_API_KEY before starting." >&2
  echo "     (cp .env.example .env && edit) — service will 503 on /analyze without it."
fi
if [ ! -f data/ispv_2025.xlsx ]; then
  echo "  -> data/ispv_2025.xlsx MISSING. scp it from the laptop or fetch via the UI downloader." >&2
fi

echo "[setup] 4/6 systemd unit"
sudo cp deploy/jobfit.service /etc/systemd/system/jobfit.service
sudo systemctl daemon-reload
sudo systemctl enable jobfit.service
sudo systemctl restart jobfit.service
sleep 2
curl -fsS "http://127.0.0.1:${PORT}/health" && echo "  -> service healthy on :${PORT}"

echo "[setup] 5/6 nginx vhost (HTTP first, certbot adds SSL)"
# Temporary HTTP-only vhost so certbot HTTP-01 can validate. certbot rewrites it
# to add the 443 server + cert paths. The repo's deploy/nginx-jobfit.conf is the
# full SSL reference; certbot will produce the equivalent.
sudo tee /etc/nginx/sites-available/${DOMAIN} >/dev/null <<NGINX
limit_req_zone \$binary_remote_addr zone=jobfit_analyze:10m rate=5r/m;
server {
    listen 80;
    server_name ${DOMAIN};
    client_max_body_size 10m;
    location = /analyze {
        limit_req zone=jobfit_analyze burst=2 nodelay;
        limit_req_status 429;
        proxy_pass http://127.0.0.1:${PORT};
        proxy_set_header Host \$host;
        proxy_set_header X-Forwarded-For \$remote_addr;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_read_timeout 120s;
    }
    location / {
        proxy_pass http://127.0.0.1:${PORT};
        proxy_set_header Host \$host;
        proxy_set_header X-Forwarded-For \$remote_addr;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
NGINX
sudo ln -sf /etc/nginx/sites-available/${DOMAIN} /etc/nginx/sites-enabled/${DOMAIN}
sudo nginx -t && sudo systemctl reload nginx

echo "[setup] 6/6 certbot SSL (needs DNS resolving to this host)"
sudo certbot --nginx -d ${DOMAIN} --non-interactive --agree-tos \
  -m smolkamartin155@gmail.com --redirect || {
  echo "  -> certbot failed. Most likely DNS for ${DOMAIN} does not resolve yet." >&2
  echo "     Add an A record ${DOMAIN} -> 89.167.88.19, wait for propagation, re-run." >&2
  exit 1
}

echo "[setup] done. https://${DOMAIN} should be live."
