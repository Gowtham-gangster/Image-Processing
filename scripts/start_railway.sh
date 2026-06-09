#!/bin/sh
set -e

PORT="${PORT:-8000}"

echo "=============================================="
echo " MaskAware API"
echo " PORT=${PORT}"
echo " RAILWAY_ENVIRONMENT=${RAILWAY_ENVIRONMENT:-unset}"
echo " DATA_DIR=${DATA_DIR:-unset}"
echo " RAILWAY_VOLUME_MOUNT_PATH=${RAILWAY_VOLUME_MOUNT_PATH:-unset}"
echo " CLOUD_LITE=${CLOUD_LITE:-auto}"
echo "=============================================="

exec python -m uvicorn api.index:app \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --timeout-keep-alive 120 \
  --log-level info
