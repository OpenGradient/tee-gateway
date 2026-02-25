#!/bin/sh

nitriding -fqdn localhost -appwebsrv "http://127.0.0.1:8000" -ext-pub-port 443 -intport 8080 -wait-for-app &
NITRIDING_PID=$!
echo "[sh] Started nitriding (PID $NITRIDING_PID)."

sleep 1

if ! kill -0 $NITRIDING_PID 2>/dev/null; then
    echo "[sh] ERROR: nitriding died immediately, aborting."
    exit 1
fi

echo "[sh] Starting LLM backend on port 8001..."
TEE_ENABLED=false LLM_SERVER_PORT=8001 LLM_SERVER_HOST=127.0.0.1 python3 /bin/server.py &
LLM_PID=$!
echo "[sh] Started LLM backend (PID $LLM_PID)."

sleep 3

if ! kill -0 $LLM_PID 2>/dev/null; then
    echo "[sh] ERROR: LLM backend died, aborting."
    exit 1
fi

echo "[sh] Starting OpenAI-compatible API server on port 8000..."
cd /app
exec gunicorn \
    --bind 0.0.0.0:8000 \
    --workers 1 \
    --threads 4 \
    --timeout 120 \
    --keep-alive 5 \
    --preload \
    --log-level info \
    "openapi_server.__main__:application"
