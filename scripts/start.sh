#!/bin/sh

# Start nitriding - proxies external port 443 to Flask app on port 8000
# Internal API on port 8080 (for /enclave/ready, /enclave/hash)
nitriding -fqdn localhost -appwebsrv "http://127.0.0.1:8000" -ext-pub-port 443 -intport 8080 -wait-for-app &
echo "[sh] Started nitriding."

sleep 1

# Start server.py as internal LLM backend on port 8001 (no TEE management)
echo "[sh] Starting LLM backend on port 8001..."
TEE_ENABLED=false LLM_SERVER_PORT=8001 python3 /bin/server.py &
echo "[sh] LLM backend started."

sleep 2

# Start the Flask/connexion OpenAI-compatible API on port 8000
# This is the front-facing server that nitriding proxies to.
# TEE key management and nitriding readiness signaling happen here.
echo "[sh] Starting OpenAI-compatible API server on port 8000..."
cd /app
python3 -m openapi_server
echo "[sh] API server exited."

