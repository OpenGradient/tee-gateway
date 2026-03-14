#!/bin/sh

# Start nitriding - proxies external port 443 to Flask app on port 8000
# Internal API on port 8080 (for /enclave/ready, /enclave/hash)
nitriding -fqdn localhost -appwebsrv "http://127.0.0.1:8000" -ext-pub-port 443 -intport 8080 -wait-for-app &
echo "[sh] Started nitriding."

sleep 1

# Start the Flask/connexion OpenAI-compatible API on port 8000.
# TEE key management (key generation, nitriding registration, response signing)
# and nitriding readiness signaling all happen inside this process.
echo "[sh] Starting OpenAI-compatible API server on port 8000..."
cd /app
python3 -m tee_gateway
echo "[sh] API server exited."
