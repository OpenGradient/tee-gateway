#!/bin/sh

# Start nitriding - proxies external port 443 to Flask app on port 8000
# Internal API on port 8080 (for /enclave/ready, /enclave/hash)
nitriding -fqdn localhost -appwebsrv "http://127.0.0.1:8000" -ext-pub-port 443 -intport 8080 -wait-for-app &
echo "[sh] Started nitriding."

sleep 1

# Serve the Flask/connexion OpenAI-compatible API on port 8000 with gunicorn
# (production WSGI server) instead of the werkzeug dev server that
# `python3 -m tee_gateway` runs (that path remains for local development).
# TEE key management (key generation, nitriding registration, response signing)
# and nitriding readiness signaling all happen inside this process.
#
# Exactly ONE worker process, on purpose — all of these live in process
# memory and must be shared by every request:
#   - the TEE keys (RSA signing + HPKE): generated per process and registered
#     with nitriding; a second worker would sign and decrypt with DIFFERENT
#     keys behind the same endpoint,
#   - the one-time /v1/keys provider-key injection and the x402 middleware it
#     installs,
#   - the price feed and heartbeat background threads.
# Concurrency comes from the gthread pool instead, matching the dev server's
# threaded behavior. --timeout 0 disables gunicorn's worker watchdog so
# long-lived requests (image generation, streamed completions) aren't killed
# mid-flight — request duration is already bounded by the app's own httpx
# timeouts, and the dev server had no watchdog either. No --max-requests:
# recycling the worker would regenerate the TEE keys mid-life.
echo "[sh] Starting OpenAI-compatible API server on port 8000..."
cd /app
python3 -m gunicorn \
    --workers 1 \
    --worker-class gthread \
    --threads 32 \
    --timeout 0 \
    --bind "${API_SERVER_HOST:-0.0.0.0}:${API_SERVER_PORT:-8000}" \
    --access-logfile - \
    --error-logfile - \
    tee_gateway.__main__:application
echo "[sh] API server exited."
