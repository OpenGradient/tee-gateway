#!/bin/sh

# Start nitriding - proxies external port 443 to Flask app on port 8000
# Internal API on port 8080 (for /enclave/ready, /enclave/hash)
nitriding -fqdn localhost -appwebsrv "http://127.0.0.1:8000" -ext-pub-port 443 -intport 8080 -wait-for-app &
echo "[sh] Started nitriding."

sleep 1

# Start the Flask/connexion OpenAI-compatible API on port 8000 under gunicorn
# (one gthread worker; settings and rationale in tee_gateway/gunicorn_conf.py).
# TEE key management (key generation, nitriding registration, response signing)
# and nitriding readiness signaling all happen inside the worker process. A
# worker exit halts gunicorn — see child_exit in the config — so this script
# ends exactly when the bare `python3 -m tee_gateway` used to.
echo "[sh] Starting OpenAI-compatible API server on port 8000..."
cd /app
gunicorn -c python:tee_gateway.gunicorn_conf tee_gateway.wsgi:application
echo "[sh] API server exited."
