#!/bin/sh

# Sync system clock once at startup via public NTP.
# The Nitro Enclave's kvm-clock drifts ~1 sec/day and is never auto-corrected.
# ntpdate -b does a one-shot step correction (sets clock immediately, then exits).
# Tries Cloudflare first, falls back to Google; logs a warning but continues if both fail.
echo "[sh] Syncing system clock..."
if ntpdate -b time.cloudflare.com 2>&1; then
    echo "[sh] Clock synced via time.cloudflare.com"
elif ntpdate -b time.google.com 2>&1; then
    echo "[sh] Clock synced via time.google.com"
else
    echo "[sh] WARNING: Clock sync failed — timestamps may be inaccurate"
fi

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
