#!/bin/sh

# Add "-acme" to use Let's-Encrypt for the TLS cert. in order to do this, need to have a legit fqdn though -- localhost will cause an error.
nitriding -fqdn localhost -appwebsrv "http://127.0.0.1:8000" -ext-pub-port 443 -intport 8080 -wait-for-app &
echo "[sh] Started nitriding."

sleep 1

server.py
echo "[sh] Ran Python script."
