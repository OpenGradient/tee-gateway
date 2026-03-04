#!/bin/bash

if [ $# -ne 1 ]
then
	echo >&2 "Usage: $0 IMAGE_EIF"
	exit 1
fi
image_eif="$1"

# gvproxy is the untrusted proxy application that runs on the EC2 host.
# It acts as the bridge between the Internet and the enclave.
echo "[ec2] Starting gvproxy."
sudo gvproxy -listen vsock://:1024 -listen unix:///tmp/network.sock &
pid="$!"

# Wait for the socket file to be created
echo "[ec2] Waiting for gvproxy to initialize..."
for i in {1..10}; do
    if [ -S /tmp/network.sock ]; then
        break
    fi
    sleep 1
done

# Exit if not found
if [ ! -S /tmp/network.sock ]; then
    echo "Error: gvproxy failed to create the socket file"
    exit 1
fi

# Ensure that socket file has correct permissions
sudo chmod 777 /tmp/network.sock

# Add port forwarding through gvproxy to nitriding
echo "[ec2] Adding port forwarding to nitriding"
echo "[ec2] Forwarding port 443"
sudo curl \
  --unix-socket /tmp/network.sock \
  http:/unix/services/forwarder/expose \
  -X POST \
  -d '{"local":":443","remote":"192.168.127.2:443"}'

echo "[ec2] Forwarding port 8000"
sudo curl \
  --unix-socket /tmp/network.sock \
  http:/unix/services/forwarder/expose \
  -X POST \
  -d '{"local":":8000","remote":"192.168.127.2:8000"}'

# Print out ports forwarding through gproxy
echo "[ec2] Forwarded ports:"
sudo curl --unix-socket /tmp/network.sock http:/unix/services/forwarder/all 

# Run enclave with set memory and CPU count.
echo "[ec2] Starting enclave."
enclave_id=$(nitro-cli run-enclave \
	--cpu-count 2 \
	--memory 8192 \
	--enclave-cid 4 \
	--eif-path "$image_eif" | jq -r '.EnclaveID')

echo "[ec2] Enclave ID: $enclave_id"

echo "[ec2] Saving PCR measurements."
measurements=$(nitro-cli describe-enclaves | jq --arg ENCLAVE_ID "$enclave_id" -r '.[] | select(.EnclaveID == $ENCLAVE_ID) | {Measurements: .Measurements}')
echo "$measurements" > measurements.txt

# Inject API keys from .env if present
ENV_FILE="$(dirname "$(readlink -f "$0")")/../.env"
if [ -f "$ENV_FILE" ]; then
    echo "[ec2] Found .env file. Waiting for server to be ready..."

    # Poll the health endpoint until the Flask server is up (up to ~2 min)
    server_ready=0
    for i in $(seq 1 60); do
        if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
            server_ready=1
            echo "[ec2] Server is ready (attempt $i)."
            break
        fi
        sleep 2
    done

    if [ "$server_ready" -eq 0 ]; then
        echo "[ec2] Warning: Server did not become ready in time. Skipping key injection."
    else
        # Parse each key directly from the file to avoid polluting the shell environment
        OPENAI_API_KEY="$(grep -E '^OPENAI_API_KEY=' "$ENV_FILE" | cut -d'=' -f2-)"
        GOOGLE_API_KEY="$(grep -E '^GOOGLE_API_KEY=' "$ENV_FILE" | cut -d'=' -f2-)"
        ANTHROPIC_API_KEY="$(grep -E '^ANTHROPIC_API_KEY=' "$ENV_FILE" | cut -d'=' -f2-)"
        XAI_API_KEY="$(grep -E '^XAI_API_KEY=' "$ENV_FILE" | cut -d'=' -f2-)"

        echo "[ec2] Injecting API keys into enclave..."
        http_status=$(curl -s -o /dev/null -w "%{http_code}" \
            -X POST \
            -H "Content-Type: application/json" \
            -d "{\"openai_api_key\":\"${OPENAI_API_KEY}\",\"google_api_key\":\"${GOOGLE_API_KEY}\",\"anthropic_api_key\":\"${ANTHROPIC_API_KEY}\",\"xai_api_key\":\"${XAI_API_KEY}\"}" \
            http://localhost:8000/v1/keys)

        if [ "$http_status" = "200" ]; then
            echo "[ec2] API keys injected successfully."
        else
            echo "[ec2] Warning: Key injection returned HTTP $http_status. Check enclave logs."
        fi

        # Clear key variables from this shell immediately after use
        unset OPENAI_API_KEY GOOGLE_API_KEY ANTHROPIC_API_KEY XAI_API_KEY
    fi
else
    echo "[ec2] No .env file found at $ENV_FILE"
    echo "[ec2] API keys must be injected manually: POST http://localhost:8000/v1/keys"
fi

echo "[ec2] Enclave is running!"
echo "[ec2] Access endpoints:"
echo "  - Health: https://localhost:443/health"
echo "  - Attestation: https://localhost:443/signing-key"
echo "  - Chat: https://localhost:443/v1/chat/completions"
echo ""
echo "[ec2] To view logs: nitro-cli console --enclave-id $enclave_id"
echo "[ec2] To stop: nitro-cli terminate-enclave --enclave-id $enclave_id"
echo ""
echo "[ec2] Press Ctrl+C to stop gvproxy and clean up..."

# Wait for user interrupt
trap "echo '[ec2] Stopping gvproxy...'; sudo kill -INT $pid; exit 0" INT TERM

# Keep script running
while true; do
    if ! nitro-cli describe-enclaves | grep -q "$enclave_id"; then
        echo "[ec2] Enclave has stopped."
        break
    fi
    sleep 5
done

echo "[ec2] Stopping gvproxy."
sudo kill -INT "$pid"
