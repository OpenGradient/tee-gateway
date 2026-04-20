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
sudo chmod 660 /tmp/network.sock

# Add port forwarding through gvproxy to nitriding
echo "[ec2] Adding port forwarding to nitriding"
echo "[ec2] Forwarding port 443"
sudo curl \
  --unix-socket /tmp/network.sock \
  http:/unix/services/forwarder/expose \
  -X POST \
  -d '{"local":":443","remote":"192.168.127.2:443"}'

echo "[ec2] Forwarding port 8000 (loopback only — used for key injection from this host)"
sudo curl \
  --unix-socket /tmp/network.sock \
  http:/unix/services/forwarder/expose \
  -X POST \
  -d '{"local":"127.0.0.1:8000","remote":"192.168.127.2:8000"}'

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

        # FACILITATOR_URL is used for both x402 payment verification and the heartbeat relay.
        # HEARTBEAT_CONTRACT_ADDRESS and TEE_HEARTBEAT_INTERVAL are optional heartbeat parameters.
        # The TEE wallet key is generated inside the enclave and never injected.
        HEARTBEAT_CONTRACT_ADDRESS="$(grep -E '^HEARTBEAT_CONTRACT_ADDRESS=' "$ENV_FILE" | cut -d'=' -f2-)"
        FACILITATOR_URL="$(grep -E '^FACILITATOR_URL=' "$ENV_FILE" | cut -d'=' -f2-)"
        TEE_HEARTBEAT_INTERVAL="$(grep -E '^TEE_HEARTBEAT_INTERVAL=' "$ENV_FILE" | cut -d'=' -f2-)"

        # Build the JSON payload using jq for safe escaping
        # Note: wallet private key is generated inside the TEE, not injected
        JSON_PAYLOAD=$(jq -n \
            --arg openai "$OPENAI_API_KEY" \
            --arg google "$GOOGLE_API_KEY" \
            --arg anthropic "$ANTHROPIC_API_KEY" \
            --arg xai "$XAI_API_KEY" \
            --arg hb_contract "$HEARTBEAT_CONTRACT_ADDRESS" \
            --arg facilitator "$FACILITATOR_URL" \
            --arg hb_interval "$TEE_HEARTBEAT_INTERVAL" \
            '{
                openai_api_key: $openai,
                google_api_key: $google,
                anthropic_api_key: $anthropic,
                xai_api_key: $xai
            }
            + if $hb_contract != "" then {heartbeat_contract_address: $hb_contract} else {} end
            + if $facilitator != "" then {facilitator_url: $facilitator} else {} end
            + if $hb_interval != "" then {tee_heartbeat_interval: $hb_interval} else {} end
            ')

        echo "[ec2] Injecting API keys and configuration into enclave..."
        http_status=$(curl -s -o /dev/null -w "%{http_code}" \
            -X POST \
            -H "Content-Type: application/json" \
            -d "$JSON_PAYLOAD" \
            http://localhost:8000/v1/keys)

        if [ "$http_status" = "200" ]; then
            echo "[ec2] API keys injected successfully."
            if [ -n "$HEARTBEAT_CONTRACT_ADDRESS" ]; then
                if [ -n "$FACILITATOR_URL" ]; then
                    echo "[ec2] Heartbeat service configured via facilitator relay (contract: ${HEARTBEAT_CONTRACT_ADDRESS})"
                else
                    echo "[ec2] Heartbeat service configured using enclave default facilitator URL (contract: ${HEARTBEAT_CONTRACT_ADDRESS})"
                fi
            else
                echo "[ec2] Heartbeat service not configured (missing HEARTBEAT_CONTRACT_ADDRESS)."
            fi
        else
            echo "[ec2] Warning: Key injection returned HTTP $http_status. Check enclave logs."
        fi

        # Clear key variables from this shell immediately after use
        unset OPENAI_API_KEY GOOGLE_API_KEY ANTHROPIC_API_KEY XAI_API_KEY
        unset HEARTBEAT_CONTRACT_ADDRESS FACILITATOR_URL TEE_HEARTBEAT_INTERVAL
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
