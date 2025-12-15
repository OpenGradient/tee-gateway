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
	--memory 4096 \
	--enclave-cid 4 \
	--eif-path "$image_eif" | jq -r '.EnclaveID')

echo "[ec2] Enclave ID: $enclave_id"

echo "[ec2] Saving PCR measurements."
measurements=$(nitro-cli describe-enclaves | jq --arg ENCLAVE_ID "$enclave_id" -r '.[] | select(.EnclaveID == $ENCLAVE_ID) | {Measurements: .Measurements}')
echo "$measurements" > measurements.txt

echo "[ec2] Enclave is running!"
echo "[ec2] Access endpoints:"
echo "  - Health: http://localhost:443/health"
echo "  - Attestation: http://localhost:443/attestation"
echo "  - Chat: http://localhost:443/v1/chat/completions"
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
