#!/bin/bash

if [ $# -ne 1 ]
then
	echo >&2 "Usage: $0 IMAGE_EIF"
	exit 1
fi
image_eif="$1"

# gvproxy is the untrusted proxy application that runs on the EC2 host.  It
# acts as the bridge between the Internet and the enclave.  The code is
# available here:
# https://github.com/brave-intl/bat-go/tree/master/nitro-shim/tools/gvproxy
echo "[ec2] Starting gvproxy."
sudo gvproxy -listen vsock://:1024 -listen unix:///tmp/network.sock &
# sudo gvproxy -listen vsock://:1024 &
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
  -d '{"local":":8000","remote":"192.168.127.2:443"}'

# Print out ports forwarding through gproxy
echo "[ec2] Forwarded ports:"
sudo curl --unix-socket /tmp/network.sock http:/unix/services/forwarder/all 

# Run enclave with set memory and CPU count. Can also add debug mode and attach
# console if desired (--debug-mode and --attach-console), but this disables
# remote attestation.
echo "[ec2] Starting enclave."
nitro-cli run-enclave \
	--cpu-count 2 \
	--memory 4096 \
	--enclave-cid 4 \
	--eif-path "$image_eif"

echo "[ec2] Enclave ID: $enclave_id"

echo "[ec2] Saving PCR measurements."
measurements=$(nitro-cli describe-enclaves | jq --arg ENCLAVE_ID "$enclave_id" -r '.[] | select(.EnclaveID == $ENCLAVE_ID) | {Measurements: .Measurements}')
echo "$measurements" > measurements.txt

# Wait for the enclave to complete execution
echo "[ec2] Waiting for enclave to complete."
while nitro-cli describe-enclaves | grep -q "$enclave_id"; do
	sleep 5
done

echo "[ec2] Enclave has completed."

echo "[ec2] Stopping gvproxy."
sudo kill -INT "$pid"
