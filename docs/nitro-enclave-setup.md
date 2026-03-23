# Nitro Enclave Setup Guide

This guide covers setting up an AWS Nitro Enclave instance and running the TEE gateway.

## Prerequisites

- AWS EC2 instance with Nitro Enclaves enabled (recommended: `m5.xlarge`)
- Nitro Enclaves must be enabled in the instance settings before launch

---

## 1. Install Dependencies

```bash
# Nitro CLI
sudo dnf install -y aws-nitro-enclaves-cli
sudo yum install aws-nitro-enclaves-cli-devel -y
nitro-cli --version  # verify installation

# gvproxy (vsock network bridge)
curl -OL https://github.com/containers/gvisor-tap-vsock/releases/download/v0.7.3/gvproxy-linux-amd64
sudo mv gvproxy-linux-amd64 /usr/local/bin/gvproxy
sudo chmod +x /usr/local/bin/gvproxy
gvproxy --version  # verify installation

# Build tools
sudo yum install git pip make -y

# Docker (required for make image)
sudo yum install docker -y
sudo systemctl start docker && sudo systemctl enable docker
```

## 2. Start Enclave Services

```bash
sudo systemctl start nitro-enclaves-allocator.service && sudo systemctl enable nitro-enclaves-allocator.service
sudo systemctl start nitro-enclaves-vsock-proxy.service && sudo systemctl enable nitro-enclaves-vsock-proxy.service

# Verify all services are running
sudo systemctl status
```

## 3. Reboot

```bash
sudo shutdown -r now
```

## 4. Allocate Enclave Resources

Configure memory and CPU for the enclave. Ensure enough memory is allocated to load the model — the TEE gateway uses **8192 MB**:

```bash
sudo vi /etc/nitro_enclaves/allocator.yaml
```

Key settings:
```yaml
memory_mib: 8192
cpu_count: 2
```

---

## 5. Build and Run the TEE Gateway

```bash
git clone <tee-gateway-repo>
cd tee-gateway

make image   # Build reproducible Docker image
make run     # Build EIF and launch enclave
```

After `make run` completes, `measurements.txt` is updated with the PCR measurements of the built EIF image. These measurements cryptographically fingerprint the exact enclave image and are used to verify what code is running inside the TEE.

### Registering on the OG Network

TEE instances with valid PCR measurements can be registered in the TEE registry on the OpenGradient blockchain. Once registered, the instance is visible at [explorer.opengradient.ai/tee-registry](https://explorer.opengradient.ai/tee-registry) and is eligible to participate in the decentralized inference network.

Third-party operators can reproduce the PCR values locally by building the image themselves and comparing against `measurements.txt` — this is how anyone can independently verify what code a registered TEE is running.

> **Note:** Instructions for registering a TEE instance into the OG registry are currently private. This process is in active use but not yet publicly documented.

---

## Managing Enclaves

### Restart / Replace a Running Enclave

```bash
nitro-cli describe-enclaves                          # Get enclave ID
nitro-cli terminate-enclave --enclave-id <enclave-id>
```

Then re-run `make run` to launch a fresh instance.

---

## Troubleshooting

### VSOCK Connection Issues

If gvproxy or the vsock proxy fails to connect, clean up leftover state from the previous session:

```bash
rm /tmp/network.sock

lsof -i :2222        # Find the PID holding the VSOCK port
kill <PID>
```

Then retry `make run`.
