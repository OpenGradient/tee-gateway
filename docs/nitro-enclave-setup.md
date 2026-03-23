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
```

## 2. Configure Users and Docker

Replace `<USER>` with your IAM instance user (e.g. `ec2-user`, or use `$USER`):

```bash
sudo usermod -aG ne <USER>
sudo usermod -aG docker <USER>
sudo systemctl start docker && sudo systemctl enable docker
```

## 3. Start Enclave Services

```bash
sudo systemctl start nitro-enclaves-allocator.service && sudo systemctl enable nitro-enclaves-allocator.service
sudo systemctl start nitro-enclaves-vsock-proxy.service && sudo systemctl enable nitro-enclaves-vsock-proxy.service

# Verify all services are running
sudo systemctl status
```

## 4. Reboot

```bash
sudo shutdown -r now
```

## 5. Allocate Enclave Resources

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

## 6. Build and Run the TEE Gateway

```bash
git clone <tee-gateway-repo>
cd tee-gateway

make image   # Build reproducible Docker image
make run     # Build EIF and launch enclave
```

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
