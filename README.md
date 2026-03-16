# OpenGradient TEE-gateway

[![Lint](https://github.com/OpenGradient/tee-gateway/actions/workflows/lint.yml/badge.svg)](https://github.com/OpenGradient/tee-gateway/actions/workflows/lint.yml)

A secure LLM routing service designed to run within AWS Nitro Enclave TEE (Trusted Execution Environment). Provides cryptographically verifiable LLM responses with remote attestation, enabling clients to prove that responses were generated inside a trusted enclave and were not tampered with.

## Why TEE for LLM Requests?

When using third-party LLM providers, you typically must trust:
1. The routing service operator isn't modifying your requests/responses
2. Responses actually came from the claimed LLM provider
3. Your requests weren't logged or intercepted

The gateway solves this by running inside a hardware-isolated Nitro Enclave where:
- Every response is **cryptographically signed** with a key generated inside the enclave
- The signing key is bound to **remote attestation** proving the enclave's code integrity
- Clients can **verify signatures** to ensure responses weren't tampered with

## Features

- **Multi-provider routing** - OpenAI, Anthropic, Google Gemini, xAI Grok
- **Remote attestation** - AWS Nitro attestation documents with PCR measurements
- **Response signing** - RSA-PSS signatures on all inference results
- **Request integrity** - SHA256 hash of original request included in signed response
- **Streaming support** - SSE streaming for chat completions
- **Tool/function calling** - Full support for LLM tool use

## Supported Models

| Provider | Models |
|----------|--------|
| OpenAI | gpt-4.1, gpt-5, gpt-5-mini, o4-mini |
| Anthropic | claude-sonnet-4-5, claude-sonnet-4-6, claude-haiku-4-5, claude-opus-4-5, claude-opus-4-6 |
| Google | gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.5-pro, gemini-3-pro-preview, gemini-3-flash-preview |
| xAI | grok-4, grok-4-fast, grok-4-1-fast, grok-4-1-fast-non-reasoning |

## Quick Start

### Local Development (without TEE)

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...
export XAI_API_KEY=...

# Run server (starts the Flask/connexion app on port 8000)
make test-local
# or: python3 -m tee_gateway
```

### Test Endpoints

```bash
# Chat completion
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4.1",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7
  }'

# Streaming
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [{"role": "user", "content": "Write a haiku"}],
    "stream": true
  }'

# Text completion
curl -X POST http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3.7-sonnet",
    "prompt": "Explain quantum computing in one sentence"
  }'
```

## Deployment to Nitro Enclave

Requires an EC2 instance with Nitro Enclave support (e.g., m5.xlarge with enclave enabled).

```bash
# Build enclave image
make image

# Build EIF and run enclave
make run
```

The enclave runs with:
- 2 CPUs
- 8GB memory
- Port 443 (HTTPS via nitriding)
- Port 8000 (internal server)

### PCR Measurements

PCR (Platform Configuration Register) measurements uniquely fingerprint the enclave image — they change whenever the code or build environment changes. They are automatically written to `measurements.txt` by `scripts/run-enclave.sh` when the enclave starts.

The `measurements.txt` checked into this repository reflects the OpenGradient-operated deployment. **If you build and run your own enclave image, your PCR values will differ.** After running `make run`, your `measurements.txt` will be updated with your enclave's measurements. Share this file with your clients so they can verify attestation documents match your specific build.

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/enclave/attestation?nonce={nonce}` | GET | Nitro-enclave TEE attestation with public key hash and PCR information |
| `/attestation` | GET | Get public key (PEM format) and enclave info |
| `/v1/completions` | POST | Text completion (signed) |
| `/v1/chat/completions` | POST | Chat completion (signed) |
| `/v1/chat/completions/stream` | POST | Streaming chat (SSE) |
| `/v1/models` | GET | List available models |

### Request Format

```json
{
  "model": "gpt-4.1",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 100,
  "tools": [...]  // optional
}
```

### Signed Response Format

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1747000000,
  "model": "gpt-4.1",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello! How can I help?"},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 8,
    "total_tokens": 18
  },
  "tee_signature": "PLyCgScL1Jr6OSb7wazEbor4yhBYJpau...",
  "tee_request_hash": "3cd5e62557ea16dc77aef5c2c66188d1...",
  "tee_output_hash":  "a7f3d91c4b08e2f50c3a6d8e...",
  "tee_timestamp": 1747000000,
  "tee_id": "0x4a2b..."
}
```

The `tee_*` fields provide cryptographic proof of the response:
- **`tee_request_hash`** — keccak256 of the canonicalized request JSON (proves input wasn't modified)
- **`tee_output_hash`** — keccak256 of the response content (proves output wasn't modified)
- **`tee_signature`** — RSA-PSS-SHA256 signature over `keccak256(requestHash || outputHash || timestamp)`
- **`tee_timestamp`** — Unix timestamp when the response was signed (proves freshness)
- **`tee_id`** — keccak256 of the enclave's DER-encoded public key (stable identifier for this enclave instance)

## Verification

### 1. Verify Attestation

Get the attestation document and verify it against AWS Nitro root certificate:

```bash
curl https://your-enclave:443/enclave/attestation?nonce=your-nonce
```

See `examples/verify_attestation.py` for full verification including:
- PCR measurement validation
- Certificate chain verification
- Nonce verification
- Public key extraction

### 2. Verify Response Signature

After getting a response, verify the signature using the attested public key:

```python
import base64, json
from eth_hash.auto import keccak
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes, serialization

# Load attested public key (from /signing-key endpoint)
public_key = serialization.load_pem_public_key(public_key_pem.encode())

# Reconstruct the msg_hash the server signed:
#   keccak256(abi.encodePacked(inputHash, outputHash, timestamp))
request_hash  = bytes.fromhex(response["tee_request_hash"])
output_hash   = bytes.fromhex(response["tee_output_hash"])
timestamp_bytes = response["tee_timestamp"].to_bytes(32, "big")
msg_hash = keccak(request_hash + output_hash + timestamp_bytes)

# Verify RSA-PSS-SHA256 signature (salt_length=32 matches server)
public_key.verify(
    base64.b64decode(response["tee_signature"]),
    msg_hash,
    padding.PSS(
        mgf=padding.MGF1(hashes.SHA256()),
        salt_length=32,
    ),
    hashes.SHA256(),
)
```

See `examples/verify_signature_example.py` for a complete example.

### 3. Verify Request Hash

The `tee_request_hash` proves your original request wasn't modified:

```python
from eth_hash.auto import keccak
import json

# Canonical request (same fields the server serializes, sorted keys)
original_request = {
    "model": "gpt-4.1",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
}
request_bytes = json.dumps(original_request, sort_keys=True).encode()
computed_hash = keccak(request_bytes).hex()

assert computed_hash == response["tee_request_hash"]
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Nitro Enclave                           │
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │    nitriding    │    │         tee_gateway/            │ │
│  │    (TLS/443)    │───▶│    TEEKeyManager (RSA keys)     │ │
│  │                 │    │    LangChain routing            │ │
│  │  /enclave/*     │    │    Response signing             │ │
│  └─────────────────┘    └─────────────────────────────────┘ │
│          │                           │                      │
│          │ Register key hash         │ LLM API calls        │
│          ▼                           ▼                      │
│   PCR measurements            OpenAI/Anthropic/etc          │
└─────────────────────────────────────────────────────────────┘
          │
          │ HTTPS (port 443)
          ▼
     gvproxy (EC2 host) ◀──── Internet
```

**Flow:**
1. On startup, `TEEKeyManager` generates RSA-2048 keypair
2. Public key hash registered with nitriding for attestation binding
3. Incoming requests routed to LLM provider via LangChain
4. Response signed with private key (includes request hash + timestamp)
5. Clients verify attestation → get public key → verify signatures

## Payment Model (x402)

This gateway uses [x402](https://github.com/opengradient/x402) micropayments for access control. Clients pay per request using on-chain EVM transactions (USDC or OPG on supported networks).

To operate your own gateway:
1. Set `EVM_PAYMENT_ADDRESS` to your wallet address in `.env`
2. Set `FACILITATOR_URL` to point to your facilitator service (or use the default)
3. Configure payment amounts in `tee_gateway/definitions.py` (`CHAT_COMPLETIONS_USDC_AMOUNT`, etc.)

Clients use an x402-compatible client (e.g., the [x402 SDK](https://github.com/opengradient/x402)) to authorize payments and include them in request headers.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `API_SERVER_PORT` | 8000 | Internal server port |
| `API_SERVER_HOST` | 0.0.0.0 | Server bind address |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `ANTHROPIC_API_KEY` | - | Anthropic API key |
| `GOOGLE_API_KEY` | - | Google AI API key |
| `XAI_API_KEY` | - | xAI API key |
| `EVM_PAYMENT_ADDRESS` | - | Wallet address to receive x402 payments |
| `FACILITATOR_URL` | see `tee_gateway/__main__.py` | x402 payment facilitator endpoint |

API keys can also be injected at runtime via `POST /v1/keys` (preferred for TEE deployments to avoid baking secrets into the image).

## License

See LICENSE file for details.
