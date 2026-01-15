# TEE-llm-routing

A secure LLM routing service designed to run within AWS Nitro Enclave TEE (Trusted Execution Environment). Provides cryptographically verifiable LLM responses with remote attestation, enabling clients to prove that responses were generated inside a trusted enclave and were not tampered with.

## Why TEE for LLM Routing?

When using third-party LLM providers, you typically must trust:
1. The routing service operator isn't modifying your requests/responses
2. Responses actually came from the claimed LLM provider
3. Your requests weren't logged or intercepted

TEE-llm-routing solves this by running inside a hardware-isolated Nitro Enclave where:
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
| OpenAI | gpt-4o, gpt-4o-mini, gpt-4-turbo, o3, o4-mini |
| Anthropic | claude-3.5-haiku, claude-3.7-sonnet, claude-4.0-sonnet |
| Google | gemini-1.5-pro, gemini-2.0-flash, gemini-2.5-flash, gemini-2.5-pro |
| xAI | grok-2, grok-3, grok-4.1-fast |

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

# Run server
make test-local
# or: python3 server.py
```

### Test Endpoints

```bash
# Chat completion
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7
  }'

# Streaming
curl -X POST http://127.0.0.1:8000/v1/chat/completions/stream \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "model": "gemini-2.5-flash-lite",
    "messages": [{"role": "user", "content": "Write a haiku"}]
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
- 4GB memory
- Port 443 (HTTPS via nitriding)
- Port 8000 (internal server)

### PCR Measurements

After launching, PCR measurements are saved to `measurements.txt`. Share these with clients so they can verify attestation documents match the expected enclave code.

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/attestation` | GET | TEE attestation with public key |
| `/v1/completions` | POST | Text completion (signed) |
| `/v1/chat/completions` | POST | Chat completion (signed) |
| `/v1/chat/completions/stream` | POST | Streaming chat (SSE) |
| `/v1/models` | GET | List available models |

### Request Format

```json
{
  "model": "gpt-4o-mini",
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
  "message": {
    "role": "assistant",
    "content": "Hello! How can I help?"
  },
  "model": "gpt-4o-mini",
  "finish_reason": "stop",
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 8,
    "total_tokens": 18
  },
  "timestamp": "2025-01-15T10:30:00+00:00",
  "request_hash": "3cd5e62557ea16dc77aef5c2c66188d1...",
  "signature": "PLyCgScL1Jr6OSb7wazEbor4yhBYJpau..."
}
```

## Verification

### 1. Verify Attestation

Get the attestation document and verify it against AWS Nitro root certificate:

```bash
curl https://your-enclave:443/enclave/attestation?nonce=your-nonce
```

See `verify_attestation.py` for full verification including:
- PCR measurement validation
- Certificate chain verification
- Nonce verification
- Public key extraction

### 2. Verify Response Signature

After getting a response, verify the signature using the attested public key:

```python
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
import json

# Reconstruct signed data (must match server exactly)
signed_data = {
    "finish_reason": response["finish_reason"],
    "message": response["message"],
    "model": response["model"],
    "request_hash": response["request_hash"],
    "timestamp": response["timestamp"]
}

# Verify signature
public_key.verify(
    base64.b64decode(response["signature"]),
    json.dumps(signed_data, sort_keys=True).encode('utf-8'),
    padding.PSS(
        mgf=padding.MGF1(hashes.SHA256()),
        salt_length=padding.PSS.MAX_LENGTH
    ),
    hashes.SHA256()
)
```

See `verify_signature_example.py` for a complete example.

### 3. Verify Request Hash

The `request_hash` proves your original request wasn't modified:

```python
import hashlib
import json

original_request = {"model": "...", "messages": [...], ...}
computed_hash = hashlib.sha256(
    json.dumps(original_request, sort_keys=True).encode()
).hexdigest()

assert computed_hash == response["request_hash"]
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Nitro Enclave                           │
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │    nitriding    │    │          server.py              │ │
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

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_SERVER_PORT` | 8000 | Internal server port |
| `LLM_SERVER_HOST` | 127.0.0.1 | Server bind address |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `ANTHROPIC_API_KEY` | - | Anthropic API key |
| `GOOGLE_API_KEY` | - | Google AI API key |
| `XAI_API_KEY` | - | xAI API key |

API keys can also be passed per-request via `Authorization: Bearer <key>` header.

## License

See LICENSE file for details.
