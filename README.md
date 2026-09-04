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

- **Multi-provider routing** - OpenAI, Anthropic, Google Gemini, xAI Grok, ByteDance (BytePlus ModelArk), OpenRouter
- **Remote attestation** - AWS Nitro attestation documents with PCR measurements
- **Response signing** - RSA-PSS signatures on all inference results
- **Request integrity** - SHA256 hash of original request included in signed response
- **Streaming support** - SSE streaming for chat completions
- **Tool/function calling** - Full support for LLM tool use
- **In-enclave web search** - Dedicated `/v1/web_search` endpoint (backed by
  Exa) for clients running their own tool loop. The query never leaves the
  enclave except to the search backend; every search is billed at one flat
  per-search rate

## Supported Models

| Provider | Models |
|----------|--------|
| OpenAI | gpt-6-astra, gpt-4.1, gpt-5, gpt-5-mini, gpt-5.6-sol, gpt-5.6-terra, gpt-5.6-luna, o4-mini |
| Anthropic | claude-fable-5-1, claude-sonnet-4-5, claude-sonnet-4-6, claude-haiku-4-5, claude-opus-4-5, claude-opus-4-6 |
| Google | gemini-3.8-flash, gemini-3.7-flash, gemini-3.6-flash, gemini-3.5-flash-lite, gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.5-pro, gemini-3-pro-preview, gemini-3-flash-preview |
| xAI | grok-4.6, grok-4.5, grok-4.3, grok-4, grok-4-fast, grok-4-1-fast, grok-4-1-fast-non-reasoning |
| ByteDance | seed-1.6, seed-1.8, seed-2.0-lite, deepseek-v4-flash, deepseek-v4-pro |
| OpenRouter | hermes-4-405b, hermes-4-70b, hy3 |

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
export ARK_API_KEY=...   # BytePlus / ByteDance ModelArk
export OPENROUTER_API_KEY=...  # OpenRouter
export ZAI_API_KEY=...   # Z.ai Model API

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

# Web search (dedicated endpoint; advertise a `web_search` tool to your model
# and call this when the model invokes it)
curl -X POST http://127.0.0.1:8000/v1/web_search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What happened in the news today?",
    "num_results": 6
  }'
```

> **Web search & billing.** `/v1/web_search` runs the search inside the enclave
> against Exa — it does not use any provider's built-in search, and the gateway
> runs no tool loop: feed the returned `content` back to your model as the tool
> result, and show `citations` to your user. The query never leaves the TEE
> except to the search backend, and via OHTTP (inner `"endpoint":
> "web_search"`) it is also invisible to the relay.
>
> Each search is one flat price, settled via x402 from the response's
> `opengradient` block like any other paid endpoint; failed searches return no
> cost block and are not charged. The response is signed with the same `tee_*`
> fields as chat (request hash over the canonical JSON body, output hash over
> `content`). Requires `EXA_API_KEY` to be injected; check `web_search_enabled`
> on `/health`. The old chat-request `web_search` flag is a deprecated no-op.

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
| `/health` | GET | Health check (status, version, tee_enabled) |
| `/enclave/attestation?nonce={nonce}` | GET | Nitro-enclave TEE attestation with public key hash and PCR information |
| `/signing-key` | GET | TEE public key (PEM format) and tee_id |
| `/v1/completions` | POST | Text completion (signed) |
| `/v1/chat/completions` | POST | Chat completion (signed) |
| `/v1/ohttp` | POST | Anonymous chat completion (OHTTP-encapsulated, relay-paid) |
| `/v1/ohttp/config` | GET | HPKE key configuration (RFC 9458) for OHTTP clients |

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

## Anonymous Inference (Oblivious HTTP)

`/v1/ohttp` is a thin wrapper around `/v1/chat/completions` that adds **client unlinkability** via [RFC 9458 OHTTP](https://www.rfc-editor.org/rfc/rfc9458) + [draft-ietf-ohai-chunked-ohttp-08](https://datatracker.ietf.org/doc/draft-ietf-ohai-chunked-ohttp/). HPKE ciphersuite is fixed: DHKEM(X25519,HKDF-SHA256) / HKDF-SHA256 / ChaCha20-Poly1305.

**Flow:**

1. Client fetches `/v1/ohttp/config` (HPKE pubkey, key_id, suite IDs) and verifies it. The HPKE key is **not** in the Nitro attestation transcript; instead the config carries an RSA-PSS `signature` from the enclave's attested signing key over `computeOHTTPConfigHash(...)`, giving the chain PCRs → attested signing key → signature → HPKE config (see [Verify OHTTP Config](#4-verify-ohttp-config)).
2. Client HPKE-encapsulates a normal chat-completion JSON body and POSTs the ciphertext to a **relay**. The client carries no payment material.
3. Relay forwards the ciphertext to `/v1/ohttp` and attaches its own `X-Payment: <x402 payload>` header. **`/v1/ohttp` is the x402-paid boundary** — verification and settlement happen on this outer request, against the relay's payment.
4. Enclave decrypts → re-issues the request in-process to `/v1/chat/completions` against the pre-x402 WSGI app (so connexion routing, validation, TEE signing and the LLM call still run, but x402 does **not** fire a second time and the relay's `X-Payment` is **not** forwarded into the inner dispatch) → response is sealed back to the client.

**Two response modes** (chosen by the inner `stream` flag):

| Mode | Outer content-type | Body |
|------|---|---|
| `stream=false` | `message/ohttp-res` | Single-shot sealed body (RFC 9458 §4.5) |
| `stream=true` | `message/ohttp-chunked-res` | `response_nonce \|\| (varint(len) \|\| sealed_ct)+ \|\| varint(0) \|\| sealed_final_ct` — one OHTTP chunk per SSE event, AAD=`b"final"` on the last chunk (chunked-ohttp draft §3) |

**Billing channel for the relay.** Both modes settle the actual cost via x402 against the relay's `X-Payment` (`upto` scheme); the gateway is the source of truth for the amount.

- `stream=false`: outer response exposes billing/cost headers — `X-Inference-Cost-OPG`, `X-Inference-Cost-USD`, `X-Inference-Price-OPG-USD` — for the relay's own bookkeeping. Per-token `usage` detail is carried in the sealed body for the client, not in outer `X-Usage-*` headers.
- `stream=true`: **no** per-token detail in outer headers (they're flushed before any body chunk, so we can't know token counts at header-write time) and the sealed chunks are opaque to the relay. The relay reads the actual settled amount from x402 — either by querying the facilitator with its `X-Upto-Session`, or via `X-Payment-Response` on its next call. The client still sees per-token detail in the final SSE event inside the decrypted stream.

On non-2xx (e.g. 402 payment required) the body is forwarded plaintext so the relay can read x402 payment requirements and retry — those bodies never contain prompts or completions.

**Trust split:**

- **Relay** terminates the client's TCP/TLS connection, so it does see the client's IP — that's unavoidable. What it doesn't see is content: only OHTTP ciphertext + its own wallet's `x-payment` material + the outer billing/cost headers used to settle and reconcile charges.
- **Enclave** sees plaintext prompts/completions (it has to run the LLM call) but at the network layer only sees the relay's IP, never the client's. This is the unlinkability claim — the enclave can't tie a plaintext request to a specific end user.
- **Client** decrypts and verifies the TEE signature embedded in the response body against the attested public key.

Unlinkability between a client identity and a plaintext request holds unless relay and enclave collude (the relay would have to share its client-IP log alongside the enclave's plaintext log). Streaming additionally leaks per-chunk timing and length — clients who can't accept that signal should use `stream=false`.

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

### 4. Verify OHTTP Config

The HPKE key for anonymous inference is **not** covered by the Nitro attestation
transcript. Before encrypting a request to it, verify the RSA-PSS `signature` in
`/v1/ohttp/config` against the **attested** signing key (recovered from the
attestation document — never trusted from the same response you're verifying):

```python
import base64
from eth_hash.auto import keccak
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

cfg = requests.get("https://your-enclave:443/v1/ohttp/config").json()

# public_key MUST come from the verified Nitro attestation document, not cfg.
public_key = serialization.load_pem_public_key(attested_public_key_pem.encode())
der = public_key.public_bytes(
    serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
)

# 1. The config is bound to *this* attested key: tee_id == keccak256(DER(key)).
tee_id = keccak(der)
assert tee_id == bytes.fromhex(cfg["tee_id"].removeprefix("0x"))

# 2. Recompute the signed hash (== TEERegistryV2.computeOHTTPConfigHash).
def word(v): return v.to_bytes(32, "big")
config_hash = keccak(
    keccak(b"OPENGRADIENT_TEE_OHTTP_CONFIG_V1")    # domain
    + tee_id                                        # bytes32
    + word(cfg["key_id"]) + word(cfg["kem_id"])
    + word(cfg["kdf_id"]) + word(cfg["aead_id"])
    + keccak(bytes.fromhex(cfg["public_key"]))      # keccak256(public_key)
    + keccak(base64.b64decode(cfg["key_config"]))   # keccak256(key_config)
)
assert config_hash == bytes.fromhex(cfg["signature_hash"].removeprefix("0x"))

# 3. Verify the RSA-PSS-SHA256 signature (salt_length=32 matches server).
public_key.verify(
    base64.b64decode(cfg["signature"]),
    config_hash,
    padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
    hashes.SHA256(),
)
# Only now is cfg["public_key"] safe to HPKE-encapsulate to.
```

See `examples/verify_ohttp_config.py` for a complete example. Skipping this step
leaves you with an unauthenticated HPKE key any network attacker could swap.

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
| `ARK_API_KEY` | - | BytePlus / ByteDance ModelArk API key (injected as `bytedance_api_key`) |
| `OPENROUTER_API_KEY` | - | OpenRouter API key (injected as `openrouter_api_key`) |
| `ZAI_API_KEY` | - | Z.ai Model API key (injected as `zai_api_key`) |
| `EVM_PAYMENT_ADDRESS` | - | Wallet address to receive x402 payments |
| `FACILITATOR_URL` | see `tee_gateway/__main__.py` | x402 payment facilitator endpoint |

API keys can also be injected at runtime via `POST /v1/keys` (preferred for TEE deployments to avoid baking secrets into the image).

## License

See LICENSE file for details.
