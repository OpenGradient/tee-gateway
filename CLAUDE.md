# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenGradient TEE-gateway is an LLM routing service designed to run within AWS Nitro Enclave TEE (Trusted Execution Environment). It provides a secure, cryptographically verifiable interface to multiple LLM providers (OpenAI, Anthropic, Google Gemini, xAI Grok) with remote attestation, response signing, and x402v2 micropayment access control.

## Project Structure highlighting core files

```
├── tee_gateway/             # Main application package (Flask/connexion)
│   ├── __main__.py          # Entry point: app factory, x402 middleware setup, key injection
│   ├── llm_backend.py       # LLM provider routing via LangChain, HTTP client management
│   ├── tee_manager.py       # TEE key generation, nitriding registration, response signing
│   ├── model_registry.py    # Model config and per-token pricing
│   ├── definitions.py       # On-chain addresses, network IDs, payment amounts
│   ├── facilitator_api.py   # x402 facilitator API client
│   ├── heartbeat/           # Heartbeat/health monitoring
│   ├── controllers/         # Request handlers (chat, completions, security)
│   ├── models/              # OpenAI-compatible Pydantic models
│   ├── openapi/             # openapi.yaml spec
│   └── test/                # Unit tests
├── scripts/
│   ├── start.sh             # Enclave startup script (nitriding + server)
│   ├── run-enclave.sh       # EC2 host launcher (gvproxy, EIF, key injection)
├── examples/                # Client-side verification examples
├── pyproject.toml           # Project metadata and dependencies (managed by uv)
├── uv.lock                  # Locked dependency versions and hashes
├── Dockerfile               # Multi-stage: nitriding builder + python:3.12-slim-bullseye + uv
├── Makefile
└── measurements.txt         # PCR measurements for the deployed enclave image
```

## Common Commands

```bash
# Dependency management (uses uv — https://docs.astral.sh/uv/)
uv sync                      # Install/update dependencies from uv.lock
uv add <package>             # Add a new dependency
uv lock                      # Regenerate lockfile after editing pyproject.toml
# IMPORTANT: uv.lock is baked into the Docker image and affects PCR measurements.
# Only regenerate the lockfile when intentionally changing dependencies.

# Run server locally for development (without TEE)
make test-local              # Runs: uv run python -m tee_gateway

# Linting and type checking
make lint                    # Run ruff format + ruff check + mypy
make mypy                    # Run mypy type checker only

# Build enclave image
make image                   # Build Docker image as TAR using Kaniko

# Build EIF and run in Nitro Enclave
make run                     # or: make all

# Clean build artifacts
make clean

# Show all available targets
make help
```

## Environment Variables

API keys (injected at runtime via POST /v1/keys — do NOT bake into the image):
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `XAI_API_KEY`

Server configuration:
- `API_SERVER_PORT` (default: 8000)
- `API_SERVER_HOST` (default: 0.0.0.0)
- `EVM_PAYMENT_ADDRESS` — wallet address to receive x402 payments
- `FACILITATOR_URL` — x402 facilitator endpoint

## Architecture

### Core Flow

1. **TEEKeyManager** (`tee_manager.py`) generates RSA-2048 key pair on startup and registers the public key hash with the nitriding daemon
2. Incoming requests pass through x402v2 payment middleware before reaching handlers
3. Requests are routed to the appropriate LLM provider via LangChain (`llm_backend.py`)
4. All responses are signed with RSA-PSS-SHA256 over `keccak256(requestHash || outputHash || timestamp)`
5. Clients verify attestation → get public key → verify signatures

### Key Components

- **`tee_manager.py`**: RSA key generation, nitriding registration (`/enclave/hash`), response signing
- **`llm_backend.py`**: LangChain model instantiation, HTTP client management, provider routing from model name
- **`model_registry.py`**: Maps model names to providers and per-token USD pricing (used by dynamic cost calculator)
- **`definitions.py`**: On-chain constants (addresses, network IDs, payment amounts) — configure here for your deployment
- **`util.py`**: `dynamic_session_cost_calculator` converts actual token usage to x402 payment amounts

### API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/health` | Health check with system metrics |
| `/signing-key` | TEE public key (PEM) and tee_id |
| `/enclave/attestation` | Nitro attestation document (served by nitriding) |
| `/v1/keys` | One-time API key injection (POST, loopback-only) |
| `/v1/completions` | Text completion (signed) |
| `/v1/chat/completions` | Chat completion with tool support (signed) |
| `/v1/models` | List available models |

### TEE Integration

- **Nitriding daemon** runs on localhost:8080, provides TLS termination (port 443 externally)
- Endpoints `/enclave/ready` and `/enclave/hash` used for nitriding registration
- PCR measurements in `measurements.txt` fingerprint the exact enclave image

### Supported Providers

Model name prefixes determine routing:
- **OpenAI**: gpt-4.1, gpt-5, gpt-5-mini, gpt-5.2, o4-mini
- **Anthropic**: claude-sonnet-4-0/4-5/4-6, claude-haiku-4-5, claude-opus-4-5/4-6, claude-3-7-sonnet, claude-3-5-haiku
- **Google**: gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.5-pro, gemini-3-pro-preview, gemini-3-flash-preview
- **xAI**: grok-2, grok-3, grok-3-mini, grok-4, grok-4-fast, grok-4-1-fast

## Verification Examples

- `examples/verify_attestation.py` — Validates AWS Nitro attestation documents against the root CA
- `examples/verify_signature_example.py` — Demonstrates request hash and RSA-PSS signature verification

## Deployment

Multi-stage Docker build: nitriding compiled from source (`brave/nitriding-daemon`), then copied into `python:3.12.10-slim-bullseye`. Dependencies are installed via `uv sync --frozen` from the lockfile for reproducible builds. Enclave launched via `scripts/run-enclave.sh` with gvproxy as the vsock network bridge, allocating 2 CPUs and 8GB memory.

Port 8000 is forwarded to `127.0.0.1` only on the EC2 host (loopback-only for key injection). Port 443 is forwarded publicly via gvproxy.
