# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TEE-llm-routing is a 3rd-party LLM routing node designed to run within AWS Nitro Enclave TEE (Trusted Execution Environment). It provides a secure, cryptographically verifiable interface to multiple LLM providers (OpenAI, Anthropic, Google Gemini, xAI Grok) with attestation and response signing capabilities.

## Common Commands

```bash
# Run server locally for development (without TEE)
make test-local              # Runs: python3 server.py

# Build enclave image
make image                   # Build Docker image as TAR using Kaniko

# Build and run in Nitro Enclave
make all                     # or: make run

# Test endpoints
make test-completion         # Test /v1/completions
make test-chat              # Test /v1/chat/completions
make test-stream            # Test /v1/chat/completions/stream

# Clean build artifacts
make clean
```

## Environment Variables

Required API keys (set via environment or pass in Authorization header):
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `XAI_API_KEY`

Server configuration:
- `LLM_SERVER_PORT` (default: 8000)
- `LLM_SERVER_HOST` (default: 127.0.0.1)

## Architecture

### Core Flow

1. **TEEKeyManager** generates RSA-2048 key pair on startup and registers public key hash with nitriding daemon
2. Incoming requests are routed to appropriate LLM provider via LangChain
3. All responses are signed with RSA-PSS + SHA256, including request hash and timestamp

### Key Components in server.py

- **TEEKeyManager**: RSA key generation, registration with nitriding, response signing
- **Provider routing**: `get_provider_from_model()` infers provider from model name, `get_chat_model()` creates LangChain model instance
- **Message conversion**: `convert_messages()` transforms API messages to LangChain types

### API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/attestation` | TEE attestation document with public key |
| `/v1/completions` | Text completion (signed) |
| `/v1/chat/completions` | Chat completion with tool support (signed) |
| `/v1/chat/completions/stream` | Streaming chat via SSE |
| `/v1/models` | List available models |

### TEE Integration

- **Nitriding daemon** runs on localhost:8080, provides TLS termination (port 443 public)
- Endpoints `/enclave/ready` and `/enclave/hash` used for registration
- PCR measurements tracked in `measurements.txt`

### Supported Providers

Model name prefixes determine routing:
- **OpenAI**: gpt-4o, gpt-4-turbo, o3, o4
- **Anthropic**: claude-3.5, claude-3.7, claude-4.0
- **Google**: gemini-1.5, gemini-2.0, gemini-2.5
- **xAI**: grok-2, grok-3, grok-4

## Verification Examples

- `verify_attestation.py` - Validates AWS Nitro attestation documents
- `verify_signature_example.py` - Demonstrates request hash and signature verification

## Deployment

Uses multi-stage Docker build with nitriding daemon from `brave/nitriding-daemon`. Enclave launched via `run-enclave.sh` with gvproxy network bridge, allocating 2 CPUs and 4GB memory.
