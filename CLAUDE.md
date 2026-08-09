# Project Overview

OpenGradient TEE-gateway is an LLM routing service designed to run within AWS Nitro Enclave TEE (Trusted Execution Environment). It provides a secure, cryptographically verifiable interface to multiple LLM providers (OpenAI, Anthropic, Google Gemini, xAI Grok) with remote attestation, response signing, and x402v2 micropayment access control. The tee-gateway is a part of the decentralized OpenGradient network providing verifiable inference.

The repo must provide a stable AWS Nitro PCR when the code doesn't change in order to allow anyone to reproduce the PCRs locally by building the image as a way to verify what code we are running and also for 3rd party operators to set up their own tee-gateway nodes with the same PCRs in order to participate in the network.

## Project Structure highlighting core files

```
├── tee_gateway/             # Main application package (Flask/connexion)
│   ├── __main__.py          # Entry point: app factory, x402 middleware setup, key injection
│   ├── llm_backend.py       # LLM provider routing via LangChain, HTTP client management
│   ├── image_generation.py  # Endpoint-based image gen (/images/generations): request shaping, URL→inline-bytes, signed responses
│   ├── tee_manager.py       # TEE key generation, nitriding registration, response signing
│   ├── web_search.py        # In-enclave web search: Exa client, execution, result formatting
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
├── pyproject.toml           # Project metadata and dependencies (managed by uv)
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
- `ARK_API_KEY` (BytePlus / ByteDance ModelArk; injected as `bytedance_api_key`)
- `NOUS_API_KEY` (Nous Research / Nous Portal; injected as `nous_api_key`)
- `ZAI_API_KEY` (Z.ai Model API; injected as `zai_api_key`)
- `EXA_API_KEY` (Exa search; injected as `exa_api_key`) — backs the in-enclave
  `/v1/web_search` endpoint, not an LLM provider. Without it the endpoint
  returns 503 and `/health` reports `web_search_enabled: false`.

Server configuration:
- `API_SERVER_PORT` (default: 8000)
- `API_SERVER_HOST` (default: 0.0.0.0)
- `EVM_PAYMENT_ADDRESS` — wallet address to receive x402 payments
- `FACILITATOR_URL` — x402 facilitator endpoint

## Architecture

### Core Flow

1. **TEEKeyManager** (`tee_manager.py`) generates RSA-2048 key pair on startup and registers the public key hash with the nitriding daemon
2. Incoming requests pass through x402 payment middleware before reaching handlers
3. Requests are routed to the appropriate LLM provider via LangChain (`llm_backend.py`)
4. All responses are signed with RSA-PSS-SHA256 over `keccak256(requestHash || outputHash || timestamp)`
5. Clients verify attestation → get public key → verify signatures

### Key Components

- **`tee_manager.py`**: RSA key generation, nitriding registration (`/enclave/hash`), response signing
- **`llm_backend.py`**: LangChain model instantiation, HTTP client management, provider routing from model name
- **`model_registry.py`**: Maps model names to providers and per-token USD pricing (used by dynamic cost calculator)
- **`definitions.py`**: On-chain constants (addresses, network IDs, payment amounts) — configure here for your deployment
- **`web_search.py`**: Exa HTTP client, search execution, and result formatting/citation extraction (serves `/v1/web_search`)
- **`util.py`**: `dynamic_session_cost_calculator` converts actual token usage to x402 payment amounts

### API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/health` | Health check (status, version, tee_enabled, web_search_enabled) |
| `/signing-key` | TEE public key (PEM) and tee_id |
| `/enclave/attestation` | Nitro attestation document (served by nitriding) |
| `/v1/keys` | One-time API key injection (POST, loopback-only) |
| `/v1/completions` | Text completion (signed) |
| `/v1/chat/completions` | Chat completion with tool support (signed) |
| `/v1/web_search` | In-enclave Exa web search (signed, flat per-search price) |

### TEE Integration

- **Nitriding daemon** runs on localhost:8080, provides TLS termination (port 443 externally)
- Endpoints `/enclave/ready` and `/enclave/hash` used for nitriding registration
- PCR measurements in `measurements.txt` fingerprint the exact enclave image

### Supported Providers

Model name prefixes determine routing:
- **OpenAI**: gpt-4.1, gpt-5, gpt-5-mini, gpt-5.2, gpt-5.6-sol/terra/luna, o4-mini; image generation: gpt-image-2
- **Anthropic**: claude-sonnet-4-0/4-5/4-6, claude-sonnet-5, claude-haiku-4-5, claude-opus-4-5/4-6/4-7/4-8, claude-opus-5, claude-fable-5, claude-3-7-sonnet, claude-3-5-haiku
- **Google**: gemini-3.6-flash, gemini-3.5-flash-lite, gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.5-pro, gemini-3-pro-preview, gemini-3-flash-preview, gemini-3.1-pro-preview, gemini-3.5-flash; image generation: gemini-2.5-flash-image, gemini-3.1-flash-image
- **xAI**: grok-2, grok-3, grok-3-mini, grok-4, grok-4.3, grok-4.5, grok-4-fast, grok-4-1-fast; image generation: grok-2-image
- **ByteDance** (BytePlus ModelArk, OpenAI-compatible, ap-southeast): seed-1.6, seed-1.8, seed-2.0-lite, deepseek-v4-flash, deepseek-v4-pro, glm-5.2 (Z.ai's model served via a ModelArk deployment endpoint); image generation: seedream-4.0, seedream-5.0-lite, seedance-4.5, seedance-5.0
- **Nous Research** (Nous Portal, OpenAI-compatible): hermes-4-405b, hermes-4-70b
- **Z.ai** (Model API, OpenAI-compatible): image generation: glm-image (glm-5.2 chat is routed through BytePlus ModelArk, see ByteDance above)

Image generation via OpenAI (gpt-image-2), xAI (grok-2-image), ByteDance
(seedream-4.0, seedream-5.0-lite, seedance-4.5, seedance-5.0), and Z.ai (glm-image) is served
through a provider `/images/generations` endpoint rather than the chat path (see
`image_generation.py`), but is surfaced on `/v1/chat/completions` exactly like
Gemini's inline-image models (images returned out-of-band under the message
`images` key). The client always receives inline bytes: providers that hand back
a hosted URL (Z.ai, Seedance, Seedream 5.0 Lite) are fetched inside the enclave
and inlined as `data:` URIs (the fetch is guarded: http(s) only, non-public IP
hosts rejected, redirects + size capped, and only ever called on provider-
response URLs, never client input). Image-to-image editing and multi-image
compositing ("add this logo to this photo") send the input images inline
(`data:` URIs / `image_url` content parts on the latest user turn, up to 10),
forwarded to providers that support it. Delivery is one of two per-model paths:
ByteDance carries the references inline in the JSON `image` field of
`/images/generations`; OpenAI gpt-image is routed to its separate
`/images/edits` endpoint, where the references ride as multipart `image[]` file
uploads (only inline `data:` references are uploaded — a plain-URL reference is
skipped rather than dereferenced in the enclave). Per-provider request quirks
(response format, `n`, size/watermark, reference support, edit endpoint) live in
`model_registry.py`. These models are billed a flat per-image
price (see `per_image_price_usd`), not per token.

### Web Search

Web search is a dedicated endpoint — `POST /v1/web_search` — not a chat feature.
It does NOT use any provider's native web search (OpenAI/Anthropic/Google/xAI
all have one; those were removed), and the gateway runs no tool loop of its own:
the client advertises a `web_search` function tool to its model, calls this
endpoint when the model invokes it, and feeds the returned `content` back as the
tool result. The search runs inside the enclave against Exa (`web_search.py`),
so a query rides the same encrypted OHTTP channel as chat and is never visible
to the relay or the gateway operator. Points to keep in mind:

- **The chat/completions `web_search` request flag is a deprecated no-op.** It
  is still accepted (and still part of the signed request hash when sent) so
  old clients' requests parse and verify, but it binds nothing and bills
  nothing.
- **Every text model can search** — the tool lives in the client, so this is
  purely a question of function calling, not of provider search support.
- **Request/response**: `{"query", "num_results"?, "recency_days"?}` in;
  `content` (model-ready numbered results), `citations` (structured sources),
  and the standard `tee_*` signing fields out. The request hash covers the
  canonical (sorted-keys) JSON body; the output hash covers `content`.
- **Reachable through OHTTP**: the inner payload's `endpoint` field
  (`"web_search"`) routes the sealed request; absent means chat, so existing
  OHTTP clients are unaffected. Billing flows through the same outer
  cost-header / billing-frame channel the relay already consumes.
- **Billing is one flat rate** (`WEB_SEARCH_PRICE_USD`) per search that reached
  Exa, settled from the response's `opengradient` block like every paid
  endpoint. Validation failures (400/503) and Exa failures (502) return no cost
  block and are never settled. A search that ran but matched nothing IS billed.
- Exa's self-reported `costDollars` is logged for margin reconciliation only;
  settlement never depends on it.

### Content Moderation

Every `/v1/chat/completions` request (text and image-generation alike) is
scored against OpenAI's free `omni-moderation-latest` endpoint before any
provider is called (`moderation.py`). The check covers the newest user turn:
its text plus any attached images. Points to keep in mind:

- **Fail-open**: no OpenAI key or a moderation outage means requests proceed
  unscored (`checked: false`); a positive verdict always comes from a real
  moderation response. `/health` reports `moderation_enabled`.
- **Blocking**: a request flagged for a category in
  `moderation.BLOCKED_CATEGORIES` (default: `sexual/minors` only) is refused
  with HTTP 451 + `code: "moderation_blocked"` and never reaches a provider.
  All other flagged categories are reported but still served.
- **Response surface**: the full verdict (flagged/blocked/categories/scores)
  rides inside the sealed response body under the `moderation` key —
  non-streaming responses and the final SSE frame alike — outside the signed
  output hash, exactly like `images` and `usage`.
- **Relay signal**: flagged requests additionally carry content-free
  `X-Moderation-Flagged` / `X-Moderation-Categories` / `X-Moderation-Blocked`
  outer headers (forwarded through the OHTTP path) so the relay can run its
  per-user strike/blacklist policy. Clean traffic carries none of these — it
  is byte-identical to before.
- **Billing**: the moderation call is free and adds no cost block changes;
  blocked (451) requests produce no `opengradient` block and are never
  settled.

## Verification Examples

- `examples/verify_attestation.py` — Validates AWS Nitro attestation documents against the root CA
- `examples/verify_signature_example.py` — Demonstrates request hash and RSA-PSS signature verification

## Deployment

Multi-stage Docker build: nitriding compiled from source (`brave/nitriding-daemon`), then copied into `python:3.12.10-slim-bullseye`. Dependencies are installed via `uv sync --frozen` from the lockfile for reproducible builds. Enclave launched via `scripts/run-enclave.sh` with gvproxy as the vsock network bridge, allocating 2 CPUs and 8GB memory.

Port 8000 is forwarded to `127.0.0.1` only on the EC2 host (loopback-only for key injection). Port 443 is forwarded publicly via gvproxy.
