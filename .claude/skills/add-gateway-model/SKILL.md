---
name: add-gateway-model
description: Add a new chat or image-generation model to the TEE gateway's model registry — enum entry, lookup aliases, pricing tests, and the two docs tables. Use when asked to add, register, price, retire or rename a model in tee-gateway, or when checking whether a provider's newly released model is already supported.
---

# Adding a model to the gateway

`tee_gateway/model_registry.py` is the single source of truth. An unregistered
model name is rejected outright — there is no fallback — so every model the
gateway can route is added here, with pricing, before anything else can use it.

Adding a model to a provider the gateway **already** speaks (OpenAI, Anthropic,
Google, xAI, ByteDance/ModelArk, OpenRouter, Z.ai) touches only the files
below. `llm_backend.py` routes off `ModelConfig.provider`, so it needs no
change. A model from a *new* provider is a different, larger job: it needs a
client, an API key path in `__main__.py` and `/v1/keys`, and is out of scope
for a routine model addition.

## Steps

1. **Check it isn't already there.** Grep `_MODEL_LOOKUP` for the api name and
   for near-miss aliases (`grok-4.5` / `grok-4-5` / `grok-4.5-latest`).

2. **Get real pricing from the provider's own pricing page.** Prices are
   `Decimal` USD **per token**, not per million: `$0.75/MTok` is
   `Decimal("0.00000075")`. Never carry a price over from memory or from
   another model "at the same tier" — confirm it against the provider's
   published page and cite that page in the PR body. A wrong price here is
   billed to real users on every request.

3. **Add the enum member** to `SupportedModel`, under its provider's `── … ──`
   section, newest first within the section:

   ```python
   GEMINI_3_8_FLASH = ModelConfig(
       provider="google",
       api_name="gemini-3.8-flash",   # exactly what the provider API expects
       input_price_usd=Decimal("0.00000075"),
       output_price_usd=Decimal("0.00000375"),
   )
   ```

   Reach for the optional fields only when the model needs them, and leave a
   comment saying why:
   - `supports_temperature=False` — the API 400s if `temperature` is present.
   - `force_temperature=1.0` — reasoning models that accept only one value.
   - `responses_api_for_tools=True` — OpenAI reasoning models that reject
     `reasoning_effort` alongside function tools on Chat Completions.
   - `image_output=True` + `image_output_price_usd` — inline-image chat models
     (Gemini "nano banana"), which bill image output tokens at a higher rate.
   - `image_generation=True` + `per_image_price_usd` — `/images/generations`
     models, billed flat per image (set the token prices to 0). These also
     need their request shaping decided: `image_response_format`,
     `image_send_n`, `image_supports_reference`, `image_edit_endpoint`,
     `image_extra_params`. Copy from the closest existing model on the same
     provider endpoint rather than guessing.

4. **Register every name a caller might send** in `_MODEL_LOOKUP`, beside the
   provider's other entries: the canonical id, the dotted/dashed variant, any
   `-latest` alias, the vendor-prefixed form OpenRouter uses
   (`tencent/hy3`), and — for ByteDance — the dated `seed-1-8-251228` form and
   the opaque `ep-…` deployment endpoint id when the model is served that way.
   Lookup is lowercased, so keep keys lowercase.

5. **Add tests** to `tests/test_pricing.py`, mirroring the neighbours:
   - a `test_<model>_resolves` asserting provider, `api_name` and both prices
     (plus `image_generation` / `per_image_price_usd` for image models);
   - a `test_<model>_cost` asserting the settled cost for 1000 in / 500 out
     against `_expected_cost_opg`, with the literal wei figure written out.

6. **Update the two model tables**, which are hand-maintained and drift
   otherwise: the provider list under "Supported Providers" in `CLAUDE.md`,
   and the table in `README.md`. Add the new name at the front of its row.

7. **Run `make lint` and `uv run pytest tests/test_pricing.py`.** Do not touch
   `uv.lock` — it is baked into the image and changes the PCR measurements.

## What this does *not* cover

The gateway only learns to route the model. Clients still show nothing until
the model is added to chat-api's `src/api/v1/model_catalog.py`, which owns the
user-facing name, description, tier, badge and access grants. Land the gateway
change first: the catalog offering a model the gateway rejects is a request
that fails after the user picks it.
