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
for a routine model addition. Always use the same provider as the existing model/family uses.

## Register only what will actually be offered

The registry is cheap to append to, which is exactly why it grows without
anyone deciding to. Do not register a model speculatively: routing exists so
chat-api's catalog can offer the model, and that catalog holds a real curation
bar (`.claude/skills/add-catalog-model/SKILL.md` there — a new model earns a
picker row only if it gives users something no current row does, and a better
model replaces the one it supersedes rather than joining it). Apply the same
judgement here first. If the model would not clear that bar, do not register
it; if it supersedes a model already registered, expect the catalog change to
hide the old one, and say which in your PR.

Registered-but-unoffered ids are not free: every one of them is a name the
gateway must keep routing, price correctly and test, for a model no user can
select.

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
   - `supports_temperature=False` — the API 400s if `temperature` is present
     at all. Read the section below before deciding this one: getting it wrong
     breaks 100% of the model's requests, and nothing short of a live call
     tells you.
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

8. **Send one real request to the model before you call it done** (see below).
   The unit suite constructs models but never calls a provider, so it cannot
   tell a working registration from one that 400s every time.

## Parameter restrictions are the thing that bites

Pricing is checkable on a web page. Which request fields a model *accepts* is
not, and the registry's optional flags exist because providers keep removing
fields from their newest models — `temperature` first among them. A model whose
API rejects a field the gateway sends fails **every** request, immediately,
with nothing partial about it:

```
400 Unsupported parameter: 'temperature' is not supported with this model.
```

Every client sends a temperature (chat-app pins `0.0`, and it is inside the
signed request hash), so there is no traffic that avoids the bad path and no
client-side workaround. Three things make this easy to miss:

- **Restrictions do not travel down a family.** GPT-6 Astra rejects
  `temperature`; every gpt-5.x model in the registry is registered without the
  flag. Copying the closest sibling's `ModelConfig` — right for pricing shape,
  request shaping and `responses_api_for_tools` — is exactly wrong here.
- **langchain hides it for some names and not others.** `langchain-openai`
  strips `temperature` itself for model names starting with `gpt-5`, and only
  those, so the whole gpt-5.6 family worked without the flag and the first
  `gpt-6-*` model did not. Never conclude "the sibling works, so the field is
  fine" — the sibling may be surviving on a provider-package special case
  keyed to its *name*.
- **Reasoning-first models are the usual suspects.** Anthropic dropped
  `temperature` at Opus 4.7 and kept it dropped (Fable 5, Fable 5.1); OpenAI
  dropped it for the o-series, then constrained it for gpt-5, then dropped it
  for GPT-6. Treat a new flagship reasoning model as rejecting it until a live
  call says otherwise, and read the provider's API changelog for the release,
  not just its pricing page.

### Proving it

`tee_gateway/test/test_provider_usage_integration.py` makes real, billable
requests through the actual chat-controller path, with `temperature=0` and
`max_tokens` set, in both streaming and non-streaming modes. Point its
`NEW_CHAT_MODELS` tuple (or `IMAGE_MODELS`) at the model you just registered —
it holds the newest model per provider, so replace that provider's entry rather
than growing the list — and run it:

```sh
RUN_PROVIDER_INTEGRATION_TESTS=1 \
  OPENAI_API_KEY=... ANTHROPIC_API_KEY=... GOOGLE_API_KEY=... XAI_API_KEY=... \
  ARK_API_KEY=... OPENROUTER_API_KEY=... ZAI_API_KEY=... \
  uv run --group test pytest \
  tee_gateway/test/test_provider_usage_integration.py -v
```

No local keys? The same suite is a manual CI job — GitHub Actions → **Test** →
*Run workflow* runs `live-provider-usage` with the repo's provider secrets
(`workflow_dispatch` only, so it never fires on a PR).

It needs every provider key (the module refuses to run without them) and costs
real money, which is why it is not part of `make test`. Run it anyway for a new
model: a few cents here against a model that answers nothing in production. If
the keys are genuinely not available to you, say so in the PR body — plainly,
as an unverified registration — rather than letting silence imply it was
checked.

A 400 from this test is the finding. Read the `param` field in the error, set
the matching flag, and re-run until the model actually answers; `llm_backend`
passes `None` for a flagged field, which every `langchain-<provider>` package
turns into "omit the key" rather than "send null".

## What this does *not* cover

The gateway only learns to route the model. Clients still show nothing until
the model is added to chat-api's `src/api/v1/model_catalog.py`, which owns the
user-facing name, description, tier, badge and access grants. Land the gateway
change first: the catalog offering a model the gateway rejects is a request
that fails after the user picks it.
