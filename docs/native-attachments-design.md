# Design: Native LLM Attachments over the Private (OHTTP) Path

## Status

Proposal. Spans three repos: `chat-app` (browser), `chat-api` (relay), `tee-gateway`
(enclave). The bulk of the change lands in `tee-gateway`.

## Motivation

Today attachments are handled by **server-side parsing in `chat-api`**:

- `chat-api/src/core/attachments.py` downloads each attachment and runs PyMuPDF /
  python-docx to extract **plain text**, then injects that text into the prompt.
- Images are classified by content-type and passed through as URLs.

This is the wrong layer to solve the problem:

1. **It throws away everything the models do natively.** Modern Claude / GPT /
   Gemini ingest PDFs and images directly — layout, tables, figures, charts,
   handwriting, embedded images. Flattening a PDF to `page.get_text()` loses all
   of that and feeds the model a worse input than it could handle itself.
2. **It only works on the non-private path.** The parsing in `attachments.py` is
   invoked exclusively from the regular `POST /api/v1/chat` handler. On the
   **OHTTP path**, `chat-api` is a dumb relay — it forwards opaque ciphertext to
   the enclave and never sees the body — so attachments are simply not processed.
   Worse, in the enclave `llm_backend.convert_messages` flattens multimodal
   content parts to text only (`"".join(part.get("text", "") ...)`), so any
   `image_url` part is **silently dropped** before it reaches the provider.

Net result: **attachments and privacy are currently mutually exclusive.**
Attachments only work on the route where `chat-api` reads the plaintext, and the
private route drops them.

## Goal

Send attachments to the model **natively**, on the **private (OHTTP) path**:

- No server-side text extraction. The file bytes reach the model as a native
  image/document content part.
- `chat-api` and Cloudflare never see attachment plaintext (same trust boundary
  as the message text already enjoys on OHTTP).
- The enclave converts the inner request's multimodal content into each
  provider's native format via LangChain.

## Trust boundary (what this does and does not hide)

- **Hidden from:** the browser→relay transport, `chat-api`, the OHTTP relay,
  Cloudflare/R2. They see only HPKE ciphertext.
- **Visible to:** the enclave (it decrypts — that's the trust anchor) and the
  **upstream LLM provider** (OpenAI/Anthropic/Google/xAI/ByteDance), which
  receives the attachment as part of the completion request. This is identical
  to how message *text* is already handled: whatever you send the model, the
  model provider sees. Fully provider-blind attachments would require the model
  to run inside the TEE and are out of scope here.

## Transport: how the attachment reaches the enclave

### Phase 1 — inline base64 (recommended starting point)

The browser embeds the file directly in the message content as a standard
OpenAI-style content part, inside the HPKE-encrypted OHTTP payload:

```jsonc
{
  "model": "claude-sonnet-4-6",
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "Summarize this contract." },
        { "type": "image_url",
          "image_url": { "url": "data:image/png;base64,iVBORw0K..." } },
        { "type": "file",
          "file": { "filename": "contract.pdf",
                    "file_data": "data:application/pdf;base64,JVBERi0..." } }
      ]
    }
  ]
}
```

- Pros: nothing outside the enclave/provider ever sees the bytes; no R2 round
  trip; no presigned-URL machinery; no SSRF surface.
- Cons: base64 inflates ~33%; bounded by request/OHTTP size limits; no
  persistence (re-sent each turn). Fine for the common case (a few MB of PDF or
  an image). Enforce a hard per-request attachment-bytes cap in the enclave.

### Phase 2 — encrypted blob in R2 (only if large files / persistence needed)

Browser client-side-encrypts the file (AES-GCM), uploads **ciphertext** to R2
(Cloudflare sees only ciphertext), and includes inside the OHTTP payload an R2
reference plus the AES key **wrapped to the TEE attestation/HPKE public key**.
The enclave fetches the ciphertext and decrypts internally. Defer until Phase 1
limits become a real constraint.

> Note: do **not** go back to plaintext-in-R2 + presigned URLs. That reintroduces
> the public-bearer-token leak and the SSRF surface in `attachments.py`.

## Enclave changes (`tee-gateway`) — the core of the work

### 1. `convert_messages` must preserve multimodal content

`llm_backend.py:248-255` currently does:

```python
elif role == "user":
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    langchain_messages.append(HumanMessage(content=content))
```

Replace the flattening with a converter that maps OpenAI-style content parts to
**LangChain standard multimodal content blocks**, which `langchain-anthropic`,
`langchain-openai`, `langchain-google-genai`, and `langchain-xai` each translate
into their provider's native API:

- `text` → `{"type": "text", "text": ...}`
- `image_url` (data: URI or https) → LangChain image block
  (`{"type": "image", "source_type": "base64"|"url", ...}`)
- `file` / document (base64 PDF etc.) → LangChain file block
  (`{"type": "file", "source_type": "base64", "mime_type": ..., "data": ...}`)

Keep a `HumanMessage` with a **list** content when parts are present; only
collapse to a plain string when the message is text-only (preserves current
behavior for the no-attachment case).

### 2. No new heavy dependencies (PCR constraint)

Native handoff means the enclave does **not** parse PDFs/DOCX itself — it passes
the bytes to the provider. So we should **not** add PyMuPDF/python-docx to
`tee-gateway`. Avoiding new deps keeps `uv.lock` — and therefore the **PCR
measurements** — stable except for the deps actually required by LangChain's
multimodal blocks (verify whether the pinned `langchain-*` versions already
support file blocks; bump only if needed, and treat any `uv.lock` change as an
intentional PCR change per `CLAUDE.md`).

### 3. Per-provider capability gating

Not every model accepts every modality. Extend `model_registry` with capability
flags (e.g. `supports_image`, `supports_pdf`) and reject (clear 4xx inside the
inner request) when a request sends a modality the target model can't handle,
rather than silently dropping it as today.

### 4. Request signing / hashing

`chat_controller.py` (~645-651) hashes user content via `str(msg.content)`. With
multimodal content that would hash megabytes of base64 and is not canonical.
Define a stable hashing rule, e.g. hash each attachment as
`sha256(mime_type || raw_bytes)` and include those digests (not the base64) in
the canonical request JSON that feeds `keccak256(requestHash ...)`. This keeps
signatures meaningful and bounded while still committing to the exact attachment
content.

### 5. Limits & validation

- Hard cap on total attachment bytes per request (post-decode).
- Allowlist of accepted mime types per modality.
- Reject `image_url` values that are remote `https` URLs on the private path if
  we want to guarantee the enclave makes no outbound fetch for user content
  (Phase 1 = base64 only). Decide explicitly.

## `chat-api` changes

- OHTTP path: **no change needed** to the relay itself — attachments ride inside
  the encrypted payload it already forwards opaquely.
- Regular `POST /api/v1/chat` path: stop calling `load_documents` /
  `is_image_url` and stop injecting extracted text. Either (a) build native
  content parts here too, or (b) deprecate attachment support on the non-private
  path and route all attachments through OHTTP. Recommend (b) for a single code
  path.
- The presigned-URL / `attachments: string[]` machinery and `attachments.py`
  become dead code for inference and can be removed once Phase 1 ships (R2 may
  still be used for chat-history storage — that is a separate concern and should
  be client-side-encrypted if kept).

## `chat-app` changes

- Replace "upload to R2 → store presigned URL → send URL in `attachments`" with:
  read the file in the browser, base64-encode, and add a native `image_url` /
  `file` content part to the outgoing (to-be-encrypted) message.
- Enforce client-side size/type limits matching the enclave caps; surface a clear
  error when a file exceeds them.
- Drop the presigned-upload/download hooks from the send path.

## Rollout

1. Enclave: `convert_messages` multimodal support + capability flags + hashing +
   limits (behind the existing OHTTP path). Ship and verify PCRs.
2. `chat-app`: send native base64 content parts on the OHTTP path.
3. Remove server-side parsing from `chat-api`; retire `attachments.py` and the
   presigned-URL attachment flow.
4. (Optional, later) Phase 2 encrypted-R2-blob for large files.

## Open questions

- Pinned `langchain-*` versions: do they already support `file` (PDF) content
  blocks for Anthropic/OpenAI/Gemini, or do we need a version bump (→ PCR change)?
- Hard size cap value for inline attachments, and the OHTTP request size ceiling.
- Keep or drop attachment support entirely on the non-private path?
