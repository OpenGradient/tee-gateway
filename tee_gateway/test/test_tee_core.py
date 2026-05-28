"""
Tests for core TEE server functionality.

Covers the components that make the TEE gateway trustworthy:
  - TEEKeyManager: RSA-2048 key generation, tee_id derivation, RSA-PSS signing
  - compute_tee_msg_hash: on-chain-compatible keccak256 hash
  - model_registry: provider routing and per-token pricing
  - llm_backend.convert_messages: OpenAI-format → LangChain message conversion

None of these tests require a running server, API keys, or nitriding.
"""

import base64
import unittest

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from eth_hash.auto import keccak
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from tee_gateway import ohttp
from tee_gateway.llm_backend import convert_messages, extract_usage
from tee_gateway.model_registry import get_model_config, get_rate_card
from tee_gateway.tee_manager import (
    TEEKeyManager,
    compute_ohttp_config_hash,
    compute_tee_msg_hash,
)


# ---------------------------------------------------------------------------
# TEEKeyManager
# ---------------------------------------------------------------------------


class TestTEEKeyManager(unittest.TestCase):
    """RSA key generation and signing are the core TEE trust primitive.

    register=False skips the nitriding network call so the tests are
    fully self-contained with no external dependencies.
    """

    def setUp(self):
        self.tee = TEEKeyManager(register=False)

    # --- Key generation ---

    def test_rsa_key_pair_generated(self):
        self.assertIsNotNone(self.tee.private_key)
        self.assertIsNotNone(self.tee.public_key)

    def test_public_key_pem_format(self):
        pem = self.tee.get_public_key()
        self.assertIn("-----BEGIN PUBLIC KEY-----", pem)
        self.assertIn("-----END PUBLIC KEY-----", pem)

    # --- tee_id derivation ---

    def test_tee_id_is_64_hex_chars(self):
        """tee_id is the hex-encoded 32-byte keccak256 hash."""
        tee_id = self.tee.get_tee_id()
        self.assertEqual(len(tee_id), 64)
        int(tee_id, 16)  # raises ValueError if not valid hex

    def test_tee_id_matches_keccak256_of_der_public_key(self):
        """tee_id = keccak256(DER public key) — must match the on-chain derivation.

        External verifiers (and the smart contract) compute the tee_id by:
          1. Stripping PEM headers and base64-decoding the body to get the DER bytes.
          2. keccak256-hashing those bytes.
        This test confirms the server uses exactly the same derivation.
        """
        pem = self.tee.get_public_key()
        der = base64.b64decode("".join(pem.strip().splitlines()[1:-1]))
        expected = keccak(der).hex()
        self.assertEqual(self.tee.get_tee_id(), expected)

    # --- Wallet address ---

    def test_wallet_address_is_valid_ethereum(self):
        addr = self.tee.get_wallet_address()
        self.assertTrue(addr.startswith("0x"), f"Expected 0x prefix, got {addr!r}")
        self.assertEqual(len(addr), 42, f"Expected 42-char address, got {len(addr)}")

    # --- RSA-PSS signing ---

    def test_sign_data_returns_base64(self):
        sig = self.tee.sign_data(b"hello")
        decoded = base64.b64decode(sig)  # raises if not valid base64
        self.assertGreater(len(decoded), 0)

    def test_signature_verifies_against_public_key(self):
        """The signature produced by sign_data must pass RSA-PSS verification.

        This is the same verification a client would perform after receiving a
        signed response. Salt length is fixed at 32 bytes (SHA-256 digest size)
        to match the on-chain verifier's expectations.
        """
        data = b"test payload for signing"
        sig_b64 = self.tee.sign_data(data)
        sig = base64.b64decode(sig_b64)
        # Raises cryptography.exceptions.InvalidSignature on failure
        self.tee.public_key.verify(
            sig,
            data,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
            hashes.SHA256(),
        )

    def test_wrong_data_fails_signature_verification(self):
        """A signature over one payload must not verify against a different payload."""
        from cryptography.exceptions import InvalidSignature

        sig = base64.b64decode(self.tee.sign_data(b"correct data"))
        with self.assertRaises(InvalidSignature):
            self.tee.public_key.verify(
                sig,
                b"tampered data",
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
                hashes.SHA256(),
            )

    def test_pss_salt_is_random_so_signatures_differ(self):
        """RSA-PSS uses a random salt, so signing the same message twice yields
        different ciphertext — but both must verify successfully."""
        sig1 = base64.b64decode(self.tee.sign_data(b"same msg"))
        sig2 = base64.b64decode(self.tee.sign_data(b"same msg"))
        self.assertNotEqual(
            sig1, sig2, "PSS salt should make signatures non-deterministic"
        )
        for sig in (sig1, sig2):
            self.tee.public_key.verify(
                sig,
                b"same msg",
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
                hashes.SHA256(),
            )

    # --- Attestation document ---

    def test_attestation_document_has_required_fields(self):
        doc = self.tee.get_attestation_document()
        for field in (
            "public_key",
            "tee_id",
            "wallet_address",
            "timestamp",
            "enclave_info",
        ):
            self.assertIn(field, doc, f"Missing required attestation field: {field!r}")

    def test_attestation_document_tee_id_has_0x_prefix(self):
        doc = self.tee.get_attestation_document()
        self.assertTrue(doc["tee_id"].startswith("0x"))

    def test_attestation_document_platform_is_aws_nitro(self):
        doc = self.tee.get_attestation_document()
        self.assertEqual(doc["enclave_info"]["platform"], "aws-nitro")

    # --- Instance isolation ---

    def test_two_managers_have_distinct_keys_and_tee_ids(self):
        """Each TEEKeyManager instance generates its own independent key pair."""
        other = TEEKeyManager(register=False)
        self.assertNotEqual(self.tee.get_tee_id(), other.get_tee_id())
        self.assertNotEqual(self.tee.get_wallet_address(), other.get_wallet_address())
        self.assertNotEqual(self.tee.get_public_key(), other.get_public_key())


# ---------------------------------------------------------------------------
# compute_tee_msg_hash
# ---------------------------------------------------------------------------


class TestComputeTEEMsgHash(unittest.TestCase):
    """The message hash links the request, response, and timestamp together.

    It must match the on-chain verifier's computation:
      keccak256(keccak256(request_bytes) || keccak256(response_bytes) || timestamp_uint256)
    """

    def _hash(self, req=b"request", resp="response", ts=1_000_000):
        return compute_tee_msg_hash(req, resp, ts)

    def test_returns_32_byte_hash(self):
        msg_hash, _, _ = self._hash()
        self.assertIsInstance(msg_hash, bytes)
        self.assertEqual(len(msg_hash), 32)

    def test_input_and_output_hashes_are_hex_strings(self):
        _, inp_hex, out_hex = self._hash()
        self.assertEqual(len(inp_hex), 64)
        self.assertEqual(len(out_hex), 64)
        int(inp_hex, 16)  # valid hex
        int(out_hex, 16)

    def test_deterministic_for_same_inputs(self):
        h1, _, _ = self._hash()
        h2, _, _ = self._hash()
        self.assertEqual(h1, h2)

    def test_different_request_bytes_change_hash(self):
        h1, _, _ = compute_tee_msg_hash(b"request_A", "response", 1000)
        h2, _, _ = compute_tee_msg_hash(b"request_B", "response", 1000)
        self.assertNotEqual(h1, h2)

    def test_different_response_changes_hash(self):
        h1, _, _ = compute_tee_msg_hash(b"request", "response_A", 1000)
        h2, _, _ = compute_tee_msg_hash(b"request", "response_B", 1000)
        self.assertNotEqual(h1, h2)

    def test_different_timestamp_changes_hash(self):
        h1, _, _ = compute_tee_msg_hash(b"request", "response", 1000)
        h2, _, _ = compute_tee_msg_hash(b"request", "response", 1001)
        self.assertNotEqual(h1, h2)

    def test_input_hash_matches_keccak256_of_request(self):
        """The intermediate input_hash must equal keccak256(request_bytes)."""
        req = b"my request payload"
        _, inp_hex, _ = compute_tee_msg_hash(req, "response", 999)
        self.assertEqual(inp_hex, keccak(req).hex())

    def test_output_hash_matches_keccak256_of_response(self):
        """The intermediate output_hash must equal keccak256(response_bytes)."""
        resp = "my response text"
        _, _, out_hex = compute_tee_msg_hash(b"request", resp, 999)
        self.assertEqual(out_hex, keccak(resp.encode("utf-8")).hex())


# ---------------------------------------------------------------------------
# compute_ohttp_config_hash — must byte-match TEERegistryV2.computeOHTTPConfigHash
# ---------------------------------------------------------------------------


class TestComputeOHTTPConfigHash(unittest.TestCase):
    """The OHTTP config hash binds the enclave's HPKE key to its attested RSA
    signing key. The signature returned by get_signed_hpke_config covers this
    hash, so it MUST be computed exactly the way the on-chain
    TEERegistryV2.computeOHTTPConfigHash does — otherwise every signature
    silently fails verification on-chain.

    The layout (EIP-712 style, every field a 32-byte word):
      keccak256(
          keccak256("OPENGRADIENT_TEE_OHTTP_CONFIG_V1")  // domain
          || tee_id                                       // bytes32
          || uint256(key_id) || uint256(kem_id)
          || uint256(kdf_id) || uint256(aead_id)
          || keccak256(public_key)                        // bytes32
          || keccak256(key_config)                        // bytes32
      )
    """

    # Deterministic, non-random vectors so the golden hash below is stable.
    TEE_ID = bytes(range(32))
    PUBLIC_KEY = b"\x02" * 32
    KEY_CONFIG = ohttp.key_config(PUBLIC_KEY)

    def _hash(self):
        return compute_ohttp_config_hash(
            self.TEE_ID,
            ohttp.KEY_CONFIG_ID,
            ohttp.KEM_ID_X25519,
            ohttp.KDF_ID_HKDF_SHA256,
            ohttp.AEAD_ID_CHACHA20_POLY1305,
            self.PUBLIC_KEY,
            self.KEY_CONFIG,
        )

    def test_golden_vector(self):
        """Frozen expected hash for the fixed vectors above.

        IMPORTANT: this value must equal the output of
        TEERegistryV2.computeOHTTPConfigHash for the same inputs. If this test
        fails after an intentional encoding change, regenerate it from the
        deployed contract (do NOT just paste the new Python output) — the whole
        point is to catch off-chain/on-chain divergence.
        """
        expected = "f2a783a780198b72145ce4f4824c3de4993975e17ee8df1e59f3b5dd78a0f1d9"
        self.assertEqual(self._hash().hex(), expected)

    def test_matches_independent_reconstruction(self):
        """Re-derive the layout independently from the implementation, so a
        change to either side of the encoding is caught even if someone also
        updates the golden vector."""

        def word(v):
            return v.to_bytes(32, "big")

        domain = keccak(b"OPENGRADIENT_TEE_OHTTP_CONFIG_V1")
        expected = keccak(
            domain
            + self.TEE_ID
            + word(ohttp.KEY_CONFIG_ID)
            + word(ohttp.KEM_ID_X25519)
            + word(ohttp.KDF_ID_HKDF_SHA256)
            + word(ohttp.AEAD_ID_CHACHA20_POLY1305)
            + keccak(self.PUBLIC_KEY)
            + keccak(self.KEY_CONFIG)
        )
        self.assertEqual(self._hash(), expected)

    def test_returns_32_bytes(self):
        self.assertEqual(len(self._hash()), 32)

    def test_rejects_wrong_length_tee_id(self):
        with self.assertRaises(ValueError):
            compute_ohttp_config_hash(
                b"\x00" * 31,  # not 32 bytes
                ohttp.KEY_CONFIG_ID,
                ohttp.KEM_ID_X25519,
                ohttp.KDF_ID_HKDF_SHA256,
                ohttp.AEAD_ID_CHACHA20_POLY1305,
                self.PUBLIC_KEY,
                self.KEY_CONFIG,
            )

    def test_each_field_affects_the_hash(self):
        """Flipping any single field must change the digest — confirms no field
        is dropped from the preimage."""
        base = self._hash()
        variants = [
            (
                bytes(reversed(self.TEE_ID)),
                ohttp.KEY_CONFIG_ID,
                ohttp.KEM_ID_X25519,
                ohttp.KDF_ID_HKDF_SHA256,
                ohttp.AEAD_ID_CHACHA20_POLY1305,
                self.PUBLIC_KEY,
                self.KEY_CONFIG,
            ),
            (
                self.TEE_ID,
                ohttp.KEY_CONFIG_ID + 1,
                ohttp.KEM_ID_X25519,
                ohttp.KDF_ID_HKDF_SHA256,
                ohttp.AEAD_ID_CHACHA20_POLY1305,
                self.PUBLIC_KEY,
                self.KEY_CONFIG,
            ),
            (
                self.TEE_ID,
                ohttp.KEY_CONFIG_ID,
                ohttp.KEM_ID_X25519 + 1,
                ohttp.KDF_ID_HKDF_SHA256,
                ohttp.AEAD_ID_CHACHA20_POLY1305,
                self.PUBLIC_KEY,
                self.KEY_CONFIG,
            ),
            (
                self.TEE_ID,
                ohttp.KEY_CONFIG_ID,
                ohttp.KEM_ID_X25519,
                ohttp.KDF_ID_HKDF_SHA256 + 1,
                ohttp.AEAD_ID_CHACHA20_POLY1305,
                self.PUBLIC_KEY,
                self.KEY_CONFIG,
            ),
            (
                self.TEE_ID,
                ohttp.KEY_CONFIG_ID,
                ohttp.KEM_ID_X25519,
                ohttp.KDF_ID_HKDF_SHA256,
                ohttp.AEAD_ID_CHACHA20_POLY1305 + 1,
                self.PUBLIC_KEY,
                self.KEY_CONFIG,
            ),
            (
                self.TEE_ID,
                ohttp.KEY_CONFIG_ID,
                ohttp.KEM_ID_X25519,
                ohttp.KDF_ID_HKDF_SHA256,
                ohttp.AEAD_ID_CHACHA20_POLY1305,
                b"\x03" * 32,
                self.KEY_CONFIG,
            ),
            (
                self.TEE_ID,
                ohttp.KEY_CONFIG_ID,
                ohttp.KEM_ID_X25519,
                ohttp.KDF_ID_HKDF_SHA256,
                ohttp.AEAD_ID_CHACHA20_POLY1305,
                self.PUBLIC_KEY,
                self.KEY_CONFIG + b"\x00",
            ),
        ]
        for v in variants:
            self.assertNotEqual(compute_ohttp_config_hash(*v), base)


class TestSignedHPKEConfig(unittest.TestCase):
    """get_signed_hpke_config must return a signature that verifies against the
    enclave's attested RSA key over the OHTTP config hash — this is the binding
    a relay/registry relies on now that the HPKE key is no longer in the
    nitriding attestation transcript."""

    def setUp(self):
        self.tee = TEEKeyManager(register=False)

    def test_signature_verifies_over_config_hash(self):
        cfg = self.tee.get_signed_hpke_config()

        # Recompute the hash the same way an off-chain verifier would.
        config_hash = compute_ohttp_config_hash(
            bytes.fromhex(self.tee.tee_id),
            cfg["key_id"],
            cfg["kem_id"],
            cfg["kdf_id"],
            cfg["aead_id"],
            bytes.fromhex(cfg["public_key"]),
            base64.b64decode(cfg["key_config"]),
        )
        self.assertEqual(cfg["signature_hash"], f"0x{config_hash.hex()}")

        # The signature must verify against the attested signing key.
        self.tee.public_key.verify(
            base64.b64decode(cfg["signature"]),
            config_hash,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
            hashes.SHA256(),
        )

    def test_tee_id_field_has_0x_prefix(self):
        cfg = self.tee.get_signed_hpke_config()
        self.assertEqual(cfg["tee_id"], f"0x{self.tee.tee_id}")


# ---------------------------------------------------------------------------
# model_registry
# ---------------------------------------------------------------------------


class TestModelRegistry(unittest.TestCase):
    """Model routing is the gateway's core value: it maps user-facing names to
    the right provider and determines per-token pricing for x402 payments."""

    def test_all_four_providers_reachable(self):
        providers = {
            get_model_config("gpt-4.1").provider,
            get_model_config("claude-sonnet-4-5").provider,
            get_model_config("gemini-2.5-flash").provider,
            get_model_config("grok-4").provider,
        }
        self.assertEqual(providers, {"openai", "anthropic", "google", "x-ai"})

    def test_openai_model_lookup(self):
        cfg = get_model_config("gpt-4.1")
        self.assertEqual(cfg.provider, "openai")
        self.assertIsNotNone(cfg.api_name)

    def test_anthropic_model_lookup(self):
        cfg = get_model_config("claude-sonnet-4-5")
        self.assertEqual(cfg.provider, "anthropic")

    def test_google_model_lookup(self):
        cfg = get_model_config("gemini-2.5-flash")
        self.assertEqual(cfg.provider, "google")

    def test_xai_model_lookup(self):
        cfg = get_model_config("grok-4")
        self.assertEqual(cfg.provider, "x-ai")

    def test_unknown_model_raises_value_error(self):
        with self.assertRaises(ValueError):
            get_model_config("definitely-not-a-real-model-xyz")

    def test_pricing_values_are_positive(self):
        cfg = get_model_config("gpt-4.1")
        self.assertGreater(cfg.input_price_usd, 0)
        self.assertGreater(cfg.output_price_usd, 0)

    def test_get_rate_card_structure(self):
        rates = get_rate_card("claude-sonnet-4-5")
        self.assertIn("input", rates)
        self.assertIn("output", rates)
        self.assertGreater(rates["input"], 0)
        self.assertGreater(rates["output"], 0)

    def test_model_alias_and_dated_name_resolve_identically(self):
        """Short alias and full dated name must route to the same provider and api_name."""
        cfg_short = get_model_config("gpt-4.1")
        cfg_dated = get_model_config("gpt-4.1-2025-04-14")
        self.assertEqual(cfg_short.provider, cfg_dated.provider)
        self.assertEqual(cfg_short.api_name, cfg_dated.api_name)

    def test_models_with_force_temperature_have_positive_value(self):
        """Any model that forces a temperature must set it to a positive float.
        We don't pin the exact value — just confirm the field is plausible if set."""
        cfg = get_model_config("o4-mini")
        if cfg.force_temperature is not None:
            self.assertGreater(cfg.force_temperature, 0)


# ---------------------------------------------------------------------------
# llm_backend.convert_messages
# ---------------------------------------------------------------------------


class TestConvertMessages(unittest.TestCase):
    """convert_messages translates OpenAI-format message dicts into LangChain
    message objects that are sent to each provider. Incorrect conversion would
    silently corrupt every request."""

    def test_user_message(self):
        result = convert_messages([{"role": "user", "content": "Hello"}])
        self.assertIsInstance(result[0], HumanMessage)
        self.assertEqual(result[0].content, "Hello")

    def test_system_message(self):
        result = convert_messages([{"role": "system", "content": "Be concise"}])
        self.assertIsInstance(result[0], SystemMessage)
        self.assertEqual(result[0].content, "Be concise")

    def test_assistant_message(self):
        result = convert_messages([{"role": "assistant", "content": "Sure!"}])
        self.assertIsInstance(result[0], AIMessage)
        self.assertEqual(result[0].content, "Sure!")

    def test_tool_result_message(self):
        result = convert_messages(
            [
                {
                    "role": "tool",
                    "content": '{"temperature": 72}',
                    "tool_call_id": "call_abc123",
                }
            ]
        )
        self.assertIsInstance(result[0], ToolMessage)
        self.assertEqual(result[0].tool_call_id, "call_abc123")
        self.assertEqual(result[0].content, '{"temperature": 72}')

    def test_assistant_message_with_tool_calls(self):
        """Tool call arguments must be JSON-parsed from the string form."""
        result = convert_messages(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "San Francisco"}',
                            },
                        }
                    ],
                }
            ]
        )
        self.assertIsInstance(result[0], AIMessage)
        self.assertEqual(len(result[0].tool_calls), 1)
        tc = result[0].tool_calls[0]
        self.assertEqual(tc["name"], "get_weather")
        self.assertEqual(tc["args"], {"location": "San Francisco"})
        self.assertEqual(tc["id"], "call_1")

    def test_multi_turn_order_preserved(self):
        msgs = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User question"},
            {"role": "assistant", "content": "Assistant reply"},
        ]
        result = convert_messages(msgs)
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], SystemMessage)
        self.assertIsInstance(result[1], HumanMessage)
        self.assertIsInstance(result[2], AIMessage)

    def test_user_content_as_list_of_parts(self):
        """Multimodal content parts should be concatenated into a single string."""
        result = convert_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello "},
                        {"type": "text", "text": "world"},
                    ],
                }
            ]
        )
        self.assertIsInstance(result[0], HumanMessage)
        self.assertEqual(result[0].content, "Hello world")

    def test_full_tool_call_conversation(self):
        """End-to-end multi-turn with tool use: user → assistant (tool call) → tool result."""
        msgs = [
            {"role": "user", "content": "What's the weather in NYC?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_xyz",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "NYC"}',
                        },
                    }
                ],
            },
            {"role": "tool", "content": '{"temp": 68}', "tool_call_id": "call_xyz"},
        ]
        result = convert_messages(msgs)
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], HumanMessage)
        self.assertIsInstance(result[1], AIMessage)
        self.assertEqual(len(result[1].tool_calls), 1)
        self.assertIsInstance(result[2], ToolMessage)
        self.assertEqual(result[2].tool_call_id, "call_xyz")


# ---------------------------------------------------------------------------
# llm_backend.extract_usage
# ---------------------------------------------------------------------------


class TestExtractUsage(unittest.TestCase):
    """Token usage feeds the dynamic x402 payment amount — wrong values mean
    under- or over-charging."""

    def test_extracts_all_token_counts(self):
        mock_resp = type(
            "R",
            (),
            {
                "usage_metadata": {
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 30,
                }
            },
        )()
        usage = extract_usage(mock_resp)
        self.assertEqual(usage["prompt_tokens"], 10)
        self.assertEqual(usage["completion_tokens"], 20)
        self.assertEqual(usage["total_tokens"], 30)

    def test_returns_none_when_usage_metadata_is_none(self):
        mock_resp = type("R", (), {"usage_metadata": None})()
        self.assertIsNone(extract_usage(mock_resp))

    def test_returns_none_when_attribute_missing(self):
        mock_resp = type("R", (), {})()
        self.assertIsNone(extract_usage(mock_resp))


if __name__ == "__main__":
    unittest.main()
