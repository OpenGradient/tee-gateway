"""Tests for endpoint-based image generation (xAI Grok, ByteDance Seedream,
ByteDance Seedance, Z.ai GLM-Image).

Unlike Gemini's inline-image chat models (see test_image_billing.py), these
models are served via a dedicated OpenAI-compatible ``/images/generations``
endpoint and billed a flat price per generated image. These tests pin:

  1. The request/response handling in ``generate_images`` (b64_json -> data URI,
     n clamping, url fallback, provider-specific payloads).
  2. The flat per-image billing in ``compute_session_cost``.

No network or API key required — the provider HTTP client is mocked and a stub
price feed is injected.
"""

import unittest
from decimal import Decimal
from unittest.mock import MagicMock, patch

from tee_gateway import llm_backend
from tee_gateway.llm_backend import generate_images
from tee_gateway.model_registry import get_model_config
from tee_gateway.price_feed import get_price_feed, set_price_feed
from tee_gateway.pricing import compute_session_cost

GROK_IMAGE = "grok-2-image"
SEEDREAM = "seedream-4.0"
SEEDANCE = "seedance-4.5"
GLM_IMAGE = "glm-image"


def _mock_response(data: list[dict]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"data": data}
    return resp


class TestGenerateImages(unittest.TestCase):
    """Request shaping and response parsing for the images endpoint."""

    def test_b64_json_becomes_data_uri(self):
        client = MagicMock()
        client.post.return_value = _mock_response(
            [{"b64_json": "aGVsbG8="}, {"b64_json": "d29ybGQ="}]
        )
        with patch.object(llm_backend, "xai_http_client", client):
            images, count = generate_images(GROK_IMAGE, "a red cube", n=2)

        self.assertEqual(count, 2)
        self.assertEqual(
            images,
            [
                "data:image/jpeg;base64,aGVsbG8=",
                "data:image/jpeg;base64,d29ybGQ=",
            ],
        )

        # Verify the outgoing request shape.
        _, kwargs = client.post.call_args
        payload = kwargs["json"]
        self.assertEqual(payload["model"], get_model_config(GROK_IMAGE).api_name)
        self.assertEqual(payload["prompt"], "a red cube")
        self.assertEqual(payload["n"], 2)
        self.assertEqual(payload["response_format"], "b64_json")

    def test_url_fallback_when_no_b64(self):
        client = MagicMock()
        client.post.return_value = _mock_response([{"url": "https://img/1.jpg"}])
        with patch.object(llm_backend, "bytedance_http_client", client):
            images, count = generate_images(SEEDREAM, "a blue sphere", n=1)

        self.assertEqual(count, 1)
        self.assertEqual(images, ["https://img/1.jpg"])

    def test_zai_glm_image_uses_documented_payload_and_url_response(self):
        client = MagicMock()
        client.post.return_value = _mock_response([{"url": "https://z.ai/img.png"}])
        with patch.object(llm_backend, "zai_http_client", client):
            images, count = generate_images(GLM_IMAGE, "a poster", n=3)

        self.assertEqual(count, 1)
        self.assertEqual(images, ["https://z.ai/img.png"])

        _, kwargs = client.post.call_args
        payload = kwargs["json"]
        self.assertEqual(payload["model"], "glm-image")
        self.assertEqual(payload["prompt"], "a poster")
        self.assertEqual(payload["size"], "1280x1280")
        self.assertNotIn("n", payload)
        self.assertNotIn("response_format", payload)

    def test_seedance_uses_url_format_and_extra_params(self):
        client = MagicMock()
        client.post.return_value = _mock_response([{"url": "https://cdn/img.jpg"}])
        with patch.object(llm_backend, "bytedance_http_client", client):
            images, count = generate_images(SEEDANCE, "a black hole", n=1)

        self.assertEqual(count, 1)
        self.assertEqual(images, ["https://cdn/img.jpg"])

        _, kwargs = client.post.call_args
        payload = kwargs["json"]
        self.assertEqual(payload["model"], get_model_config(SEEDANCE).api_name)
        self.assertEqual(payload["prompt"], "a black hole")
        self.assertEqual(payload["response_format"], "url")
        self.assertEqual(payload["sequential_image_generation"], "disabled")
        self.assertFalse(payload["watermark"])
        self.assertEqual(payload["size"], "2K")
        self.assertFalse(payload["stream"])
        self.assertNotIn("n", payload)

    def test_n_is_clamped_to_provider_range(self):
        client = MagicMock()
        client.post.return_value = _mock_response([{"b64_json": "x"}])
        with patch.object(llm_backend, "xai_http_client", client):
            generate_images(GROK_IMAGE, "p", n=999)
        self.assertEqual(client.post.call_args.kwargs["json"]["n"], 10)

        with patch.object(llm_backend, "xai_http_client", client):
            generate_images(GROK_IMAGE, "p", n=0)
        self.assertEqual(client.post.call_args.kwargs["json"]["n"], 1)

    def test_uninitialized_client_raises(self):
        with patch.object(llm_backend, "xai_http_client", None):
            with self.assertRaises(RuntimeError):
                generate_images(GROK_IMAGE, "p", n=1)

    def test_quality_maps_to_seedream_size_preset(self):
        client = MagicMock()
        client.post.return_value = _mock_response([{"b64_json": "x"}])
        for quality, expected in (("low", "1K"), ("medium", "2K"), ("high", "4K")):
            with self.subTest(quality=quality):
                with patch.object(llm_backend, "bytedance_http_client", client):
                    generate_images(SEEDREAM, "p", n=1, quality=quality)
                self.assertEqual(client.post.call_args.kwargs["json"]["size"], expected)

    def test_quality_maps_to_zai_pixel_dimensions(self):
        client = MagicMock()
        client.post.return_value = _mock_response([{"url": "https://z.ai/i.png"}])
        with patch.object(llm_backend, "zai_http_client", client):
            generate_images(GLM_IMAGE, "p", n=1, quality="high")
        self.assertEqual(client.post.call_args.kwargs["json"]["size"], "2048x2048")

    def test_quality_overrides_seedance_default_size(self):
        client = MagicMock()
        client.post.return_value = _mock_response([{"url": "https://cdn/i.jpg"}])
        with patch.object(llm_backend, "bytedance_http_client", client):
            generate_images(SEEDANCE, "p", n=1, quality="high")
        self.assertEqual(client.post.call_args.kwargs["json"]["size"], "4K")

    def test_no_quality_keeps_provider_defaults(self):
        client = MagicMock()
        client.post.return_value = _mock_response([{"b64_json": "x"}])
        # Seedream omits size entirely when no quality is requested.
        with patch.object(llm_backend, "bytedance_http_client", client):
            generate_images(SEEDREAM, "p", n=1)
        self.assertNotIn("size", client.post.call_args.kwargs["json"])
        # Z.ai falls back to its documented default.
        with patch.object(llm_backend, "zai_http_client", client):
            generate_images(GLM_IMAGE, "p", n=1)
        self.assertEqual(client.post.call_args.kwargs["json"]["size"], "1280x1280")

    def test_quality_ignored_for_grok_without_resolution_control(self):
        client = MagicMock()
        client.post.return_value = _mock_response([{"b64_json": "x"}])
        with patch.object(llm_backend, "xai_http_client", client):
            generate_images(GROK_IMAGE, "p", n=1, quality="high")
        self.assertNotIn("size", client.post.call_args.kwargs["json"])

    def test_invalid_quality_raises(self):
        client = MagicMock()
        client.post.return_value = _mock_response([{"b64_json": "x"}])
        with patch.object(llm_backend, "bytedance_http_client", client):
            with self.assertRaises(ValueError):
                generate_images(SEEDREAM, "p", n=1, quality="ultra")


class TestPerImageBilling(unittest.TestCase):
    """Flat per-image pricing, independent of token usage."""

    def setUp(self):
        try:
            self._prev_feed = get_price_feed()
        except RuntimeError:
            self._prev_feed = None
        # 1 OPG == 1 USD keeps the OPG<->USD reconciliation arithmetic trivial.
        stub = MagicMock()
        stub.get_price.return_value = Decimal("1")
        set_price_feed(stub)

    def tearDown(self):
        if self._prev_feed is not None:
            set_price_feed(self._prev_feed)

    @staticmethod
    def _zero_usage() -> dict:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def test_single_image_charged_flat_price(self):
        for model in (GROK_IMAGE, SEEDREAM, SEEDANCE, GLM_IMAGE):
            with self.subTest(model=model):
                cfg = get_model_config(model)
                cost = compute_session_cost(model, self._zero_usage(), image_count=1)
                self.assertIsNotNone(cost)
                self.assertAlmostEqual(cost.cost_usd, cfg.per_image_price_usd, places=9)

    def test_cost_scales_with_image_count(self):
        cfg = get_model_config(GROK_IMAGE)
        one = compute_session_cost(GROK_IMAGE, self._zero_usage(), image_count=1)
        three = compute_session_cost(GROK_IMAGE, self._zero_usage(), image_count=3)
        self.assertIsNotNone(one)
        self.assertIsNotNone(three)
        self.assertAlmostEqual(three.cost_usd, cfg.per_image_price_usd * 3, places=9)
        self.assertGreater(three.cost_opg, one.cost_opg)

    def test_zero_images_is_free(self):
        cost = compute_session_cost(GROK_IMAGE, self._zero_usage(), image_count=0)
        self.assertIsNotNone(cost)
        self.assertEqual(cost.cost_opg, 0)


if __name__ == "__main__":
    unittest.main()
