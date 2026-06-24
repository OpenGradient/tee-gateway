"""Tests for endpoint-based image generation (xAI Grok, ByteDance Seedream,
ByteDance Seedance, Z.ai GLM-Image).

Unlike Gemini's inline-image chat models (see test_image_billing.py), these
models are served via a dedicated OpenAI-compatible ``/images/generations``
endpoint and billed a flat price per generated image. These tests pin:

  1. The request/response handling in ``generate_images`` (b64_json -> data URI,
     n clamping, hosted-URL fetch -> data URI, provider-specific payloads).
  2. The flat per-image billing in ``compute_session_cost``.

No network or API key required — the provider HTTP client is mocked, the URL
fetch is patched, and a stub price feed is injected.
"""

import unittest
from decimal import Decimal
from unittest.mock import MagicMock, patch

from tee_gateway import image_generation, llm_backend
from tee_gateway.image_generation import generate_images
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

    def test_hosted_url_is_fetched_and_inlined(self):
        # Providers that return a hosted URL instead of inline bytes are fetched
        # into the enclave so the client always receives a data: URI.
        client = MagicMock()
        client.post.return_value = _mock_response([{"url": "https://img/1.jpg"}])
        with (
            patch.object(llm_backend, "bytedance_http_client", client),
            patch.object(
                image_generation,
                "_fetch_url_as_data_uri",
                return_value="data:image/jpeg;base64,RkVUQ0hFRA==",
            ) as fetch,
        ):
            images, count = generate_images(SEEDREAM, "a blue sphere", n=1)

        self.assertEqual(count, 1)
        self.assertEqual(images, ["data:image/jpeg;base64,RkVUQ0hFRA=="])
        fetch.assert_called_once_with("https://img/1.jpg")

    def test_zai_glm_image_uses_documented_payload_and_fetches_url(self):
        client = MagicMock()
        client.post.return_value = _mock_response([{"url": "https://z.ai/img.png"}])
        with (
            patch.object(llm_backend, "zai_http_client", client),
            patch.object(
                image_generation,
                "_fetch_url_as_data_uri",
                return_value="data:image/png;base64,RkVUQ0hFRA==",
            ),
        ):
            images, count = generate_images(GLM_IMAGE, "a poster", n=3)

        self.assertEqual(count, 1)
        self.assertEqual(images, ["data:image/png;base64,RkVUQ0hFRA=="])

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
        with (
            patch.object(llm_backend, "bytedance_http_client", client),
            patch.object(
                image_generation,
                "_fetch_url_as_data_uri",
                return_value="data:image/jpeg;base64,RkVUQ0hFRA==",
            ),
        ):
            images, count = generate_images(SEEDANCE, "a black hole", n=1)

        self.assertEqual(count, 1)
        self.assertEqual(images, ["data:image/jpeg;base64,RkVUQ0hFRA=="])

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

    def test_seedance_forwards_single_reference_image(self):
        client = MagicMock()
        client.post.return_value = _mock_response([{"url": "https://cdn/edited.jpg"}])
        with (
            patch.object(llm_backend, "bytedance_http_client", client),
            patch.object(
                image_generation,
                "_fetch_url_as_data_uri",
                return_value="data:image/jpeg;base64,RkVUQ0hFRA==",
            ),
        ):
            generate_images(
                SEEDANCE,
                "add a hat",
                n=1,
                reference_images=["https://cdn/original.jpg"],
            )

        payload = client.post.call_args.kwargs["json"]
        # A single reference is sent as a bare string (Seedream/Seedance accept
        # either a string or an array for the `image` field).
        self.assertEqual(payload["image"], "https://cdn/original.jpg")

    def test_seedream_forwards_multiple_reference_images_as_array(self):
        client = MagicMock()
        client.post.return_value = _mock_response([{"b64_json": "x"}])
        refs = ["data:image/png;base64,AAA", "https://cdn/b.jpg"]
        with patch.object(llm_backend, "bytedance_http_client", client):
            generate_images(SEEDREAM, "fuse these", n=1, reference_images=refs)

        payload = client.post.call_args.kwargs["json"]
        self.assertEqual(payload["image"], refs)

    def test_reference_images_clamped_to_ten(self):
        client = MagicMock()
        client.post.return_value = _mock_response([{"b64_json": "x"}])
        refs = [f"https://cdn/{i}.jpg" for i in range(15)]
        with patch.object(llm_backend, "bytedance_http_client", client):
            generate_images(SEEDREAM, "p", n=1, reference_images=refs)

        payload = client.post.call_args.kwargs["json"]
        self.assertEqual(len(payload["image"]), 10)

    def test_reference_images_ignored_for_non_bytedance(self):
        # xAI/Z.ai text-to-image endpoints don't support image edit; the `image`
        # field must not leak into their payloads.
        client = MagicMock()
        client.post.return_value = _mock_response([{"b64_json": "x"}])
        with patch.object(llm_backend, "xai_http_client", client):
            generate_images(
                GROK_IMAGE, "p", n=1, reference_images=["https://cdn/x.jpg"]
            )
        self.assertNotIn("image", client.post.call_args.kwargs["json"])

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


class TestFetchUrlAsDataUri(unittest.TestCase):
    """The hosted-URL fetch that inlines provider images into the enclave."""

    def test_encodes_bytes_with_content_type(self):
        fetch_client = MagicMock()
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.headers = {"content-type": "image/png; charset=binary"}
        resp.content = b"hello"
        fetch_client.get.return_value = resp

        with patch.object(image_generation, "_image_fetch_client", fetch_client):
            uri = image_generation._fetch_url_as_data_uri("https://cdn/x.png")

        # base64("hello") == "aGVsbG8=", mime taken from content-type (params dropped)
        self.assertEqual(uri, "data:image/png;base64,aGVsbG8=")
        fetch_client.get.assert_called_once_with("https://cdn/x.png")

    def test_defaults_mime_when_header_missing(self):
        fetch_client = MagicMock()
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.headers = {}
        resp.content = b"hello"
        fetch_client.get.return_value = resp

        with patch.object(image_generation, "_image_fetch_client", fetch_client):
            uri = image_generation._fetch_url_as_data_uri("https://cdn/x")

        self.assertEqual(uri, "data:image/jpeg;base64,aGVsbG8=")


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
