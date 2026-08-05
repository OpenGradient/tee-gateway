"""Tests for endpoint-based image generation (OpenAI gpt-image, xAI Grok,
ByteDance Seedream, ByteDance Seedance, Z.ai GLM-Image).

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

import httpx

from tee_gateway import image_generation, llm_backend
from tee_gateway.image_generation import generate_images
from tee_gateway.model_registry import get_model_config
from tee_gateway.price_feed import get_price_feed, set_price_feed
from tee_gateway.pricing import compute_session_cost

GROK_IMAGE = "grok-2-image"
SEEDREAM = "seedream-4.0"
SEEDREAM_5_LITE = "seedream-5.0-lite"
SEEDANCE = "seedance-4.5"
SEEDANCE_5 = "seedance-5.0"
GLM_IMAGE = "glm-image"
GPT_IMAGE = "gpt-image-2"


def _mock_response(data: list[dict]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"data": data}
    return resp


class TestGenerateImages(unittest.TestCase):
    """Request shaping and response parsing for the images endpoint."""

    def test_xai_hosted_url_becomes_data_uri(self):
        client = MagicMock()
        client.post.return_value = _mock_response([{"url": "https://x.ai/image.jpg"}])
        with (
            patch.object(llm_backend, "xai_http_client", client),
            patch.object(
                image_generation,
                "_fetch_url_as_data_uri",
                return_value="data:image/jpeg;base64,aGVsbG8=",
            ) as fetch,
        ):
            images, count = generate_images(GROK_IMAGE, "a red cube", n=2)

        self.assertEqual(count, 1)
        self.assertEqual(images, ["data:image/jpeg;base64,aGVsbG8="])
        fetch.assert_called_once_with("https://x.ai/image.jpg")

        # Verify the outgoing request shape.
        _, kwargs = client.post.call_args
        payload = kwargs["json"]
        self.assertEqual(payload["model"], get_model_config(GROK_IMAGE).api_name)
        self.assertEqual(payload["prompt"], "a red cube")
        self.assertEqual(payload["n"], 2)
        self.assertEqual(payload["response_format"], "url")

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

    def test_openai_gpt_image_omits_response_format_and_pins_size_quality(self):
        # gpt-image models always return base64 and reject `response_format`, so
        # the field must be omitted; size/quality are pinned for predictable
        # billing. The shared openai_http_client is reused (base_url ends in /v1,
        # so the request lands on OpenAI's /v1/images/generations).
        client = MagicMock()
        client.post.return_value = _mock_response([{"b64_json": "aGVsbG8="}])
        with patch.object(llm_backend, "openai_http_client", client):
            images, count = generate_images(GPT_IMAGE, "a red cube", n=1)

        self.assertEqual(count, 1)
        self.assertEqual(images, ["data:image/jpeg;base64,aGVsbG8="])

        payload = client.post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], get_model_config(GPT_IMAGE).api_name)
        self.assertEqual(payload["prompt"], "a red cube")
        self.assertEqual(payload["n"], 1)
        self.assertEqual(payload["size"], "1024x1024")
        self.assertEqual(payload["quality"], "medium")
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

    def test_seedream_5_lite_uses_ep_deployment_params(self):
        # Seedream 5.0 Lite is an ep- deployment endpoint and must use the same
        # URL/no-n/seedance-style payload as Seedance — a regression guard, since
        # this used to be auto-detected from the "ep-" api_name prefix and is now
        # driven by explicit registry fields.
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
            images, count = generate_images(SEEDREAM_5_LITE, "a koi pond", n=1)

        self.assertEqual(images, ["data:image/jpeg;base64,RkVUQ0hFRA=="])
        payload = client.post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], get_model_config(SEEDREAM_5_LITE).api_name)
        self.assertEqual(payload["response_format"], "url")
        self.assertEqual(payload["sequential_image_generation"], "disabled")
        self.assertFalse(payload["watermark"])
        self.assertEqual(payload["size"], "2K")
        self.assertFalse(payload["stream"])
        self.assertNotIn("n", payload)

    def test_seedance_5_omits_sequential_image_generation(self):
        # The Seedance 5.0 deployment endpoint rejects sequential_image_generation
        # with HTTP 400 ("not supported by the current model"), so its payload is
        # the shared ep- shape minus that field.
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
            images, count = generate_images(SEEDANCE_5, "a black hole", n=1)

        self.assertEqual(count, 1)
        self.assertEqual(images, ["data:image/jpeg;base64,RkVUQ0hFRA=="])
        payload = client.post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], get_model_config(SEEDANCE_5).api_name)
        self.assertEqual(payload["response_format"], "url")
        self.assertNotIn("sequential_image_generation", payload)
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

    def test_gpt_image_edits_uploads_references_as_multipart(self):
        # gpt-image reference edits go to /images/edits as multipart image[]
        # file uploads (not the JSON generations path), with n/size/quality as
        # form fields and no response_format (gpt-image rejects it).
        client = MagicMock()
        client.post.return_value = _mock_response([{"b64_json": "aGVsbG8="}])
        refs = [
            "data:image/png;base64,QUJD",  # "ABC"
            "data:image/jpeg;base64,REVG",  # "DEF"
        ]
        with patch.object(llm_backend, "openai_http_client", client):
            images, count = generate_images(
                GPT_IMAGE, "add the logo to the photo", n=1, reference_images=refs
            )

        self.assertEqual(count, 1)
        self.assertEqual(images, ["data:image/jpeg;base64,aGVsbG8="])

        args, kwargs = client.post.call_args
        self.assertEqual(args[0], "/images/edits")
        # No JSON body on the multipart path.
        self.assertNotIn("json", kwargs)
        form = kwargs["data"]
        self.assertEqual(form["model"], get_model_config(GPT_IMAGE).api_name)
        self.assertEqual(form["prompt"], "add the logo to the photo")
        self.assertEqual(form["n"], "1")
        self.assertEqual(form["size"], "1024x1024")
        self.assertEqual(form["quality"], "medium")
        self.assertNotIn("response_format", form)
        # Both references are uploaded under the repeated image[] field, decoded
        # back to their raw bytes with a mime-appropriate filename.
        uploads = kwargs["files"]
        self.assertEqual([field for field, _ in uploads], ["image[]", "image[]"])
        self.assertEqual(uploads[0][1][0], "image_0.png")
        self.assertEqual(uploads[0][1][1], b"ABC")
        self.assertEqual(uploads[0][1][2], "image/png")
        self.assertEqual(uploads[1][1][0], "image_1.jpg")
        self.assertEqual(uploads[1][1][1], b"DEF")

    def test_gpt_image_without_references_uses_generations(self):
        # No references -> plain text-to-image on the JSON generations endpoint.
        client = MagicMock()
        client.post.return_value = _mock_response([{"b64_json": "aGVsbG8="}])
        with patch.object(llm_backend, "openai_http_client", client):
            generate_images(GPT_IMAGE, "a red cube", n=1)

        args, kwargs = client.post.call_args
        self.assertEqual(args[0], "/images/generations")
        self.assertIn("json", kwargs)
        self.assertNotIn("files", kwargs)
        self.assertNotIn("image", kwargs["json"])

    def test_gpt_image_non_inline_references_fall_back_to_generation(self):
        # A plain URL reference can't be uploaded (we won't dereference client
        # URLs in the enclave); with nothing uploadable, fall back to a plain
        # generation rather than sending an empty edit request.
        client = MagicMock()
        client.post.return_value = _mock_response([{"b64_json": "aGVsbG8="}])
        with patch.object(llm_backend, "openai_http_client", client):
            generate_images(GPT_IMAGE, "p", n=1, reference_images=["https://cdn/x.jpg"])

        args, kwargs = client.post.call_args
        self.assertEqual(args[0], "/images/generations")
        self.assertNotIn("files", kwargs)
        self.assertNotIn("image", kwargs["json"])

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


class TestProviderErrorDetail(unittest.TestCase):
    """Provider error bodies must surface in the raised error message.

    httpx's default HTTPStatusError message is just the status and URL, which
    hides the provider's stated rejection reason (invalid parameter, prompt
    limit, content filter, …) — the only way to diagnose a 400 like ARK's
    /images/generations rejections.
    """

    @staticmethod
    def _response(status: int, *, content: bytes = b"", json_body=None):
        import httpx

        request = httpx.Request(
            "POST", "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"
        )
        if json_body is not None:
            return httpx.Response(status, json=json_body, request=request)
        return httpx.Response(status, content=content, request=request)

    def test_generate_images_surfaces_provider_error_body(self):
        import httpx

        body = {
            "error": {
                "code": "InvalidParameter",
                "message": "The request parameter `prompt` exceeds the limit.",
            }
        }
        client = MagicMock()
        client.post.return_value = self._response(400, json_body=body)
        with patch.object(llm_backend, "bytedance_http_client", client):
            with self.assertRaises(httpx.HTTPStatusError) as ctx:
                generate_images(SEEDREAM_5_LITE, "p", n=1)

        msg = str(ctx.exception)
        self.assertIn("400", msg)
        self.assertIn("InvalidParameter", msg)
        self.assertIn("exceeds the limit", msg)
        # The MDN-link footer from httpx's default message is dropped.
        self.assertNotIn("developer.mozilla.org", msg)

    def test_detail_falls_back_to_raw_body_for_non_json(self):
        detail = image_generation._provider_error_detail(
            self._response(400, content=b"plain text failure")
        )
        self.assertEqual(detail, "plain text failure")

    def test_detail_handles_string_error_field(self):
        detail = image_generation._provider_error_detail(
            self._response(400, json_body={"error": "quota exhausted"})
        )
        self.assertEqual(detail, "quota exhausted")

    def test_detail_handles_empty_body(self):
        detail = image_generation._provider_error_detail(self._response(400))
        self.assertEqual(detail, "(empty response body)")

    def test_detail_is_truncated(self):
        detail = image_generation._provider_error_detail(
            self._response(400, content=b"x" * 10_000)
        )
        self.assertEqual(len(detail), image_generation._MAX_ERROR_DETAIL_CHARS)

    def test_url_fetch_surfaces_error_body(self):
        import httpx

        resp = self._response(403, content=b"Access denied by CDN policy")
        ctx_mgr = MagicMock()
        ctx_mgr.__enter__.return_value = resp
        ctx_mgr.__exit__.return_value = False
        client = MagicMock()
        client.stream.return_value = ctx_mgr
        with patch.object(image_generation, "_image_fetch_client", client):
            with self.assertRaises(httpx.HTTPStatusError) as ctx:
                image_generation._fetch_url_as_data_uri("https://cdn/x.png")

        self.assertIn("Access denied by CDN policy", str(ctx.exception))


def _mock_stream_client(headers: dict, chunks: list[bytes]) -> MagicMock:
    """A fake httpx client whose .stream(...) yields a response with these bytes."""
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.headers = headers
    resp.iter_bytes.return_value = chunks
    ctx = MagicMock()
    ctx.__enter__.return_value = resp
    ctx.__exit__.return_value = False
    client = MagicMock()
    client.stream.return_value = ctx
    return client


def _mock_stream_context(response) -> MagicMock:
    context = MagicMock()
    context.__enter__.return_value = response
    context.__exit__.return_value = False
    return context


class TestFetchUrlAsDataUri(unittest.TestCase):
    """The hosted-URL fetch that inlines provider images into the enclave."""

    def test_encodes_bytes_with_content_type(self):
        client = _mock_stream_client(
            {"content-type": "image/png; charset=binary"}, [b"hel", b"lo"]
        )
        with patch.object(image_generation, "_image_fetch_client", client):
            uri = image_generation._fetch_url_as_data_uri("https://cdn/x.png")

        # base64("hello") == "aGVsbG8=", mime taken from content-type (params dropped)
        self.assertEqual(uri, "data:image/png;base64,aGVsbG8=")
        client.stream.assert_called_once_with("GET", "https://cdn/x.png")

    def test_defaults_mime_when_header_missing(self):
        client = _mock_stream_client({}, [b"hello"])
        with patch.object(image_generation, "_image_fetch_client", client):
            uri = image_generation._fetch_url_as_data_uri("https://cdn/x")

        self.assertEqual(uri, "data:image/jpeg;base64,aGVsbG8=")

    def test_retries_404_until_provider_image_is_available(self):
        request = httpx.Request("GET", "https://cdn/x.png")
        not_found = httpx.Response(
            404,
            json={"RetCode": -148654, "ErrMsg": "file not exist"},
            request=request,
        )
        available = MagicMock()
        available.raise_for_status.return_value = None
        available.headers = {"content-type": "image/png"}
        available.iter_bytes.return_value = [b"hello"]

        client = MagicMock()
        client.stream.side_effect = [
            _mock_stream_context(not_found),
            _mock_stream_context(available),
        ]
        with (
            patch.object(image_generation, "_image_fetch_client", client),
            patch.object(image_generation.time, "sleep") as sleep,
        ):
            uri = image_generation._fetch_url_as_data_uri("https://cdn/x.png")

        self.assertEqual(uri, "data:image/png;base64,aGVsbG8=")
        self.assertEqual(client.stream.call_count, 2)
        sleep.assert_called_once_with(1.0)

    def test_persistent_404_stops_after_bounded_retries(self):
        request = httpx.Request("GET", "https://cdn/x.png")
        retry_delays = image_generation._FETCH_RETRY_DELAYS_SECONDS
        attempts = 1 + len(retry_delays)
        responses = [
            httpx.Response(
                404,
                json={"RetCode": -148654, "ErrMsg": "file not exist"},
                request=request,
            )
            for _ in range(attempts)
        ]
        client = MagicMock()
        client.stream.side_effect = [
            _mock_stream_context(response) for response in responses
        ]
        with (
            patch.object(image_generation, "_image_fetch_client", client),
            patch.object(image_generation.time, "sleep") as sleep,
        ):
            with self.assertRaises(httpx.HTTPStatusError) as ctx:
                image_generation._fetch_url_as_data_uri("https://cdn/x.png")

        self.assertIn("file not exist", str(ctx.exception))
        self.assertEqual(client.stream.call_count, attempts)
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list],
            list(retry_delays),
        )

    def test_rejects_non_http_scheme(self):
        # Validation happens before any client use, so no client is needed.
        with self.assertRaises(ValueError):
            image_generation._fetch_url_as_data_uri("ftp://cdn/x.png")
        with self.assertRaises(ValueError):
            image_generation._fetch_url_as_data_uri("file:///etc/passwd")

    def test_rejects_private_and_loopback_ip_hosts(self):
        for url in (
            "http://127.0.0.1/x.png",
            "http://169.254.169.254/latest/meta-data",  # cloud metadata
            "http://10.0.0.5/x.png",
            "http://[::1]/x.png",
        ):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    image_generation._fetch_url_as_data_uri(url)

    def test_rejects_body_exceeding_size_cap(self):
        client = _mock_stream_client(
            {"content-type": "image/png"}, [b"x" * 4, b"x" * 4]
        )
        with (
            patch.object(image_generation, "_image_fetch_client", client),
            patch.object(image_generation, "_MAX_IMAGE_BYTES", 5),
        ):
            with self.assertRaises(ValueError):
                image_generation._fetch_url_as_data_uri("https://cdn/big.png")

    def test_rejects_declared_content_length_over_cap(self):
        client = _mock_stream_client(
            {"content-type": "image/png", "content-length": "999"}, [b"x"]
        )
        with (
            patch.object(image_generation, "_image_fetch_client", client),
            patch.object(image_generation, "_MAX_IMAGE_BYTES", 5),
        ):
            with self.assertRaises(ValueError):
                image_generation._fetch_url_as_data_uri("https://cdn/big.png")


def _error_stream_ctx(status: int, body: bytes) -> MagicMock:
    """A .stream(...) context manager yielding a real httpx error response."""
    import httpx

    resp = httpx.Response(
        status, content=body, request=httpx.Request("GET", "https://cdn/x.png")
    )
    ctx = MagicMock()
    ctx.__enter__.return_value = resp
    ctx.__exit__.return_value = False
    return ctx


def _ok_stream_ctx(chunks: list[bytes]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.headers = {"content-type": "image/png"}
    resp.iter_bytes.return_value = chunks
    ctx = MagicMock()
    ctx.__enter__.return_value = resp
    ctx.__exit__.return_value = False
    return ctx


class TestFetchUrlRetry(unittest.TestCase):
    """The hosted-URL fetch retries CDN-visibility 404s and transient errors.

    Z.ai's generations response can point at a file mfile.z.ai hasn't made
    visible yet, so the immediate fetch 404s with "file not exist" even though
    the image exists moments later. A 404 on a provider-returned URL is
    therefore retried, not trusted.
    """

    def test_404_then_success_is_retried(self):
        client = MagicMock()
        client.stream.side_effect = [
            _error_stream_ctx(404, b'{"RetCode":-148654, "ErrMsg":"file not exist"}'),
            _ok_stream_ctx([b"hello"]),
        ]
        with (
            patch.object(image_generation, "_image_fetch_client", client),
            patch.object(image_generation.time, "sleep") as sleep,
        ):
            uri = image_generation._fetch_url_as_data_uri("https://cdn/x.png")

        self.assertEqual(uri, "data:image/png;base64,aGVsbG8=")
        self.assertEqual(client.stream.call_count, 2)
        sleep.assert_called_once_with(image_generation._FETCH_RETRY_DELAYS_SECONDS[0])

    def test_transport_error_then_success_is_retried(self):
        import httpx

        client = MagicMock()
        client.stream.side_effect = [
            httpx.ConnectError("connection reset"),
            _ok_stream_ctx([b"hello"]),
        ]
        with (
            patch.object(image_generation, "_image_fetch_client", client),
            patch.object(image_generation.time, "sleep"),
        ):
            uri = image_generation._fetch_url_as_data_uri("https://cdn/x.png")

        self.assertEqual(uri, "data:image/png;base64,aGVsbG8=")
        self.assertEqual(client.stream.call_count, 2)

    def test_non_retryable_status_fails_immediately(self):
        import httpx

        client = MagicMock()
        client.stream.return_value = _error_stream_ctx(403, b"Access denied")
        with (
            patch.object(image_generation, "_image_fetch_client", client),
            patch.object(image_generation.time, "sleep") as sleep,
        ):
            with self.assertRaises(httpx.HTTPStatusError):
                image_generation._fetch_url_as_data_uri("https://cdn/x.png")

        self.assertEqual(client.stream.call_count, 1)
        sleep.assert_not_called()

    def test_persistent_404_exhausts_retries_and_raises(self):
        import httpx

        attempts = 1 + len(image_generation._FETCH_RETRY_DELAYS_SECONDS)
        client = MagicMock()
        client.stream.side_effect = [
            _error_stream_ctx(404, b'{"ErrMsg":"file not exist"}')
            for _ in range(attempts)
        ]
        with (
            patch.object(image_generation, "_image_fetch_client", client),
            patch.object(image_generation.time, "sleep") as sleep,
        ):
            with self.assertRaises(httpx.HTTPStatusError) as ctx:
                image_generation._fetch_url_as_data_uri("https://cdn/x.png")

        self.assertIn("file not exist", str(ctx.exception))
        self.assertEqual(client.stream.call_count, attempts)
        self.assertEqual(
            [c.args[0] for c in sleep.call_args_list],
            list(image_generation._FETCH_RETRY_DELAYS_SECONDS),
        )

    def test_size_cap_violation_is_not_retried(self):
        # An oversized body is a hard failure, not a transient one.
        client = MagicMock()
        client.stream.return_value = _ok_stream_ctx([b"x" * 4, b"x" * 4])
        with (
            patch.object(image_generation, "_image_fetch_client", client),
            patch.object(image_generation, "_MAX_IMAGE_BYTES", 5),
            patch.object(image_generation.time, "sleep") as sleep,
        ):
            with self.assertRaises(ValueError):
                image_generation._fetch_url_as_data_uri("https://cdn/big.png")

        self.assertEqual(client.stream.call_count, 1)
        sleep.assert_not_called()


class TestExtractImageInputs(unittest.TestCase):
    """Prompt + reference-image extraction from the user turns."""

    @staticmethod
    def _human(content):
        from langchain_core.messages import HumanMessage

        return HumanMessage(content=content)

    def test_joins_text_across_turns_no_references(self):
        msgs = [self._human("a red cube"), self._human("make it blue")]
        prompt, refs = image_generation._extract_image_inputs(msgs)
        self.assertEqual(prompt, "a red cube\nmake it blue")
        self.assertEqual(refs, [])

    def test_mixed_text_and_image_does_not_splice_base64_into_prompt(self):
        # An image-to-image edit turn: text + an attached reference image. The
        # base64 blob must never leak into the prompt text.
        data_uri = "data:image/png;base64,QUJD"
        msgs = [
            self._human(
                [
                    {"type": "text", "text": "add a hat"},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ]
            )
        ]
        prompt, refs = image_generation._extract_image_inputs(msgs)
        self.assertEqual(prompt, "add a hat")
        self.assertEqual(refs, [data_uri])

    def test_extracts_langchain_image_block_with_base64_and_url(self):
        msgs = [
            self._human(
                [
                    {"type": "image", "base64": "QUJD", "mime_type": "image/webp"},
                    {"type": "image", "url": "https://cdn/ref.jpg"},
                ]
            )
        ]
        _, refs = image_generation._extract_image_inputs(msgs)
        self.assertEqual(refs, ["data:image/webp;base64,QUJD", "https://cdn/ref.jpg"])

    def test_only_latest_turn_references_are_returned(self):
        # An earlier edit turn carried an image; the latest turn carries a new
        # one. Only the latest turn's reference should be forwarded.
        msgs = [
            self._human(
                [
                    {"type": "text", "text": "first"},
                    {"type": "image_url", "image_url": {"url": "https://cdn/old.jpg"}},
                ]
            ),
            self._human(
                [
                    {"type": "text", "text": "second"},
                    {"type": "image_url", "image_url": {"url": "https://cdn/new.jpg"}},
                ]
            ),
        ]
        prompt, refs = image_generation._extract_image_inputs(msgs)
        self.assertEqual(prompt, "first\nsecond")
        self.assertEqual(refs, ["https://cdn/new.jpg"])

    def test_text_only_latest_turn_clears_stale_references(self):
        # Edit turn with an image, then a plain text follow-up: the text-only
        # latest turn means a fresh generation, so no stale reference rides along.
        msgs = [
            self._human(
                [
                    {"type": "text", "text": "first"},
                    {"type": "image_url", "image_url": {"url": "https://cdn/old.jpg"}},
                ]
            ),
            self._human("just text now"),
        ]
        _, refs = image_generation._extract_image_inputs(msgs)
        self.assertEqual(refs, [])

    def test_malformed_image_parts_are_ignored(self):
        msgs = [
            self._human(
                [
                    {"type": "text", "text": "p"},
                    {"type": "image_url", "image_url": {"url": None}},
                    {"type": "image_url", "image_url": 123},
                    {"type": "image", "base64": None},
                ]
            )
        ]
        prompt, refs = image_generation._extract_image_inputs(msgs)
        self.assertEqual(prompt, "p")
        self.assertEqual(refs, [])


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
        for model in (GROK_IMAGE, SEEDREAM, SEEDANCE, SEEDANCE_5, GLM_IMAGE, GPT_IMAGE):
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
