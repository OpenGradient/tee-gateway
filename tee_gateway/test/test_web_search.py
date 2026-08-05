"""
Unit tests for the dedicated in-enclave web search endpoint (Exa).

Covers:
  - web_search: argument coercion, Exa request shaping, result formatting, and
    every failure mode of the Exa call
  - pricing: the flat per-search cost (compute_web_search_cost) and that chat
    token pricing no longer carries a search surcharge
  - web_search_controller: request validation, signed response shape, the
    opengradient cost block, and the unbilled failure paths
  - ohttp_controller: the inner `endpoint` discriminator routes a sealed
    request to /v1/web_search (and defaults to chat for existing clients)
"""

import json
import unittest
from decimal import Decimal
from unittest.mock import Mock, patch

from flask import Flask

from tee_gateway import web_search as ws
from tee_gateway.controllers import ohttp_controller
from tee_gateway.controllers.web_search_controller import create_web_search
from tee_gateway.model_registry import WEB_SEARCH_PRICE_USD
from tee_gateway.pricing import compute_session_cost, compute_web_search_cost


# ---------------------------------------------------------------------------
# Exa test doubles
# ---------------------------------------------------------------------------


def _exa_result(url: str, title: str = "A title", text: str = "Some body text"):
    return {
        "title": title,
        "url": url,
        "publishedDate": "2026-03-04T10:00:00.000Z",
        "author": "An author",
        "id": url,
        "text": text,
    }


def _exa_response(status: int = 200, body: dict | None = None):
    """A stand-in for httpx.Response carrying just what the code reads."""
    response = Mock()
    response.status_code = status
    response.json.return_value = body if body is not None else {}
    response.text = json.dumps(body) if body is not None else ""
    response.reason_phrase = "OK" if status == 200 else "Error"
    return response


class _ExaClient:
    """Records the payloads posted to /search and replays queued responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads: list[dict] = []

    def post(self, path, json=None):  # noqa: A002 - matches httpx.Client.post
        self.payloads.append(json)
        return self.responses.pop(0) if self.responses else _exa_response(200, {})


def _with_exa(*responses):
    """Patch in a fake Exa client, returning it so payloads can be asserted."""
    client = _ExaClient(responses)
    return patch.object(ws, "_exa_http_client", client), client


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


class TestAvailability(unittest.TestCase):
    def test_availability_tracks_the_injected_key(self):
        ws.configure_exa_client("test-key")
        try:
            self.assertTrue(ws.web_search_available())
        finally:
            ws.configure_exa_client(None)
        self.assertFalse(ws.web_search_available())


# ---------------------------------------------------------------------------
# Argument coercion
# ---------------------------------------------------------------------------


class TestArgumentCoercion(unittest.TestCase):
    def test_missing_or_blank_query_is_an_unbilled_error(self):
        for args in ({}, {"query": ""}, {"query": "   "}, {"query": 42}):
            outcome = ws.execute_web_search_call(args)
            self.assertTrue(outcome.is_error, args)
            self.assertFalse(outcome.billable, args)
            self.assertIn("query", outcome.content)

    def test_num_results_is_clamped_and_coerced(self):
        patcher, client = _with_exa(
            _exa_response(200, {"results": []}),
            _exa_response(200, {"results": []}),
            _exa_response(200, {"results": []}),
        )
        with patcher:
            ws.execute_web_search_call({"query": "q", "num_results": 99})
            ws.execute_web_search_call({"query": "q", "num_results": "3"})
            ws.execute_web_search_call({"query": "q", "num_results": "junk"})
        self.assertEqual(
            [p["numResults"] for p in client.payloads],
            [ws.MAX_NUM_RESULTS, 3, ws.DEFAULT_NUM_RESULTS],
        )

    def test_recency_days_maps_to_a_published_date_floor(self):
        patcher, client = _with_exa(_exa_response(200, {"results": []}))
        with patcher:
            ws.execute_web_search_call({"query": "q", "recency_days": 7})
        self.assertIn("startPublishedDate", client.payloads[0])

    def test_recency_days_omitted_by_default(self):
        patcher, client = _with_exa(_exa_response(200, {"results": []}))
        with patcher:
            ws.execute_web_search_call({"query": "q"})
        self.assertNotIn("startPublishedDate", client.payloads[0])


# ---------------------------------------------------------------------------
# Exa request shaping
# ---------------------------------------------------------------------------


class TestExaRequestShaping(unittest.TestCase):
    def test_request_asks_for_text_only_with_char_cap(self):
        """`highlights`/`summary` are each billed as another content type."""
        patcher, client = _with_exa(_exa_response(200, {"results": []}))
        with patcher:
            ws.run_web_search("anything")
        payload = client.payloads[0]
        self.assertEqual(
            payload["contents"], {"text": {"maxCharacters": ws.MAX_RESULT_CHARS}}
        )
        self.assertEqual(payload["type"], ws.EXA_SEARCH_TYPE)

    def test_reported_cost_is_captured_but_not_authoritative(self):
        patcher, _ = _with_exa(
            _exa_response(
                200,
                {
                    "results": [_exa_result("https://a.com")],
                    "costDollars": {"total": 0.012},
                },
            )
        )
        with patcher:
            outcome = ws.run_web_search("q")
        self.assertEqual(outcome.reported_cost_usd, 0.012)
        self.assertTrue(outcome.billable)


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------


class TestResultFormatting(unittest.TestCase):
    def _search(self, results):
        patcher, _ = _with_exa(_exa_response(200, {"results": results}))
        with patcher:
            return ws.run_web_search("test query")

    def test_results_are_numbered_and_carry_url_and_date(self):
        outcome = self._search(
            [
                _exa_result("https://a.com", title="Alpha"),
                _exa_result("https://b.com", title="Beta"),
            ]
        )
        self.assertIn("[1] Alpha", outcome.content)
        self.assertIn("[2] Beta", outcome.content)
        self.assertIn("URL: https://a.com", outcome.content)
        self.assertIn("Published: 2026-03-04", outcome.content)

    def test_citations_match_only_results_shown_to_the_model(self):
        long_text = "x" * ws.MAX_RESULT_CHARS
        results = [
            _exa_result(f"https://site{i}.com", text=long_text) for i in range(30)
        ]
        outcome = self._search(results)
        self.assertLess(len(outcome.citations), 30)
        for citation in outcome.citations:
            self.assertIn(citation["url"], outcome.content)
        self.assertLessEqual(
            len(outcome.content), ws.MAX_TOTAL_CHARS + ws.MAX_RESULT_CHARS
        )

    def test_citation_shape(self):
        outcome = self._search([_exa_result("https://a.com", title="Alpha")])
        self.assertEqual(
            outcome.citations,
            [
                {
                    "title": "Alpha",
                    "url": "https://a.com",
                    "published_date": "2026-03-04T10:00:00.000Z",
                }
            ],
        )

    def test_result_without_url_is_skipped(self):
        outcome = self._search(
            [{"title": "no url", "text": "t"}, _exa_result("https://a.com")]
        )
        self.assertEqual(len(outcome.citations), 1)

    def test_per_result_text_is_truncated(self):
        outcome = self._search(
            [_exa_result("https://a.com", text="y" * (ws.MAX_RESULT_CHARS * 2))]
        )
        self.assertIn("…", outcome.content)

    def test_zero_results_is_billable_but_says_so(self):
        outcome = self._search([])
        self.assertTrue(outcome.billable)
        self.assertFalse(outcome.is_error)
        self.assertIn("No web results", outcome.content)
        self.assertEqual(outcome.citations, [])


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestSearchFailureModes(unittest.TestCase):
    def test_no_key_configured_is_an_unbilled_error(self):
        with patch.object(ws, "_exa_http_client", None):
            outcome = ws.run_web_search("q")
        self.assertTrue(outcome.is_error)
        self.assertFalse(outcome.billable)

    def test_transport_error_is_an_unbilled_error(self):
        import httpx

        client = Mock()
        client.post.side_effect = httpx.ConnectError("boom")
        with patch.object(ws, "_exa_http_client", client):
            outcome = ws.run_web_search("q")
        self.assertTrue(outcome.is_error)
        self.assertFalse(outcome.billable)
        self.assertIn("could not reach", outcome.content)

    def test_http_error_surfaces_exa_detail(self):
        patcher, _ = _with_exa(_exa_response(401, {"error": "invalid api key"}))
        with patcher:
            outcome = ws.run_web_search("q")
        self.assertTrue(outcome.is_error)
        self.assertFalse(outcome.billable)
        self.assertIn("401", outcome.content)
        self.assertIn("invalid api key", outcome.content)

    def test_malformed_json_is_an_unbilled_error(self):
        response = Mock()
        response.status_code = 200
        response.json.side_effect = ValueError("bad json")
        client = _ExaClient([response])
        with patch.object(ws, "_exa_http_client", client):
            outcome = ws.run_web_search("q")
        self.assertTrue(outcome.is_error)
        self.assertFalse(outcome.billable)


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


class _FakePriceFeed:
    def __init__(self, price):
        self._price = price

    def get_price(self):
        if isinstance(self._price, Exception):
            raise self._price
        return self._price


class TestWebSearchPricing(unittest.TestCase):
    def test_flat_cost_converts_usd_to_opg(self):
        feed = _FakePriceFeed(Decimal("0.10"))
        with patch("tee_gateway.price_feed.get_price_feed", return_value=feed):
            cost = compute_web_search_cost()
        self.assertIsNotNone(cost)
        # $0.015 at $0.10/OPG => 0.15 OPG = 15e16 smallest units.
        expected_opg = int((WEB_SEARCH_PRICE_USD / Decimal("0.10")) * Decimal(10) ** 18)
        self.assertEqual(cost.cost_opg, expected_opg)
        # The USD figure reconciles from the rounded OPG value.
        self.assertEqual(
            cost.cost_usd,
            Decimal(cost.cost_opg) / Decimal(10) ** 18 * Decimal("0.10"),
        )

    def test_price_feed_failure_returns_none(self):
        feed = _FakePriceFeed(ValueError("feed down"))
        with patch("tee_gateway.price_feed.get_price_feed", return_value=feed):
            self.assertIsNone(compute_web_search_cost())

    def test_chat_token_cost_carries_no_search_surcharge(self):
        """Search billing left the chat path entirely with the loop."""
        feed = _FakePriceFeed(Decimal("0.10"))
        usage = {"prompt_tokens": 1000, "completion_tokens": 100}
        with patch("tee_gateway.price_feed.get_price_feed", return_value=feed):
            cost = compute_session_cost("gpt-4.1", usage)
        self.assertIsNotNone(cost)
        # Recompute from the rate card alone: tokens only, nothing else.
        from tee_gateway.model_registry import get_model_config

        cfg = get_model_config("gpt-4.1")
        raw_usd = 1000 * cfg.input_price_usd + 100 * cfg.output_price_usd
        expected_opg = int(
            ((raw_usd / Decimal("0.10")) * Decimal(10) ** 18).to_integral_value(
                rounding="ROUND_CEILING"
            )
        )
        self.assertEqual(cost.cost_opg, expected_opg)


# ---------------------------------------------------------------------------
# /v1/web_search controller
# ---------------------------------------------------------------------------


def _mock_tee_keys():
    tee = Mock()
    tee.sign_data.return_value = "bW9ja3NpZ25hdHVyZQ=="
    tee.get_tee_id.return_value = "abcdef01" * 8
    return tee


class TestWebSearchController(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        app.add_url_rule(
            "/v1/web_search", "web-search", create_web_search, methods=["POST"]
        )
        self.client = app.test_client()

        self.tee = patch(
            "tee_gateway.controllers.web_search_controller.get_tee_keys",
            return_value=_mock_tee_keys(),
        )
        self.tee.start()
        self.addCleanup(self.tee.stop)

        self.feed = patch(
            "tee_gateway.price_feed.get_price_feed",
            return_value=_FakePriceFeed(Decimal("0.10")),
        )
        self.feed.start()
        self.addCleanup(self.feed.stop)

    def _post(self, body):
        return self.client.post("/v1/web_search", json=body)

    def test_successful_search_returns_signed_result_with_cost(self):
        patcher, exa = _with_exa(
            _exa_response(200, {"results": [_exa_result("https://a.com", "Alpha")]})
        )
        with patcher:
            response = self._post({"query": "latest news"})

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["object"], "web_search.result")
        self.assertEqual(body["query"], "latest news")
        self.assertIn("[1] Alpha", body["content"])
        self.assertEqual(body["citations"][0]["url"], "https://a.com")
        # Signed like every other paid endpoint.
        for field in (
            "tee_signature",
            "tee_request_hash",
            "tee_output_hash",
            "tee_timestamp",
            "tee_id",
        ):
            self.assertIn(field, body)
        # Billed at the flat per-search rate.
        expected_opg = int((WEB_SEARCH_PRICE_USD / Decimal("0.10")) * Decimal(10) ** 18)
        self.assertEqual(body["opengradient"]["cost_opg"], str(expected_opg))
        self.assertEqual(exa.payloads[0]["query"], "latest news")

    def test_request_hash_covers_the_canonical_body(self):
        """The client can recompute the hash from exactly what it sent."""
        from tee_gateway.tee_manager import compute_tee_msg_hash

        request_body = {"query": "q", "num_results": 2}
        patcher, _ = _with_exa(
            _exa_response(200, {"results": [_exa_result("https://a.com")]})
        )
        with patcher:
            body = self._post(request_body).get_json()

        request_bytes = json.dumps(request_body, sort_keys=True).encode("utf-8")
        _, input_hash_hex, output_hash_hex = compute_tee_msg_hash(
            request_bytes, body["content"], body["tee_timestamp"]
        )
        self.assertEqual(body["tee_request_hash"], input_hash_hex)
        self.assertEqual(body["tee_output_hash"], output_hash_hex)

    def test_zero_results_still_bills_and_tells_the_model(self):
        patcher, _ = _with_exa(_exa_response(200, {"results": []}))
        with patcher:
            response = self._post({"query": "obscure thing"})
        body = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertIn("No web results", body["content"])
        self.assertEqual(body["citations"], [])
        self.assertIn("opengradient", body)

    def test_missing_query_is_400(self):
        with patch.object(ws, "_exa_http_client", Mock()):
            for payload in ({}, {"query": ""}, {"query": 7}):
                response = self._post(payload)
                self.assertEqual(response.status_code, 400, payload)

    def test_no_exa_key_is_503(self):
        with patch.object(ws, "_exa_http_client", None):
            response = self._post({"query": "q"})
        self.assertEqual(response.status_code, 503)

    def test_exa_failure_is_502_without_cost_block(self):
        patcher, _ = _with_exa(_exa_response(500, {"error": "upstream broke"}))
        with patcher:
            response = self._post({"query": "q"})
        self.assertEqual(response.status_code, 502)
        body = response.get_json()
        self.assertNotIn("opengradient", body)
        self.assertIn("upstream broke", body["error"])

    def test_price_feed_outage_returns_result_without_cost_block(self):
        """Fail-open like chat: the client gets its answer, unsettled."""
        self.feed.stop()
        feed = patch(
            "tee_gateway.price_feed.get_price_feed",
            return_value=_FakePriceFeed(ValueError("down")),
        )
        feed.start()
        self.addCleanup(feed.stop)
        # Re-arm the harness patcher reference so cleanup doesn't double-stop.
        self.feed = feed

        patcher, _ = _with_exa(
            _exa_response(200, {"results": [_exa_result("https://a.com")]})
        )
        with patcher:
            response = self._post({"query": "q"})
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("opengradient", response.get_json())


# ---------------------------------------------------------------------------
# OHTTP inner-endpoint dispatch
# ---------------------------------------------------------------------------


class _FakeDecap:
    plaintext = b""  # set per-test
    response_key = b"k" * 32
    response_key_chunked = b"c" * 32
    enc = b"e" * 32


class TestOhttpEndpointDispatch(unittest.TestCase):
    """The sealed payload's `endpoint` field picks the inner path."""

    def setUp(self):
        app = Flask(__name__)
        app.add_url_rule(
            "/v1/ohttp",
            "anonymous-chat",
            ohttp_controller.create_anonymous_chat_completion,
            methods=["POST"],
        )
        self.client = app.test_client()

        tee = Mock()
        tee.hpke_private_key = object()
        self.patchers = [
            patch.object(ohttp_controller, "get_tee_keys", return_value=tee),
            patch.object(ohttp_controller.ohttp, "decapsulate_request"),
            patch.object(ohttp_controller, "_wsgi_subrequest"),
            patch.object(
                ohttp_controller.ohttp, "encapsulate_response", return_value=b"sealed"
            ),
        ]
        _, self.decap, self.subrequest, _ = [p.start() for p in self.patchers]
        self.addCleanup(lambda: [p.stop() for p in self.patchers])

        self.subrequest.return_value = (
            200,
            [("Content-Type", "application/json")],
            iter([b'{"ok": true}']),
        )

    def _post_inner(self, inner: dict):
        decap = _FakeDecap()
        decap.plaintext = json.dumps(inner).encode("utf-8")
        self.decap.return_value = decap
        return self.client.post(
            "/v1/ohttp",
            data=b"ciphertext",
            content_type="message/ohttp-req",
        )

    def test_web_search_endpoint_routes_to_the_search_path(self):
        response = self._post_inner({"endpoint": "web_search", "query": "q"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.subrequest.call_args.kwargs["path"], "/v1/web_search")
        # The discriminator is routing metadata, not part of the inner body.
        forwarded = json.loads(self.subrequest.call_args.kwargs["body_bytes"])
        self.assertEqual(forwarded, {"query": "q"})

    def test_absent_endpoint_still_means_chat(self):
        """The original OHTTP contract: existing clients keep working."""
        response = self._post_inner({"model": "gpt-4.1", "messages": []})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            self.subrequest.call_args.kwargs["path"], "/v1/chat/completions"
        )

    def test_unknown_endpoint_is_rejected_without_dispatch(self):
        response = self._post_inner({"endpoint": "nope", "query": "q"})
        # Sealed error: outer 200 carrying an encapsulated {status: 400, ...}.
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "message/ohttp-res")
        self.subrequest.assert_not_called()


if __name__ == "__main__":
    unittest.main()
