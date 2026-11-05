"""Reproductions of the ways the gateway serves paid inference without being paid.

The gateway runs x402's ``upto`` scheme in optimistic session mode: a payer
signs one Permit2 permit up to a spend cap, the gateway serves immediately,
each response's cost is added to an in-memory session in
``StreamingSessionResponse.close()``, and a background reaper settles the
session on-chain later. A request is only paid for when BOTH of those happen:
its cost lands in the session, and that session later settles. Every test
below drives one gap between "served" and "paid".

The harness is the production code path wherever possible: the SDK's real
``PaymentMiddleware`` and ``SessionStore``, the gateway's own patched
``_stream_session_response`` and ``_session_cost_calculator`` (both installed
by importing ``tee_gateway.__main__``), and the real controllers and pricing.
Only the facilitator is faked: every payment verifies, and settlement records
the amount — failing, as the real one does, once the permit's deadline is
within 6 seconds.

The ``xfail(strict=True)`` tests assert the behaviour the gateway should have
and fail today; each reason names the money that leaks. Once a leak is fixed
its test starts passing, strict xfail turns that into a failure, and the
marker should be removed. The ``test_control_*`` tests pass today and prove
the harness does bill the happy path, so an xfail is the leak and not the
harness. Run ``pytest tests/test_x402_billing_leaks.py --runxfail`` to see
each leak as a plain assertion failure.
"""

from __future__ import annotations

import atexit
import json
import signal
import struct
import threading
import time
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Callable, Iterator
from unittest.mock import MagicMock, patch

import httpx
import pytest
from flask import Flask, Response
from werkzeug.test import EnvironBuilder
from x402.http.middleware.flask import PaymentMiddleware
from x402.schemas import PaymentPayload, PaymentRequirements
from x402.session import SessionStore

from tee_gateway import __main__ as gateway_main  # installs the gateway's x402 patches
from tee_gateway import gunicorn_conf, image_generation, llm_backend, ohttp
from tee_gateway.controllers import chat_controller, ohttp_controller
from tee_gateway.definitions import (
    BASE_MAINNET_NETWORK,
    BASE_MAINNET_OPG_ADDRESS,
    CHAT_COMPLETIONS_OPG_SESSION_MAX_SPEND,
    EVM_PAYMENT_ADDRESS,
    OHTTP_OPG_SESSION_MAX_SPEND,
)
from tee_gateway.errors import error_response
from tee_gateway.models import ChatCompletionRequestUserMessage
from tee_gateway.models.create_chat_completion_request import (
    CreateChatCompletionRequest,
)
from tee_gateway.price_feed import get_price_feed, set_price_feed
from tee_gateway.pricing import SessionCost, compute_session_cost

# OPG/USD from CoinGecko on 2026-10-14. The caps below are fixed in OPG, so
# what they cover in USD moves with this price.
OPG_PRICE_USD = Decimal("0.136707")
CHAT_CAP = int(CHAT_COMPLETIONS_OPG_SESSION_MAX_SPEND)  # 0.1 OPG
OHTTP_CAP = int(OHTTP_OPG_SESSION_MAX_SPEND)  # 5 OPG

CHAT_PATH = "/v1/chat/completions"
OHTTP_PATH = "/v1/ohttp"


# ---------------------------------------------------------------------------
# Harness


@pytest.fixture(autouse=True)
def opg_price():
    """Price every test at the live OPG/USD rate instead of CoinGecko."""
    try:
        previous = get_price_feed()
    except RuntimeError:
        previous = None
    stub = MagicMock()
    stub.get_price.return_value = OPG_PRICE_USD
    set_price_feed(stub)
    yield
    if previous is not None:
        set_price_feed(previous)


def _requirements(cap: int) -> PaymentRequirements:
    return PaymentRequirements.model_validate(
        {
            "scheme": "upto",
            "network": BASE_MAINNET_NETWORK,
            "asset": BASE_MAINNET_OPG_ADDRESS,
            "amount": str(cap),
            "payTo": EVM_PAYMENT_ADDRESS,
            "maxTimeoutSeconds": 3600,
        }
    )


def _payload(cap: int, deadline: int) -> PaymentPayload:
    return PaymentPayload.model_validate(
        {
            "x402Version": 2,
            "payload": {
                "permit2Authorization": {"deadline": str(deadline)},
                "signature": "0x",
            },
            "accepted": _requirements(cap).model_dump(by_alias=True),
        }
    )


class FakeFacilitator:
    """Stands in for ``x402HTTPResourceServerSync``.

    Every payment verifies (as a valid permit backed by enough balance
    would). Settlement records the amount, and fails the way the real
    facilitator does when the permit's deadline is less than 6 seconds away.
    """

    def __init__(self, cap: int, deadline: int):
        self.cap = cap
        self.deadline = deadline
        self.settled: list[int] = []
        self.failed: list[int] = []

    def requires_payment(self, context: Any) -> bool:
        return context.method == "POST"

    def process_http_request(self, context: Any, paywall_config: Any = None) -> Any:
        return SimpleNamespace(
            type="payment-verified",
            payment_payload=_payload(self.cap, self.deadline),
            payment_requirements=_requirements(self.cap),
        )

    def process_settlement(
        self, payload: Any, requirements: Any, settlement_overrides: Any = None
    ) -> Any:
        amount = int(settlement_overrides.amount)
        if self.deadline < time.time() + 6:
            self.failed.append(amount)
            return SimpleNamespace(
                success=False, error_reason="permit2_deadline_expired", transaction=""
            )
        self.settled.append(amount)
        return SimpleNamespace(success=True, error_reason=None, transaction="0xabc")

    def process_settlement_data(self, *args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(success=True, error_reason=None)


class Gateway:
    """A Flask app behind the real x402 session middleware."""

    def __init__(
        self,
        app: Flask,
        *,
        cap: int = OHTTP_CAP,
        deadline_in: int = 3600,
        idle_timeout: int = 100,
        reaper_interval: int = 5,
    ):
        self.app = app
        self.store = SessionStore()
        self.facilitator = FakeFacilitator(cap, int(time.time()) + deadline_in)
        self.middleware = PaymentMiddleware(
            app,
            routes={},
            server=MagicMock(),
            sync_facilitator_on_start=False,
            session_store=self.store,
            session_cost_calculator=gateway_main._session_cost_calculator,
            session_idle_timeout=idle_timeout,
            session_reaper_interval=reaper_interval,
        )
        self.middleware._http_server = self.facilitator

    def post(
        self,
        path: str = CHAT_PATH,
        body: bytes = b"{}",
        *,
        payment: str = "permit-A",
        content_type: str = "application/json",
        read_chunks: int | None = None,
    ) -> tuple[int, bytes]:
        """Run one request through WSGI like gunicorn does.

        ``read_chunks`` stops reading after that many body chunks and closes
        the response, which is what the server does when the client hangs up.
        """
        environ = EnvironBuilder(
            method="POST",
            path=path,
            data=body,
            content_type=content_type,
            headers={"Payment-Signature": payment},
        ).get_environ()
        captured: dict[str, Any] = {}

        def start_response(status: str, headers: list, exc_info: Any = None):
            captured["status"] = status
            return lambda _data: None

        body_iter = self.app.wsgi_app(environ, start_response)
        chunks: list[bytes] = []
        try:
            for chunk in body_iter:
                chunks.append(chunk)
                if read_chunks is not None and len(chunks) >= read_chunks:
                    break
        finally:
            close = getattr(body_iter, "close", None)
            if callable(close):
                close()
        return int(captured["status"].split(" ", 1)[0]), b"".join(chunks)

    def settle_idle_sessions(self) -> None:
        """Run one reaper cycle as if every session had gone idle."""
        for session in list(self.store._sessions.values()):
            session.last_activity -= 10_000
        self.middleware._settle_ready_sessions()

    def billed(self) -> int:
        """What the payer will pay: settled, plus still accumulating."""
        pending = sum(
            s.accumulated_cost for s in self.store._sessions.values() if not s.settled
        )
        return sum(self.facilitator.settled) + pending

    def stop(self) -> None:
        self.middleware._reaper_stop.set()


@pytest.fixture
def gateways():
    """Stop every reaper thread the test started."""
    started: list[Gateway] = []
    yield started
    for gateway in started:
        gateway.stop()


def _app(view: Callable[[], Any], path: str = CHAT_PATH) -> Flask:
    app = Flask(__name__)
    app.add_url_rule(path, "view", view, methods=["POST"])
    return app


def _cost(cost_opg: int) -> dict[str, str]:
    return SessionCost(
        cost_opg=cost_opg,
        cost_usd=Decimal(cost_opg) / Decimal(10**18) * OPG_PRICE_USD,
        opg_price_usd=OPG_PRICE_USD,
    ).model_dump(mode="json")


def _costing(cost_opg: int, *, before: Callable[[], None] | None = None):
    """A view that does its work, then answers with an ``opengradient`` cost."""

    def view():
        if before is not None:
            before()
        return {"id": "chatcmpl-test", "opengradient": _cost(cost_opg)}

    return view


def _user(text: str) -> list:
    return [ChatCompletionRequestUserMessage(role="user", content=text)]


# ---------------------------------------------------------------------------
# Controls: the harness bills and settles the happy path


def test_control_request_within_cap_is_billed_and_settled(gateways):
    gw = Gateway(_app(_costing(1_000)))
    gateways.append(gw)

    status, _ = gw.post()
    gw.settle_idle_sessions()

    assert status == 200
    assert gw.facilitator.settled == [1_000]


# ---------------------------------------------------------------------------
# Leak 1: a cost that doesn't fit under the session cap is dropped whole.
# ``SessionStore.add_cost`` is all-or-nothing, and the response has already
# streamed, so the request is free.


def _direct_route_cost(case: str) -> int:
    if case == "grok-2-image":  # $0.02, the cheapest per-image model
        cost = compute_session_cost(
            "grok-2-image", {"prompt_tokens": 0, "completion_tokens": 0}, image_count=1
        )
    elif case == "gpt-image-2":  # $0.05
        cost = compute_session_cost(
            "gpt-image-2", {"prompt_tokens": 0, "completion_tokens": 0}, image_count=1
        )
    else:  # an ordinary chat turn: 3k tokens in, 800 out (~$0.021)
        cost = compute_session_cost(
            "claude-sonnet-4-5", {"prompt_tokens": 3_000, "completion_tokens": 800}
        )
    assert cost is not None
    return cost.cost_opg


@pytest.mark.xfail(
    strict=True,
    reason="Direct /v1/chat/completions caps a session at 0.1 OPG (~$0.0137); "
    "anything costing more is served and never billed",
)
@pytest.mark.parametrize("case", ["grok-2-image", "gpt-image-2", "sonnet-3k-turn"])
def test_direct_route_request_costing_more_than_cap_is_billed(gateways, case):
    cost = _direct_route_cost(case)
    gw = Gateway(_app(_costing(cost)), cap=CHAT_CAP)
    gateways.append(gw)

    status, _ = gw.post()

    assert status == 200
    assert gw.billed() == cost


@pytest.mark.xfail(
    strict=True,
    reason="A request that pushes a partly used session past its cap is "
    "served and dropped whole, not billed for what's left",
)
def test_request_crossing_cap_on_partly_used_session_is_billed(gateways):
    costs = iter([OHTTP_CAP * 9 // 10, OHTTP_CAP // 5])  # 90% of cap, then 20%

    def view():
        return {"id": "chatcmpl-test", "opengradient": _cost(next(costs))}

    gw = Gateway(_app(view))
    gateways.append(gw)

    first, _ = gw.post()
    second, _ = gw.post()  # session isn't exhausted yet, so this is served

    assert (first, second) == (200, 200)
    assert gw.billed() == OHTTP_CAP * 9 // 10 + OHTTP_CAP // 5


@pytest.mark.xfail(
    strict=True,
    reason="Concurrent requests on one session all pass the not-exhausted "
    "check; the ones that finish after the cap fills are free (chat-api "
    "shares one session across concurrent streams)",
)
def test_concurrent_requests_on_one_session_are_all_billed(gateways):
    per_request = OHTTP_CAP * 2 // 5  # 40% of the cap each
    all_in_flight = threading.Barrier(3, timeout=10)
    concurrent = threading.Event()

    def hold_until_all_in_flight() -> None:
        if concurrent.is_set():
            all_in_flight.wait()

    gw = Gateway(_app(_costing(per_request, before=hold_until_all_in_flight)))
    gateways.append(gw)

    gw.post()  # opens the session the concurrent requests share
    concurrent.set()
    statuses: list[int] = []
    threads = [
        threading.Thread(target=lambda: statuses.append(gw.post()[0])) for _ in range(3)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert statuses == [200, 200, 200]
    assert gw.billed() == per_request * 4


# ---------------------------------------------------------------------------
# Leak 2: the reaper settles and closes a session while a request on it is
# still running. ``last_activity`` only moves when a cost is added, so any
# request outliving the idle timeout (100s in production) can lose its cost.


@pytest.mark.xfail(
    strict=True,
    reason="The reaper settles a session idle since its last completed "
    "request while another request on it is still running; that request's "
    "cost has nowhere to go",
)
def test_request_running_when_reaper_settles_session_is_billed(gateways):
    costs = iter([1_000, 2_000])
    slow = threading.Event()

    def view():
        cost = next(costs)
        if slow.is_set():
            time.sleep(3)  # outlives the 1s idle timeout, like a 100s+ stream
        return {"id": "chatcmpl-test", "opengradient": _cost(cost)}

    # Production is idle_timeout=100, reaper_interval=5; scaled down here.
    gw = Gateway(_app(view), idle_timeout=1, reaper_interval=1)
    gateways.append(gw)

    gw.post()
    slow.set()
    status, _ = gw.post()

    assert status == 200
    assert gw.billed() == 3_000


# ---------------------------------------------------------------------------
# Leak 3: unsettled sessions live only in process memory and nothing flushes
# them when the process exits.


@pytest.mark.xfail(
    strict=True,
    reason="Nothing calls PaymentMiddleware.shutdown() on exit, so every "
    "deploy or crash drops the sessions not yet settled",
)
def test_unsettled_sessions_are_flushed_on_process_exit(monkeypatch):
    registered: list[Any] = []
    monkeypatch.setattr(atexit, "register", lambda fn, *a, **k: registered.append(fn))
    monkeypatch.setattr(signal, "signal", lambda sig, fn: registered.append(fn))

    created: list[PaymentMiddleware] = []
    real_payment_middleware = gateway_main.payment_middleware

    def recording_payment_middleware(*args: Any, **kwargs: Any) -> PaymentMiddleware:
        middleware = real_payment_middleware(*args, **kwargs)
        created.append(middleware)
        return middleware

    monkeypatch.setattr(
        gateway_main, "payment_middleware", recording_payment_middleware
    )
    application = gateway_main.application
    flask_app = getattr(application, "app", application)
    saved_wsgi = application.wsgi_app
    saved_inner = flask_app.config.get("OHTTP_INNER_WSGI_APP")
    try:
        gateway_main._init_payment_middleware("http://facilitator.invalid")
    finally:
        application.wsgi_app = saved_wsgi
        flask_app.config["OHTTP_INNER_WSGI_APP"] = saved_inner

    (middleware,) = created
    flushed_on_exit = any(fn == middleware.shutdown for fn in registered)
    gunicorn_hook = hasattr(gunicorn_conf, "worker_exit")
    assert flushed_on_exit or gunicorn_hook


# ---------------------------------------------------------------------------
# Leak 4: a new session is opened on a permit about to expire. The deadline
# buffer (120s) is only checked for sessions that already exist.


@pytest.mark.xfail(
    strict=True,
    reason="A new session is opened on a permit expiring in 10s; the "
    "120s deadline buffer only applies to existing sessions",
)
def test_new_session_on_permit_about_to_expire_is_refused(gateways):
    gw = Gateway(_app(_costing(1_000)), deadline_in=10)
    gateways.append(gw)

    status, _ = gw.post()

    assert status == 402


@pytest.mark.xfail(
    strict=True,
    reason="A request that outlives its short-deadline permit is served, "
    "then its settlement can never go through",
)
def test_request_outliving_short_permit_is_paid(gateways):
    gw = Gateway(
        _app(_costing(1_000, before=lambda: time.sleep(3))),
        deadline_in=8,  # facilitator verify only needs deadline >= now + 6
    )
    gateways.append(gw)

    status, _ = gw.post()
    gw.middleware._settle_ready_sessions()

    assert status == 200
    assert gw.facilitator.settled == [1_000]


@pytest.mark.xfail(
    strict=True,
    reason="A session that can never settle is retried every reaper cycle "
    "forever and never surfaced as a loss",
)
def test_unsettleable_session_is_not_retried_forever(gateways):
    gw = Gateway(
        _app(_costing(1_000, before=lambda: time.sleep(3))),
        deadline_in=8,
    )
    gateways.append(gw)

    gw.post()
    for _ in range(3):
        gw.middleware._settle_ready_sessions()

    assert len(gw.facilitator.failed) == 1


# ---------------------------------------------------------------------------
# Leak 5: the cost is only produced when a response finishes cleanly. Any
# early exit is served (and paid upstream) but never billed.


def _chunk(content: str = "", usage: dict | None = None) -> MagicMock:
    chunk = MagicMock()
    chunk.content = content
    chunk.tool_call_chunks = []
    chunk.usage_metadata = usage
    chunk.response_metadata = {}
    return chunk


_USAGE = {"input_tokens": 3_000, "output_tokens": 800, "total_tokens": 3_800}


def _streaming_chat_app(stream: Callable[[], Iterator[Any]]) -> tuple[Flask, Any]:
    """The real streaming chat controller over a fake provider stream."""
    fake_model = MagicMock()
    fake_model.stream.side_effect = lambda *a, **k: stream()
    tee_keys = MagicMock()
    tee_keys.sign_data.return_value = "0xsig"
    tee_keys.get_tee_id.return_value = "deadbeef"
    patches = patch.multiple(
        chat_controller,
        get_chat_model_cached=MagicMock(return_value=fake_model),
        get_provider_from_model=MagicMock(return_value="openai"),
        get_tee_keys=MagicMock(return_value=tee_keys),
        compute_tee_msg_hash=MagicMock(return_value=(b"h", "ih", "oh")),
    )

    def view():
        request = CreateChatCompletionRequest(
            model="gpt-4.1-mini", messages=_user("hi"), stream=True
        )
        return chat_controller._create_streaming_response(request)

    return _app(view), patches


def _full_stream() -> Iterator[Any]:
    yield _chunk("hello")
    yield _chunk(" world")
    yield _chunk(usage=_USAGE)


def test_control_fully_read_stream_is_billed(gateways):
    app, patches = _streaming_chat_app(_full_stream)
    gw = Gateway(app)
    gateways.append(gw)

    with patches:
        status, _ = gw.post()

    assert status == 200
    assert gw.billed() > 0


@pytest.mark.xfail(
    strict=True,
    reason="A client that hangs up before the final SSE frame has read the "
    "answer, but the cost only exists in that frame",
)
def test_stream_client_disconnect_before_final_frame_is_billed(gateways):
    app, patches = _streaming_chat_app(_full_stream)
    gw = Gateway(app)
    gateways.append(gw)

    with patches:
        status, _ = gw.post(read_chunks=2)  # both content deltas, then hang up

    assert status == 200
    assert gw.billed() > 0


def _final_frame(body: bytes) -> dict:
    events = [
        line[len(b"data:") :].strip()
        for line in body.splitlines()
        if line.startswith(b"data:") and b"[DONE]" not in line
    ]
    return json.loads(events[-1])


def _token_cost(model: str, prompt_tokens: int, completion_tokens: int) -> int:
    cost = compute_session_cost(
        model, {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    )
    assert cost is not None
    return cost.cost_opg


def test_stream_provider_error_midway_bills_estimated_usage(gateways):
    # OpenAI only reports usage at the end, so a stream cut off mid-way has
    # none: the prompt ("hi") and what was streamed ("hello world") are
    # estimated at 4 characters per token.
    def failing_stream() -> Iterator[Any]:
        yield _chunk("hello")
        yield _chunk(" world")
        raise RuntimeError("provider connection reset")

    app, patches = _streaming_chat_app(failing_stream)
    gw = Gateway(app)
    gateways.append(gw)

    with patches:
        status, body = gw.post()

    assert status == 200
    assert "error" in _final_frame(body)
    assert gw.billed() == _token_cost("gpt-4.1-mini", 1, 3)


def test_stream_provider_error_midway_bills_reported_input_tokens(gateways):
    # Anthropic reports input tokens on the stream's first chunk; those are
    # billed as reported instead of estimated.
    def failing_stream() -> Iterator[Any]:
        yield _chunk(usage={"input_tokens": 3_000, "output_tokens": 0})
        yield _chunk("hello world")
        raise RuntimeError("overloaded_error")

    app, patches = _streaming_chat_app(failing_stream)
    gw = Gateway(app)
    gateways.append(gw)

    with patches:
        gw.post()

    assert gw.billed() == _token_cost("gpt-4.1-mini", 3_000, 3)


def test_stream_refused_before_any_output_is_not_billed(gateways):
    # A provider that refuses the request (rate limit, bad request) did no
    # work and bills nothing, so neither does the gateway.
    def refused_stream() -> Iterator[Any]:
        raise RuntimeError("rate limited")
        yield  # pragma: no cover - makes this a generator

    app, patches = _streaming_chat_app(refused_stream)
    gw = Gateway(app)
    gateways.append(gw)

    with patches:
        status, body = gw.post()

    assert status == 200
    assert "opengradient" not in _final_frame(body)
    assert gw.billed() == 0


def _image_cost(images: int) -> int:
    cost = compute_session_cost(
        "grok-2-image",
        {"prompt_tokens": 0, "completion_tokens": 0},
        image_count=images,
    )
    assert cost is not None
    return cost.cost_opg


def _image_app(*, stream: bool, urls: list[str]) -> tuple[Flask, Any, MagicMock]:
    """Grok image generation through the real responders.

    The provider returns ``urls``; fetching any URL containing ``broken``
    fails as an unreachable CDN would.
    """
    provider = MagicMock()
    provider.post.return_value = httpx.Response(
        200,
        json={"data": [{"url": url} for url in urls]},
        request=httpx.Request("POST", "https://api.x.ai/v1/images/generations"),
    )

    def fetch(url: str) -> str:
        if "broken" in url:
            raise httpx.ConnectError("CDN unreachable")
        return "data:image/png;base64,AAAA"

    tee_keys = MagicMock()
    tee_keys.sign_data.return_value = "0xsig"
    tee_keys.get_tee_id.return_value = "deadbeef"
    patches = patch.multiple(
        image_generation,
        _fetch_url_as_data_uri=MagicMock(side_effect=fetch),
        get_tee_keys=MagicMock(return_value=tee_keys),
    )

    def view():
        request = CreateChatCompletionRequest(
            model="grok-2-image",
            messages=_user("a red kite"),
            stream=stream,
            n=len(urls),
        )
        if stream:
            return image_generation.create_image_generation_streaming_response(
                request, b"{}"
            )
        try:
            return image_generation.create_image_generation_response(request, b"{}")
        except Exception as exc:
            return error_response(exc)

    return _app(view), patches, provider


def test_image_partly_fetched_bills_every_generated_image(gateways):
    app, patches, provider = _image_app(
        stream=False,
        urls=["https://imgen.x.ai/ok.png", "https://imgen.x.ai/broken.png"],
    )
    gw = Gateway(app)
    gateways.append(gw)

    with patches, patch.object(llm_backend, "xai_http_client", provider, create=True):
        status, body = gw.post()

    assert status == 200
    assert len(json.loads(body)["choices"][0]["message"]["images"]) == 1
    assert gw.billed() == _image_cost(2)


def test_image_none_fetched_while_streaming_bills_generated_images(gateways):
    app, patches, provider = _image_app(
        stream=True, urls=["https://imgen.x.ai/broken.png"]
    )
    gw = Gateway(app)
    gateways.append(gw)

    with patches, patch.object(llm_backend, "xai_http_client", provider, create=True):
        status, body = gw.post()

    assert status == 200
    assert "error" in _final_frame(body)
    assert gw.billed() == _image_cost(1)


@pytest.mark.xfail(
    strict=True,
    reason="A non-streaming request whose images all fail to fetch answers "
    "with an error status, and chat-api only debits successful responses",
)
def test_image_none_fetched_non_streaming_is_billed(gateways):
    app, patches, provider = _image_app(
        stream=False, urls=["https://imgen.x.ai/broken.png"]
    )
    gw = Gateway(app)
    gateways.append(gw)

    with patches, patch.object(llm_backend, "xai_http_client", provider, create=True):
        gw.post()

    assert gw.billed() == _image_cost(1)


# The OHTTP path wraps the same stream in HPKE chunks; its cost is recorded
# from whatever plaintext had been relayed when the outer response closed.


def _encapsulate(plaintext: bytes) -> tuple[Any, bytes]:
    sk, pk_raw = ohttp.generate_keypair()
    header = bytes([ohttp.KEY_CONFIG_ID]) + struct.pack(
        ">HHH",
        ohttp.KEM_ID_X25519,
        ohttp.KDF_ID_HKDF_SHA256,
        ohttp.AEAD_ID_CHACHA20_POLY1305,
    )
    info = b"message/bhttp request" + b"\x00" + header
    pkr = ohttp._SUITE.kem.deserialize_public_key(pk_raw)
    enc, sender = ohttp._SUITE.create_sender_context(pkr, info=info)
    return sk, header + enc + sender.seal(plaintext, aad=b"")


def _ohttp_gateway(monkeypatch, cost_opg: int) -> tuple[Gateway, bytes]:
    sk, wire = _encapsulate(
        json.dumps(
            {"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": "hi"}]}
        ).encode()
    )
    events = [
        b'data: {"choices":[{"delta":{"content":"hello"},"index":0}]}\n\n',
        b'data: {"choices":[{"delta":{"content":" world"},"index":0}]}\n\n',
        b"data: "
        + json.dumps(
            {
                "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
                "opengradient": _cost(cost_opg),
            }
        ).encode()
        + b"\n\n",
        b"data: [DONE]\n\n",
    ]

    def inner_chat(environ, start_response):
        start_response("200 OK", [("Content-Type", "text/event-stream")])
        return iter(events)

    app = _app(ohttp_controller.create_anonymous_chat_completion, OHTTP_PATH)
    app.config["OHTTP_INNER_WSGI_APP"] = inner_chat
    monkeypatch.setattr(
        ohttp_controller,
        "get_tee_keys",
        lambda: SimpleNamespace(hpke_private_key=sk),
    )
    return Gateway(app), wire


def test_control_ohttp_stream_fully_relayed_is_billed(monkeypatch, gateways):
    gw, wire = _ohttp_gateway(monkeypatch, 5_000)
    gateways.append(gw)

    status, _ = gw.post(OHTTP_PATH, wire, content_type="message/ohttp-req")

    assert status == 200
    assert gw.billed() == 5_000


@pytest.mark.xfail(
    strict=True,
    reason="An OHTTP relay that stops reading before the billing frame "
    "leaves the request unbilled (chat-api drains; other relays need not)",
)
def test_ohttp_stream_relay_disconnect_is_billed(monkeypatch, gateways):
    gw, wire = _ohttp_gateway(monkeypatch, 5_000)
    gateways.append(gw)

    # The HPKE header plus the first sealed chunk, then hang up.
    status, _ = gw.post(
        OHTTP_PATH, wire, content_type="message/ohttp-req", read_chunks=2
    )

    assert status == 200
    assert gw.billed() == 5_000


# ---------------------------------------------------------------------------
# Leak 6: pricing below cost. Prices are provider list prices with no margin,
# so any provider surcharge the registry doesn't model is a straight loss.


@pytest.mark.xfail(
    strict=True,
    reason="Gemini 2.5 Pro prompts over 200k tokens cost Google's long-context "
    "rate ($2.50/M in, $15/M out); the gateway charges the base rate",
)
def test_gemini_long_context_prompt_charged_at_long_context_rate():
    cost = compute_session_cost(
        "gemini-2.5-pro", {"prompt_tokens": 300_000, "completion_tokens": 1_000}
    )
    provider_usd = Decimal(300_000) * Decimal("0.0000025") + Decimal(1_000) * Decimal(
        "0.000015"
    )

    assert cost is not None
    assert cost.cost_usd >= provider_usd


# ---------------------------------------------------------------------------
# Leak 7: sessions that never accumulate a cost are never removed.


@pytest.mark.xfail(
    strict=True,
    reason="A session that only ever served errors has nothing to settle, so "
    "the reaper skips it and it stays in memory forever",
)
def test_session_with_nothing_billed_is_eventually_removed(gateways):
    gw = Gateway(_app(lambda: Response("upstream failed", status=502)))
    gateways.append(gw)

    gw.post()
    gw.settle_idle_sessions()

    assert gw.store.active_count == 0
