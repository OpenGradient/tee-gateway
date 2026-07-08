"""Tests for deadline-aware x402 upto-session settlement."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from tee_gateway import x402_settlement
from tee_gateway.x402_settlement import (
    DeadlineAwareSessionStore,
    _patched_settle_ready_sessions,
    _patched_settle_session,
)


def _permit_payload(deadline: int) -> dict[str, Any]:
    """Build a minimal valid upto PaymentPayload dict with a Permit2 deadline."""
    return {
        "x402Version": 2,
        "payload": {
            "permit2Authorization": {
                "from": "0x1111111111111111111111111111111111111111",
                "permitted": {
                    "token": "0xFbC2051AE2265686a469421b2C5A2D5462FbF5eB",
                    "amount": "1000",
                },
                "spender": "0x2222222222222222222222222222222222222222",
                "nonce": "1",
                "deadline": str(deadline),
                "witness": {
                    "to": "0x3333333333333333333333333333333333333333",
                    "facilitator": "0x4444444444444444444444444444444444444444",
                    "validAfter": "0",
                },
            },
            "signature": "0xabc",
        },
        "accepted": _requirements(),
    }


def _requirements() -> dict[str, Any]:
    """Minimal valid PaymentRequirements dict."""
    return {
        "scheme": "upto",
        "network": "eip155:8453",
        "asset": "0xFbC2051AE2265686a469421b2C5A2D5462FbF5eB",
        "amount": "1000",
        "payTo": "0x3333333333333333333333333333333333333333",
        "maxTimeoutSeconds": 3600,
        "extra": {},
    }


def _make_store() -> DeadlineAwareSessionStore:
    return DeadlineAwareSessionStore(
        serve_margin_seconds=420,
        settle_margin_seconds=120,
    )


# ---------------------------------------------------------------------------
# Store behavior
# ---------------------------------------------------------------------------


def test_deadline_recorded_on_create():
    store = _make_store()
    deadline = int(time.time()) + 3600
    sid = store.create_session(_permit_payload(deadline), _requirements(), 1000)
    assert store.session_deadline(sid) == deadline


def test_get_session_hidden_near_deadline():
    store = _make_store()
    # Deadline only 60s away — inside the 420s serve margin.
    deadline = int(time.time()) + 60
    sid = store.create_session(_permit_payload(deadline), _requirements(), 1000)
    # Base store still has it...
    from x402.session import SessionStore

    assert SessionStore.get_session(store, sid) is not None
    # ...but the deadline-aware store hides it from the serving path.
    assert store.get_session(sid) is None


def test_get_session_served_with_healthy_deadline():
    store = _make_store()
    deadline = int(time.time()) + 3600
    sid = store.create_session(_permit_payload(deadline), _requirements(), 1000)
    assert store.get_session(sid) is not None


def test_add_cost_saturates_at_cap():
    store = _make_store()
    deadline = int(time.time()) + 3600
    sid = store.create_session(_permit_payload(deadline), _requirements(), 1000)

    assert store.add_cost(sid, 600) is True
    # Would overflow (600 + 600 > 1000): base store refuses, we saturate.
    assert store.add_cost(sid, 600) is True
    session = store.get_session(sid)
    assert session is not None
    assert session.accumulated_cost == 1000
    assert session.is_exhausted is True
    # Fully consumed cap refuses further cost.
    assert store.add_cost(sid, 1) is False


def test_deadline_due_sessions():
    store = _make_store()
    now = int(time.time())
    # Due: 100s to deadline (< 120s settle margin) and has accrued cost.
    due_sid = store.create_session(_permit_payload(now + 100), _requirements(), 1000)
    store.add_cost(due_sid, 500)
    # Not due: far from deadline.
    far_sid = store.create_session(_permit_payload(now + 3600), _requirements(), 1000)
    store.add_cost(far_sid, 500)
    # Due deadline but zero cost -> nothing to settle, excluded.
    empty_sid = store.create_session(_permit_payload(now + 100), _requirements(), 1000)

    due_ids = {s.session_id for s in store.get_deadline_due_sessions()}
    assert due_sid in due_ids
    assert far_sid not in due_ids
    assert empty_sid not in due_ids


def test_close_session_clears_bookkeeping():
    store = _make_store()
    deadline = int(time.time()) + 3600
    sid = store.create_session(_permit_payload(deadline), _requirements(), 1000)
    store.set_settlement_job(sid, "payment-abc")
    store.close_session(sid)
    assert store.session_deadline(sid) is None
    assert store.get_settlement_job(sid) is None


def test_stale_empty_sessions():
    store = _make_store()
    deadline = int(time.time()) + 3600
    empty_sid = store.create_session(_permit_payload(deadline), _requirements(), 1000)
    paid_sid = store.create_session(_permit_payload(deadline), _requirements(), 1000)
    store.add_cost(paid_sid, 10)
    # Force both to look idle.
    for sid in (empty_sid, paid_sid):
        store._sessions[sid].last_activity = time.time() - 10_000

    stale = store.get_stale_empty_sessions(idle_timeout_seconds=100)
    assert empty_sid in stale
    assert paid_sid not in stale  # has cost to settle


# ---------------------------------------------------------------------------
# Settlement lifecycle
# ---------------------------------------------------------------------------


class _FakeSettleResult:
    def __init__(
        self, success: bool, transaction: str = "", error_reason: str | None = None
    ):
        self.success = success
        self.transaction = transaction
        self.error_reason = error_reason


class _FakeHttpServer:
    def __init__(self, result: _FakeSettleResult):
        self._result = result
        self.calls = 0

    def process_settlement(self, payload, requirements, settlement_overrides=None):
        self.calls += 1
        return self._result


class _FakeMiddleware:
    """Stand-in for PaymentMiddleware exposing only what settlement touches."""

    def __init__(self, store: DeadlineAwareSessionStore, result: _FakeSettleResult):
        self._session_store = store
        self._http_server = _FakeHttpServer(result)
        self._session_map_lock = threading.Lock()
        self._payment_to_session: dict[str, str] = {}
        self._session_idle_timeout = 100

    # Bind the patched implementations as methods.
    _settle_session = _patched_settle_session
    _settle_ready_sessions = _patched_settle_ready_sessions


@pytest.fixture(autouse=True)
def _facilitator_url(monkeypatch):
    monkeypatch.setattr(x402_settlement, "_facilitator_url", "http://facilitator.test")


def test_settle_async_job_success(monkeypatch):
    store = _make_store()
    now = int(time.time())
    sid = store.create_session(_permit_payload(now + 3600), _requirements(), 1000)
    store.add_cost(sid, 500)
    store._payment_to_session_key = sid

    mw = _FakeMiddleware(store, _FakeSettleResult(True, transaction="payment-job-1"))
    mw._payment_to_session["key"] = sid

    # Facilitator reports the job succeeded on-chain.
    monkeypatch.setattr(
        x402_settlement,
        "_fetch_settlement_job",
        lambda job_id: {
            "status": "succeeded",
            "result": {"success": True, "transaction": "0xdead"},
        },
    )

    mw._settle_session(store._sessions[sid])

    assert store.get_session(sid) is None  # closed
    assert sid not in mw._payment_to_session
    # Base-store settled marker not observable after close; ensure job cleared.
    assert store.get_settlement_job(sid) is None


def test_settle_async_job_terminal_failure_abandons(monkeypatch):
    store = _make_store()
    now = int(time.time())
    sid = store.create_session(_permit_payload(now + 3600), _requirements(), 1000)
    store.add_cost(sid, 500)

    mw = _FakeMiddleware(store, _FakeSettleResult(True, transaction="payment-job-2"))

    monkeypatch.setattr(
        x402_settlement,
        "_fetch_settlement_job",
        lambda job_id: {
            "status": "succeeded",
            "result": {"success": False, "errorReason": "permit2_deadline_expired"},
        },
    )

    mw._settle_session(store._sessions[sid])
    # Session removed (nothing more to try) — revenue abandoned + logged CRITICAL.
    from x402.session import SessionStore

    assert SessionStore.get_session(store, sid) is None


def test_settle_async_job_failed_redriven_when_time_remains(monkeypatch):
    store = _make_store()
    now = int(time.time())
    sid = store.create_session(_permit_payload(now + 3600), _requirements(), 1000)
    store.add_cost(sid, 500)

    mw = _FakeMiddleware(store, _FakeSettleResult(True, transaction="payment-job-3"))

    monkeypatch.setattr(
        x402_settlement,
        "_fetch_settlement_job",
        lambda job_id: {"status": "failed"},
    )

    mw._settle_session(store._sessions[sid])
    # Deadline is far off, so the session is released for a retry, not closed.
    from x402.session import SessionStore

    session = SessionStore.get_session(store, sid)
    assert session is not None
    assert session.settling is False
    assert store.get_settlement_job(sid) is None


def test_settle_skips_when_deadline_hopeless(monkeypatch):
    store = _make_store()
    now = int(time.time())
    # 5s to deadline — below DEADLINE_HOPELESS_SECONDS (30s).
    sid = store.create_session(_permit_payload(now + 5), _requirements(), 1000)
    store.add_cost(sid, 500)

    called = {"settle": False}

    def _should_not_call(payload, requirements, settlement_overrides=None):
        called["settle"] = True
        return _FakeSettleResult(True)

    mw = _FakeMiddleware(store, _FakeSettleResult(True))
    mw._http_server.process_settlement = _should_not_call  # type: ignore[assignment]

    mw._settle_session(store._sessions[sid])
    assert called["settle"] is False  # never attempted a doomed settlement
    from x402.session import SessionStore

    assert SessionStore.get_session(store, sid) is None  # abandoned


def test_settle_sync_transaction_closes_immediately():
    store = _make_store()
    now = int(time.time())
    sid = store.create_session(_permit_payload(now + 3600), _requirements(), 1000)
    store.add_cost(sid, 500)

    # A real 0x tx hash (sync facilitator) — not a job id.
    mw = _FakeMiddleware(store, _FakeSettleResult(True, transaction="0xabc123"))
    mw._settle_session(store._sessions[sid])

    from x402.session import SessionStore

    assert SessionStore.get_session(store, sid) is None


def test_settle_enqueue_failure_releases_claim():
    store = _make_store()
    now = int(time.time())
    sid = store.create_session(_permit_payload(now + 3600), _requirements(), 1000)
    store.add_cost(sid, 500)

    mw = _FakeMiddleware(
        store, _FakeSettleResult(False, error_reason="facilitator_down")
    )
    mw._settle_session(store._sessions[sid])

    from x402.session import SessionStore

    session = SessionStore.get_session(store, sid)
    assert session is not None  # kept for retry
    assert session.settling is False  # claim released
