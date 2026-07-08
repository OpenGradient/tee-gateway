"""Deadline-aware x402 upto-session settlement for the tee-gateway.

og-x402's stock session middleware has three failure modes that silently drop
settlements (observed as "long run of requests -> no settlement transaction"):

1. A session can outlive its Permit2 deadline. The permit is signed on the
   first request (deadline = now + max_timeout_seconds) but the session only
   settles after the *last* request goes idle. Any session that stays busy
   longer than the permit lifetime can never be settled on-chain: the
   facilitator (and Permit2 itself) reject the expired authorization.

2. The facilitator's async ``POST /settle`` returns ``202 Accepted`` with a
   queue job id, and og-x402 maps any 202 to ``success=True``. The middleware
   then marks the session settled and deletes it at *enqueue* time — if the
   settlement job later fails on-chain, the revenue is silently lost and
   nothing retries.

3. ``SessionStore.add_cost`` refuses a cost that would cross the spend cap
   after the response already streamed. The session then never reaches
   ``is_exhausted`` and keeps serving requests it cannot bill.

This module fixes all three without forking og-x402 (which is a pinned
dependency), following the same monkey-patch pattern ``__main__`` already
uses for the middleware:

- :class:`DeadlineAwareSessionStore` tracks each session's permit deadline,
  hides sessions from the serving paths before the deadline becomes
  unsettleable (clients get a 402 and sign a fresh permit), and saturates the
  spend cap instead of refusing it so exhausted sessions settle promptly.
- :func:`install_deadline_aware_settlement` replaces the middleware's reaper
  settlement with a version that settles sessions *before* their permit
  deadline and polls the facilitator job to a terminal state before closing a
  session, re-driving settlement when a job fails.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import httpx

import x402.http.middleware.flask as x402_flask
from x402.schemas import (
    PaymentPayload,
    PaymentRequirements,
    SettlementOverrides,
)
from x402.session import SessionStore, UptoSession

logger = logging.getLogger(__name__)

# Start settling a session this many seconds before its permit deadline, even
# if it is still active. Covers reaper cadence + facilitator queue latency +
# one on-chain confirmation with slack (the facilitator refuses permits within
# 6s of deadline).
SETTLE_MARGIN_SECONDS = 120

# Stop *serving* a session this many seconds before its permit deadline. Must
# exceed SETTLE_MARGIN_SECONDS by at least the longest possible request
# (gateway read timeout is 240s), so an in-flight request can still bill its
# cost before the reaper claims the session for settlement.
SERVE_MARGIN_SECONDS = 420

# Once we are within this window of the deadline (facilitator floor is 6s),
# settlement can no longer be enqueued with any chance of success.
DEADLINE_HOPELESS_SECONDS = 30

# Facilitator job polling. The reaper resumes polling on its next cycle if a
# job is still pending when the per-cycle budget runs out.
JOB_POLL_INTERVAL_SECONDS = 2.0
JOB_POLL_BUDGET_SECONDS = 30.0
JOB_STATUS_TIMEOUT_SECONDS = 10.0

_ASYNC_JOB_ID_PREFIXES = ("payment-", "payment:")

_facilitator_url: str | None = None
_install_lock = threading.Lock()
_installed = False


def _extract_permit_deadline(permit_payload: dict[str, Any]) -> int | None:
    """Pull the Permit2 deadline (unix seconds) out of a stored payment payload."""
    payload = permit_payload.get("payload")
    if not isinstance(payload, dict):
        return None
    authorization = payload.get("permit2Authorization")
    if not isinstance(authorization, dict):
        return None
    deadline = authorization.get("deadline")
    try:
        return int(str(deadline))
    except (TypeError, ValueError):
        return None


def _is_async_job_id(transaction: str) -> bool:
    """True when a settle 'transaction' is actually a facilitator queue job id."""
    return transaction.startswith(_ASYNC_JOB_ID_PREFIXES)


class DeadlineAwareSessionStore(SessionStore):
    """In-memory upto session store that respects Permit2 deadlines.

    Adds three behaviors over the base store:

    - Sessions whose permit deadline is within ``serve_margin_seconds`` are
      hidden from ``get_session`` (the middleware's serving paths), so the
      client receives a 402 and signs a fresh permit instead of accruing cost
      that can never be settled.
    - ``add_cost`` saturates at the spend cap instead of refusing, so a
      session that hits its cap becomes ``is_exhausted`` and settles promptly
      rather than serving unbillable requests.
    - Tracks the facilitator settlement job id per session so an in-flight
      async settlement can be polled to a terminal state across reaper
      cycles.
    """

    def __init__(
        self,
        serve_margin_seconds: int = SERVE_MARGIN_SECONDS,
        settle_margin_seconds: int = SETTLE_MARGIN_SECONDS,
    ) -> None:
        """Create the store.

        Args:
            serve_margin_seconds: Hide sessions from serving this long before
                their permit deadline.
            settle_margin_seconds: Settle sessions this long before their
                permit deadline.
        """
        super().__init__()
        self._serve_margin = serve_margin_seconds
        self._settle_margin = settle_margin_seconds
        self._deadlines: dict[str, int] = {}
        self._settlement_jobs: dict[str, str] = {}

    def create_session(
        self,
        permit_payload: dict[str, Any],
        requirements: dict[str, Any],
        max_amount: int,
        route_method: str | None = None,
        route_path: str | None = None,
    ) -> str:
        """Create a session and record its permit deadline."""
        session_id = super().create_session(
            permit_payload,
            requirements,
            max_amount,
            route_method,
            route_path,
        )
        deadline = _extract_permit_deadline(permit_payload)
        if deadline is None:
            logger.warning(
                "UPTO_SESSION_NO_DEADLINE id=%s — permit deadline not found in "
                "payload; deadline-based settlement gating disabled for this "
                "session",
                session_id,
            )
        else:
            with self._lock:
                self._deadlines[session_id] = deadline
            logger.info(
                "UPTO_SESSION_DEADLINE id=%s deadline=%d (in %ds)",
                session_id,
                deadline,
                int(deadline - time.time()),
            )
        return session_id

    def get_session(self, session_id: str) -> UptoSession | None:
        """Return a session for serving, unless its permit is near expiry.

        Hiding the session makes both middleware serving paths
        (payment-header lookup and ``X-Upto-Session`` resume) respond 402, so
        the client signs a fresh permit and continues on a new session while
        the old one is settled by the reaper.
        """
        session = super().get_session(session_id)
        if session is None:
            return None
        with self._lock:
            deadline = self._deadlines.get(session_id)
        if deadline is not None and time.time() >= deadline - self._serve_margin:
            logger.info(
                "UPTO_SESSION_SERVE_CUTOFF id=%s deadline=%d — refusing new "
                "requests so the session can settle before its permit expires",
                session_id,
                deadline,
            )
            return None
        return session

    def add_cost(self, session_id: str, cost: int) -> bool:
        """Add cost to a session, saturating at the spend cap.

        Unlike the base implementation, a cost that would cross the cap bills
        the remaining budget (the response has already been served) and marks
        the session exhausted, instead of refusing and leaving the session
        permanently un-exhaustible. The unbilled overflow is logged.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.settled or session.settling:
                return False
            remaining = session.max_amount - session.accumulated_cost
            if remaining <= 0:
                return False
            if cost > remaining:
                logger.warning(
                    "UPTO_SESSION_CAP_SATURATED id=%s billed=%d unbilled=%d "
                    "cap=%d — session is now exhausted and will settle",
                    session_id,
                    remaining,
                    cost - remaining,
                    session.max_amount,
                )
                session.accumulated_cost = session.max_amount
            else:
                session.accumulated_cost += cost
            session.last_activity = time.time()
            return True

    def close_session(self, session_id: str) -> UptoSession | None:
        """Close a session and drop its deadline/job bookkeeping."""
        session = super().close_session(session_id)
        with self._lock:
            self._deadlines.pop(session_id, None)
            self._settlement_jobs.pop(session_id, None)
        return session

    def session_deadline(self, session_id: str) -> int | None:
        """Return the recorded permit deadline for a session, if known."""
        with self._lock:
            return self._deadlines.get(session_id)

    def set_settlement_job(self, session_id: str, job_id: str) -> None:
        """Record the facilitator settlement job in flight for a session."""
        with self._lock:
            self._settlement_jobs[session_id] = job_id

    def get_settlement_job(self, session_id: str) -> str | None:
        """Return the in-flight facilitator settlement job id, if any."""
        with self._lock:
            return self._settlement_jobs.get(session_id)

    def clear_settlement_job(self, session_id: str) -> None:
        """Forget the in-flight settlement job for a session."""
        with self._lock:
            self._settlement_jobs.pop(session_id, None)

    def get_deadline_due_sessions(self) -> list[UptoSession]:
        """Sessions that must settle now to beat their permit deadline."""
        now = time.time()
        due: list[UptoSession] = []
        with self._lock:
            for session_id, session in self._sessions.items():
                if session.settled or session.settling:
                    continue
                if session.accumulated_cost <= 0:
                    continue
                deadline = self._deadlines.get(session_id)
                if deadline is not None and now >= deadline - self._settle_margin:
                    due.append(session)
        return due

    def get_sessions_with_pending_jobs(self) -> list[UptoSession]:
        """Sessions claimed for settlement whose facilitator job is unresolved."""
        with self._lock:
            return [
                self._sessions[session_id]
                for session_id in self._settlement_jobs
                if session_id in self._sessions
            ]

    def get_stale_empty_sessions(self, idle_timeout_seconds: int) -> list[str]:
        """Idle sessions with nothing to settle (safe to just drop)."""
        now = time.time()
        stale: list[str] = []
        with self._lock:
            for session_id, session in self._sessions.items():
                if session.settling or session.settled:
                    continue
                if session.accumulated_cost > 0:
                    continue
                if now - session.last_activity > idle_timeout_seconds:
                    stale.append(session_id)
        return stale


def _cleanup_payment_mapping(middleware: Any, session_id: str) -> None:
    """Drop payment-header -> session mappings pointing at a session."""
    with middleware._session_map_lock:
        middleware._payment_to_session = {
            key: value
            for key, value in middleware._payment_to_session.items()
            if value != session_id
        }


def _complete_settled_session(
    middleware: Any,
    store: SessionStore,
    session: UptoSession,
    transaction: str,
) -> None:
    """Mark a session settled with a confirmed transaction and remove it."""
    store.mark_settled(session.session_id, transaction or "")
    store.close_session(session.session_id)
    _cleanup_payment_mapping(middleware, session.session_id)
    logger.info(
        "UPTO_SESSION_SETTLED id=%s tx=%s amount=%d",
        session.session_id,
        transaction,
        session.accumulated_cost,
    )


def _abandon_session(
    middleware: Any,
    store: SessionStore,
    session: UptoSession,
    reason: str,
) -> None:
    """Give up on settling a session — accrued revenue is lost. Log loudly."""
    logger.critical(
        "UPTO_SESSION_SETTLEMENT_ABANDONED id=%s amount=%d reason=%s — "
        "accrued cost will NOT be collected",
        session.session_id,
        session.accumulated_cost,
        reason,
    )
    store.close_session(session.session_id)
    _cleanup_payment_mapping(middleware, session.session_id)


def _fetch_settlement_job(job_id: str) -> dict[str, Any] | None:
    """Fetch facilitator job status; None on transport errors (retry later)."""
    if not _facilitator_url:
        return None
    try:
        response = httpx.get(
            f"{_facilitator_url}/settle/{job_id}",
            timeout=JOB_STATUS_TIMEOUT_SECONDS,
        )
        if response.status_code == 404:
            return {"status": "not_found"}
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else None
    except Exception as exc:
        logger.warning("UPTO_SETTLEMENT_JOB_POLL_ERROR job=%s error=%s", job_id, exc)
        return None


def _deadline_is_hopeless(store: SessionStore, session_id: str) -> bool:
    """True when the permit deadline is too close for settlement to succeed."""
    if not isinstance(store, DeadlineAwareSessionStore):
        return False
    deadline = store.session_deadline(session_id)
    return deadline is not None and time.time() >= deadline - DEADLINE_HOPELESS_SECONDS


def _finalize_settlement_job(
    middleware: Any,
    store: DeadlineAwareSessionStore,
    session: UptoSession,
) -> None:
    """Poll an in-flight facilitator settlement job to a terminal state.

    Keeps the session claimed (``settling=True``) while the job is pending so
    no cost is added under it mid-settlement. If the polling budget for this
    reaper cycle runs out the job id stays recorded and the next cycle
    resumes polling.
    """
    job_id = store.get_settlement_job(session.session_id)
    if not job_id:
        return

    poll_deadline = time.monotonic() + JOB_POLL_BUDGET_SECONDS
    while True:
        job = _fetch_settlement_job(job_id)
        if job is None:
            return  # transport error — retry next reaper cycle

        status = job.get("status")

        if status == "succeeded":
            result = job.get("result")
            result = result if isinstance(result, dict) else {}
            if result.get("success") is False:
                # Terminal on-chain failure (worker completed the job with a
                # non-retryable error such as an expired permit deadline).
                reason = result.get("errorReason") or "unknown_settlement_error"
                _abandon_session(
                    middleware,
                    store,
                    session,
                    f"terminal settlement failure: {reason} (job={job_id})",
                )
            else:
                _complete_settled_session(
                    middleware, store, session, str(result.get("transaction") or "")
                )
            return

        if status in ("failed", "not_found"):
            # The queue exhausted its retry attempts (or lost the job).
            # Re-drive settlement with a fresh job on the next reaper cycle
            # while the permit can still settle; otherwise give up loudly.
            store.clear_settlement_job(session.session_id)
            if _deadline_is_hopeless(store, session.session_id):
                _abandon_session(
                    middleware,
                    store,
                    session,
                    f"settlement job {status} and permit deadline passed (job={job_id})",
                )
            else:
                logger.error(
                    "UPTO_SETTLEMENT_JOB_%s id=%s job=%s — will re-enqueue settlement",
                    status.upper(),
                    session.session_id,
                    job_id,
                )
                store.clear_settling(session.session_id)
            return

        # queued / processing
        if time.monotonic() >= poll_deadline:
            logger.info(
                "UPTO_SETTLEMENT_JOB_PENDING id=%s job=%s status=%s — resuming "
                "poll next reaper cycle",
                session.session_id,
                job_id,
                status,
            )
            return
        time.sleep(JOB_POLL_INTERVAL_SECONDS)


def _patched_settle_session(self: Any, session: UptoSession) -> None:
    """Settle one session, confirming the facilitator job before closing it.

    Replaces ``PaymentMiddleware._settle_session``. The stock version treats
    the facilitator's 202 (job enqueued) as settled and deletes the session;
    this version records the job id and only closes the session once the job
    reaches a terminal state.
    """
    if session.settled or session.settling or session.accumulated_cost <= 0:
        return

    store = self._session_store
    assert store is not None

    if _deadline_is_hopeless(store, session.session_id):
        _abandon_session(
            self,
            store,
            session,
            "permit deadline passed before settlement could be enqueued",
        )
        return

    claimed = False
    mark_settling = getattr(store, "mark_settling", None)
    clear_settling = getattr(store, "clear_settling", None)
    if callable(mark_settling):
        claimed = bool(mark_settling(session.session_id))
        if not claimed:
            return

    try:
        payload = PaymentPayload.model_validate(session.permit_payload)
        requirements = PaymentRequirements.model_validate(session.requirements)
        usage_metadata = dict(session.usage_metadata or {})
        if usage_metadata:
            usage_metadata.setdefault("session_id", session.session_id)
            usage_metadata.setdefault("cost_opg", str(session.accumulated_cost))
        overrides = SettlementOverrides(
            amount=str(session.accumulated_cost),
            usage_metadata=usage_metadata or None,
        )

        settle_result = self._http_server.process_settlement(
            payload,
            requirements,
            settlement_overrides=overrides,
        )

        if not settle_result.success:
            if callable(clear_settling):
                clear_settling(session.session_id)
            logger.error(
                "UPTO_SETTLEMENT_ENQUEUE_FAILED id=%s error=%s",
                session.session_id,
                settle_result.error_reason,
            )
            return

        transaction = (settle_result.transaction or "").strip()
        if isinstance(store, DeadlineAwareSessionStore) and _is_async_job_id(
            transaction
        ):
            # Async facilitator: 'transaction' is a queue job id, not a
            # settlement. Track it and confirm before closing the session.
            store.set_settlement_job(session.session_id, transaction)
            logger.info(
                "UPTO_SETTLEMENT_ENQUEUED id=%s job=%s amount=%d",
                session.session_id,
                transaction,
                session.accumulated_cost,
            )
            _finalize_settlement_job(self, store, session)
        else:
            # Sync facilitator (or zero-amount): transaction is final.
            _complete_settled_session(self, store, session, transaction)
    except Exception:
        if callable(clear_settling):
            clear_settling(session.session_id)
        logger.exception("Error settling session %s", session.session_id)


def _patched_settle_ready_sessions(self: Any) -> None:
    """Settle idle, exhausted, and deadline-due sessions; poll pending jobs.

    Replaces ``PaymentMiddleware._settle_ready_sessions``. Adds two triggers
    to the stock idle/exhausted pair: sessions approaching their permit
    deadline, and sessions whose async settlement job is still unresolved.
    Also drops idle sessions with nothing to settle so they don't accumulate.
    """
    store = self._session_store
    if store is None:
        return

    due: dict[str, UptoSession] = {}
    for session in store.get_expired_sessions(self._session_idle_timeout):
        due[session.session_id] = session
    for session in store.get_exhausted_sessions():
        due[session.session_id] = session
    if isinstance(store, DeadlineAwareSessionStore):
        for session in store.get_deadline_due_sessions():
            due[session.session_id] = session

    for session in due.values():
        self._settle_session(session)

    if isinstance(store, DeadlineAwareSessionStore):
        # Resume polling settlements still in flight from earlier cycles.
        for session in store.get_sessions_with_pending_jobs():
            if session.settling and not session.settled:
                _finalize_settlement_job(self, store, session)
        # Idle sessions that never accrued cost have nothing to settle.
        for session_id in store.get_stale_empty_sessions(self._session_idle_timeout):
            store.close_session(session_id)
            _cleanup_payment_mapping(self, session_id)


def install_deadline_aware_settlement(facilitator_url: str) -> None:
    """Patch og-x402's Flask PaymentMiddleware with deadline-aware settlement.

    Must be called before the middleware instance handles traffic. Idempotent.

    Args:
        facilitator_url: Base URL of the x402 facilitator, used to poll
            ``GET /settle/<jobId>`` for settlement job outcomes.
    """
    global _facilitator_url, _installed
    with _install_lock:
        _facilitator_url = facilitator_url.rstrip("/")
        if _installed:
            return
        setattr(
            x402_flask.PaymentMiddleware,
            "_settle_session",
            _patched_settle_session,
        )
        setattr(
            x402_flask.PaymentMiddleware,
            "_settle_ready_sessions",
            _patched_settle_ready_sessions,
        )
        _installed = True
        logger.info(
            "Deadline-aware x402 settlement installed (facilitator=%s, "
            "settle_margin=%ds, serve_margin=%ds)",
            _facilitator_url,
            SETTLE_MARGIN_SECONDS,
            SERVE_MARGIN_SECONDS,
        )
