"""Guard tests for the x402 "upto" session settlement timing constants.

A draw-down ("upto") session reuses ONE signed Permit2 authorization whose
on-chain deadline is fixed at session creation (created_at + max_timeout) and
never refreshed. If the accumulated tab is settled after that deadline the
on-chain ``settle`` reverts and the whole session's payment is silently lost.

These tests lock in the ordering invariant between the settlement timing knobs
so a future edit to definitions.py can't silently reintroduce the drop.
"""

from tee_gateway.definitions import (
    SETTLEMENT_POLL_TIMEOUT_SECONDS,
    UPTO_SESSION_IDLE_TIMEOUT_SECONDS,
    UPTO_SESSION_MAX_TIMEOUT_SECONDS,
    UPTO_SETTLEMENT_SAFETY_MARGIN_SECONDS,
)


def test_timing_constants_are_positive():
    """Every timing knob must be a positive duration."""
    assert UPTO_SESSION_IDLE_TIMEOUT_SECONDS > 0
    assert UPTO_SETTLEMENT_SAFETY_MARGIN_SECONDS > 0
    assert UPTO_SESSION_MAX_TIMEOUT_SECONDS > 0
    assert SETTLEMENT_POLL_TIMEOUT_SECONDS > 0


def test_idle_timeout_below_safety_margin():
    """An idle session must settle (idle timeout) well before the force-settle
    safety window, so normal traffic settles via idle and never races the
    deadline."""
    assert UPTO_SESSION_IDLE_TIMEOUT_SECONDS < UPTO_SETTLEMENT_SAFETY_MARGIN_SECONDS


def test_safety_margin_below_max_timeout():
    """The force-settle point (deadline - safety_margin) must be strictly before
    the authorization deadline, leaving a positive window to settle in."""
    assert UPTO_SETTLEMENT_SAFETY_MARGIN_SECONDS < UPTO_SESSION_MAX_TIMEOUT_SECONDS


def test_safety_margin_covers_poll_timeout():
    """Force-settling starts safety_margin seconds before the deadline, and the
    reaper then polls the facilitator for up to SETTLEMENT_POLL_TIMEOUT_SECONDS
    to confirm the on-chain result. The margin must exceed the poll timeout so a
    force-settled tab can confirm before its authorization expires."""
    assert UPTO_SETTLEMENT_SAFETY_MARGIN_SECONDS > SETTLEMENT_POLL_TIMEOUT_SECONDS


def test_full_ordering_invariant():
    """The complete invariant documented in definitions.py, in one assertion:

    poll_timeout < safety_margin < max_timeout
    idle_timeout < safety_margin
    """
    assert (
        UPTO_SESSION_IDLE_TIMEOUT_SECONDS
        < UPTO_SETTLEMENT_SAFETY_MARGIN_SECONDS
        < UPTO_SESSION_MAX_TIMEOUT_SECONDS
    )
    assert SETTLEMENT_POLL_TIMEOUT_SECONDS < UPTO_SETTLEMENT_SAFETY_MARGIN_SECONDS
