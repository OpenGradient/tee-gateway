"""
Payment and network constants for the TEE-LLM gateway.

All on-chain addresses, network IDs, and payment amounts are centralized here
so that operators deploying their own instance have a single file to update.

To receive payments to your own wallet, set the EVM_PAYMENT_ADDRESS environment
variable before starting the server.
"""

import os

# ---------------------------------------------------------------------------
# X402 Facilitator
# ---------------------------------------------------------------------------
# Default fallback only. The live value is injected at runtime via POST /v1/keys
# (facilitator_url field) and used for both x402 payment verification and the
# heartbeat service. Override at the OS level with the FACILITATOR_URL env var,
# or supply it directly in the injection payload.
FACILITATOR_URL = os.getenv("FACILITATOR_URL", "https://facilitator.memchat.io")

# ---------------------------------------------------------------------------
# Network IDs (EIP-155 chain identifiers)
# ---------------------------------------------------------------------------


# Base Mainnet — where OPG payments are accepted
BASE_MAINNET_NETWORK: str = "eip155:8453"

# ---------------------------------------------------------------------------
# Payment recipient
# ---------------------------------------------------------------------------

# Wallet address that receives x402 payments.
# Override with the EVM_PAYMENT_ADDRESS environment variable when deploying
# your own instance.
EVM_PAYMENT_ADDRESS: str = os.getenv(
    "EVM_PAYMENT_ADDRESS",
    "0x9deEBB5D1b22e4a6e027977CeAd13893A7E4cC1a",
)

# ---------------------------------------------------------------------------
# ERC-20 token contract addresses
# ---------------------------------------------------------------------------

# OpenGradient token (OPG) on Base Mainnet
BASE_MAINNET_OPG_ADDRESS: str = "0xFbC2051AE2265686a469421b2C5A2D5462FbF5eB"

# ---------------------------------------------------------------------------
# Token decimal places
# ---------------------------------------------------------------------------

# Maps lowercase contract address → number of decimals for unit conversion.
ASSET_DECIMALS_BY_ADDRESS: dict[str, int] = {
    BASE_MAINNET_OPG_ADDRESS.lower(): 18,  # OPG: 18 decimals (ERC-20 standard)
}

# Fallback for any asset not explicitly listed above
DEFAULT_ASSET_DECIMALS: int = 18

# ---------------------------------------------------------------------------
# Pre-check / static fallback payment amounts (in token smallest units)
#
# These are the *maximum* amounts shown during the x402 payment pre-check.
# Actual per-request costs are calculated dynamically from real token usage
# by compute_session_cost() in pricing.py.
# ---------------------------------------------------------------------------

# /v1/chat/completions — maximum OPG spend per session (18 decimals: 5000000000000000000 = 5 OPG).
# This is the upper-bound amount presented to the client during the x402 pre-check handshake.
# The x402 "upto" scheme allows the actual charge to be any value up to this cap;
# the real per-request cost is settled dynamically by compute_session_cost() in pricing.py
# based on actual token usage, so clients are never overcharged beyond what they consumed.
CHAT_COMPLETIONS_OPG_SESSION_MAX_SPEND: str = "5000000000000000000"

# /v1/ohttp — maximum OPG spend per session (18 decimals: 5000000000000000000 = 5 OPG).
# OHTTP image-generation requests can be much more expensive than text chat,
# so the relay-paid encrypted endpoint needs a higher per-session cap.
OHTTP_OPG_SESSION_MAX_SPEND: str = "5000000000000000000"

# /v1/completions — maximum OPG spend per session (18 decimals: 5000000000000000000 = 5 OPG).
# This is the upper-bound amount presented to the client during the x402 pre-check handshake.
# The x402 "upto" scheme allows the actual charge to be any value up to this cap;
# the real per-request cost is settled dynamically by compute_session_cost() in pricing.py
# based on actual token usage, so clients are never overcharged beyond what they consumed.
COMPLETIONS_OPG_SESSION_MAX_SPEND: str = "5000000000000000000"

# ---------------------------------------------------------------------------
# x402 "upto" session settlement timing
# ---------------------------------------------------------------------------
# A draw-down ("upto") session reuses ONE signed Permit2 authorization whose
# on-chain deadline is fixed at signing time (session creation) and never
# refreshed. If the accumulated tab is settled after that deadline the on-chain
# ``settle`` reverts and the whole session's payment is silently lost. These
# knobs keep the settlement schedule safely inside the signed window.
#
# INVARIANT (must hold or long sessions drop payments):
#     UPTO_SESSION_IDLE_TIMEOUT_SECONDS
#         < UPTO_SETTLEMENT_SAFETY_MARGIN_SECONDS
#         < UPTO_SESSION_MAX_TIMEOUT_SECONDS
# and the safety margin must exceed SETTLEMENT_POLL_TIMEOUT_SECONDS so a
# force-settled tab has time to confirm on-chain before the deadline.

# Validity window signed into the Permit2 authorization (deadline = creation + this).
# Advertised to clients as ``maxTimeoutSeconds`` in the 402 requirements.
UPTO_SESSION_MAX_TIMEOUT_SECONDS: int = 600

# Force-settle a session (and stop accepting new draw-downs against it) this many
# seconds before its authorization deadline, leaving room for on-chain confirmation.
UPTO_SETTLEMENT_SAFETY_MARGIN_SECONDS: int = 120

# Settle a session once it has been idle (no new requests) this long.
UPTO_SESSION_IDLE_TIMEOUT_SECONDS: int = 60

# Deferred settlement is asynchronous on the facilitator (202 + job id). Poll the
# job-status endpoint until the on-chain result is known instead of assuming success.
SETTLEMENT_POLL_INTERVAL_SECONDS: float = 2.0
SETTLEMENT_POLL_TIMEOUT_SECONDS: float = 100.0
