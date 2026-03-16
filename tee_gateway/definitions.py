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
FACILITATOR_URL = os.getenv("FACILITATOR_URL", "https://facilitator.memchat.io")

# ---------------------------------------------------------------------------
# Network IDs (EIP-155 chain identifiers)
# ---------------------------------------------------------------------------

# OG EVM — where USDC payments are accepted
EVM_NETWORK: str = "eip155:10740"

# Base Testnet — where OPG payments are accepted
BASE_TESTNET_NETWORK: str = "eip155:84532"

# ---------------------------------------------------------------------------
# Payment recipient
# ---------------------------------------------------------------------------

# Wallet address that receives x402 payments.
# Override with the EVM_PAYMENT_ADDRESS environment variable when deploying
# your own instance.
EVM_PAYMENT_ADDRESS: str = os.getenv(
    "EVM_PAYMENT_ADDRESS",
    "0x40eFb45552EDfB2502D90A657a8ab41F03ec460d",
)

# ---------------------------------------------------------------------------
# ERC-20 token contract addresses
# ---------------------------------------------------------------------------

# USDC Address
USDC_ADDRESS: str = "0x094E464A23B90A71a0894D5D1e5D470FfDD074e1"

# OpenGradient token (OPG) on Base Testnet
BASE_OPG_ADDRESS: str = "0x240b09731D96979f50B2C649C9CE10FcF9C7987F"

# ---------------------------------------------------------------------------
# Token decimal places
# ---------------------------------------------------------------------------

# Maps lowercase contract address → number of decimals for unit conversion.
ASSET_DECIMALS_BY_ADDRESS: dict[str, int] = {
    USDC_ADDRESS.lower(): 6,  # USDC / OUSDC standard: 6 decimals
    BASE_OPG_ADDRESS.lower(): 18,  # OPG: 18 decimals (ERC-20 standard)
}

# Fallback for any asset not explicitly listed above
DEFAULT_ASSET_DECIMALS: int = 18

# ---------------------------------------------------------------------------
# Pre-check / static fallback payment amounts (in token smallest units)
#
# These are the *maximum* amounts shown during the x402 payment pre-check.
# Actual per-request costs are calculated dynamically from real token usage
# by dynamic_session_cost_calculator() in util.py.
# ---------------------------------------------------------------------------

# /v1/chat/completions — 0.01 OUSDC precheck (6 decimals: 10_000 = $0.01)
CHAT_COMPLETIONS_USDC_AMOUNT: str = "10000"

# /v1/chat/completions — 0.05 OPG precheck (18 decimals)
CHAT_COMPLETIONS_OPG_AMOUNT: str = "50000000000000000"

# /v1/completions — 0.01 USDC precheck (6 decimals: 10_000 = $0.01)
COMPLETIONS_USDC_AMOUNT: str = "10000"
