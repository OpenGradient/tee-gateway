"""
Runtime configuration objects for the TEE gateway.

These are populated once at startup via POST /v1/keys and passed
explicitly to the subsystems that need them, rather than being stored
as environment variables.
"""

from dataclasses import dataclass
from typing import Optional

# Heartbeat defaults
DEFAULT_HEARTBEAT_INTERVAL = 900  # 15 minutes
DEFAULT_FACILITATOR_TIMEOUT = 20  # seconds


@dataclass(frozen=True)
class ProviderConfig:
    """API keys for each supported LLM provider."""

    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    xai_api_key: Optional[str] = None


@dataclass(frozen=True)
class HeartbeatConfig:
    """Configuration for the on-chain TEE heartbeat service."""

    contract_address: str
    facilitator_url: str
    interval: int = DEFAULT_HEARTBEAT_INTERVAL
    facilitator_timeout: int = DEFAULT_FACILITATOR_TIMEOUT
