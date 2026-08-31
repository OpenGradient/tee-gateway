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
DEFAULT_HEARTBEAT_BUFFER = (
    300  # 5 minutes — subtracted from time.time() to compensate for enclave clock drift
)
DEFAULT_FACILITATOR_TIMEOUT = 20  # seconds
HEARTBEAT_MAX_RETRIES = 3
HEARTBEAT_RETRY_DELAY = 10  # seconds


@dataclass(frozen=True)
class ProviderConfig:
    """API keys for each supported LLM provider, plus the web-search backend."""

    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    xai_api_key: Optional[str] = None
    bytedance_api_key: Optional[str] = None
    nous_api_key: Optional[str] = None
    zai_api_key: Optional[str] = None
    # Exa, which backs the `web_search` tool the gateway executes inside the
    # enclave for every model (see web_search.py). Not an LLM provider, so it is
    # deliberately absent from initialized_providers() — /health reports it
    # separately as `web_search_enabled`.
    exa_api_key: Optional[str] = None
    # GCP service-account key (full JSON, as a string). When set, Anthropic
    # (Claude) models are served through GCP Vertex AI instead of Anthropic's
    # direct API — anthropic_api_key then acts as an optional fallback for
    # operators that don't route through GCP. Gemini deliberately stays on the
    # direct API (a paid-tier Gemini key already bills to its GCP project).
    # project defaults to the service account's own project_id; location
    # defaults to Vertex's "global" endpoint.
    gcp_service_account_json: Optional[str] = None
    gcp_project_id: Optional[str] = None
    gcp_location: Optional[str] = None

    def vertex_enabled(self) -> bool:
        """Whether Anthropic (Claude) traffic is routed through GCP Vertex AI."""
        return bool(self.gcp_service_account_json)

    def initialized_providers(self) -> list[str]:
        """Return provider names the gateway can serve (API key or Vertex)."""
        providers = []
        if self.openai_api_key:
            providers.append("openai")
        if self.anthropic_api_key or self.vertex_enabled():
            providers.append("anthropic")
        if self.google_api_key:
            providers.append("google")
        if self.xai_api_key:
            providers.append("xai")
        if self.bytedance_api_key:
            providers.append("bytedance")
        if self.nous_api_key:
            providers.append("nous")
        if self.zai_api_key:
            providers.append("zai")
        return providers


@dataclass(frozen=True)
class HeartbeatConfig:
    """Configuration for the on-chain TEE heartbeat service."""

    contract_address: str
    facilitator_url: str
    interval: int = DEFAULT_HEARTBEAT_INTERVAL
    timestamp_buffer: int = DEFAULT_HEARTBEAT_BUFFER
    facilitator_timeout: int = DEFAULT_FACILITATOR_TIMEOUT
