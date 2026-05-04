"""
Smoke test: route a single chat request through the gateway's bytedance path.

Usage:
    ARK_API_KEY=... uv run python scripts/test_bytedance.py
    ARK_API_KEY=... uv run python scripts/test_bytedance.py seed-1.8
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage  # noqa: E402

from tee_gateway.config import ProviderConfig  # noqa: E402
from tee_gateway.llm_backend import (  # noqa: E402
    set_provider_config,
    get_chat_model_cached,
    extract_usage,
)


def main() -> int:
    key = os.environ.get("ARK_API_KEY")
    if not key:
        print("ERROR: ARK_API_KEY env var is not set", file=sys.stderr)
        return 2

    model = sys.argv[1] if len(sys.argv) > 1 else "seed-2.0-lite"
    print(f"Using model: {model}")

    set_provider_config(ProviderConfig(bytedance_api_key=key))

    chat = get_chat_model_cached(model=model, temperature=0.2, max_tokens=128)
    print(f"Instantiated: {type(chat).__name__}")

    response = chat.invoke([HumanMessage(content="Reply with exactly: pong")])
    print("---- response ----")
    print(response.content)
    print("---- usage ----")
    print(extract_usage(response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
