"""
Default configuration for the OpenAPI server controllers.
"""

import os

# The internal LLM backend server (server.py running inside the enclave).
# This is a temporary setup - controllers will eventually call LangChain directly.
HTTP_BACKEND_SERVER = os.getenv("LLM_BACKEND_URL", "http://127.0.0.1:8001")
