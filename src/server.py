#!/usr/bin/env python3
"""
TEE LLM Router Server
Runs within a Nitro Enclave with remote attestation and request signing.
Based on LangChain routing with async support and cryptographic verification.

When TEE_ENABLED=false, runs as a plain LLM routing backend without
nitriding registration or attestation.
"""

import os
import logging
import json
import hashlib
import base64
import asyncio
import sys
import threading
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
import urllib.request
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
import psutil
import gc
import time
import httpx

from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend
from eth_hash.auto import keccak

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_xai import ChatXAI


# Check if TEE features are enabled
TEE_ENABLED = os.getenv("TEE_ENABLED", "true").lower() != "false"

# HTTP Client Configuration
ANTHROPIC_TIMEOUT = 120.0

TIMEOUT = httpx.Timeout(
    timeout=120.0,
    connect=15.0,
    read=15.0,
    write=30.0,
    pool=10.0,
)

LIMITS = httpx.Limits(
    max_keepalive_connections=10,
    max_connections=50,
    keepalive_expiry=60 * 20,  # 20 minutes
)

# Shared HTTP clients for each provider
openai_http_client = httpx.AsyncClient(
    base_url="https://api.openai.com/v1",
    headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"},
    timeout=TIMEOUT,
    limits=LIMITS,
    http2=True,
    follow_redirects=False,
)

xai_http_client = httpx.AsyncClient(
    base_url="https://api.x.ai/v1",
    headers={"Authorization": f"Bearer {os.getenv('XAI_API_KEY', '')}"},
    timeout=TIMEOUT,
    limits=LIMITS,
    http2=True,
    follow_redirects=False,
)

# One-time key injection guard
_keys_initialized: bool = False
_keys_lock = threading.Lock()

# Store server start time for uptime calculation
SERVER_START_TIME = time.time()

class UptimeFormatter(logging.Formatter):
    """Custom formatter that includes uptime since server start"""
    
    def format(self, record):
        uptime = time.time() - SERVER_START_TIME
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        seconds = int(uptime % 60)
        record.uptime = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return super().format(record)

# Configure logging
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(UptimeFormatter(
    fmt='%(asctime)s [+%(uptime)s] [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.handlers.clear()
root_logger.addHandler(log_handler)

app = FastAPI(title="TEE LLM Router", version="1.0.0")

# Global rate limiter (100 requests per minute)
rate_limiter = InMemoryRateLimiter(
    requests_per_second=100 / 60,
    check_every_n_seconds=1,
    max_bucket_size=100
)

# TEE Cryptographic state
class TEEKeyManager:
    """Manages private/public key pair for TEE attestation and signing"""
    
    def __init__(self, register_nitriding=True):
        self.private_key = None
        self.public_key = None
        self.public_key_pem = None
        self._initialize_keys()
        if register_nitriding:
            self.register_with_nitriding()
    
    def _initialize_keys(self):
        """Generate RSA key pair for signing inference results"""
        logger.info("Generating TEE RSA key pair...")
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
        
        self.public_key_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')
        
        logger.info("TEE key pair generated successfully")

    def register_with_nitriding(self):
        """Register public key hash with nitriding"""
        try:
            public_key_der = self.public_key.public_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            key_hash = hashlib.sha256(public_key_der).digest()
            key_hash_b64 = base64.b64encode(key_hash).decode('utf-8')
            
            logger.info(f"Public key DER length: {len(public_key_der)} bytes")
            logger.info(f"Public key SHA256 hash (hex): {key_hash.hex()}")
            logger.info(f"Public key SHA256 hash (base64): {key_hash_b64}")
            
            nitriding_hash_url = "http://127.0.0.1:8080/enclave/hash"
            
            req = urllib.request.Request(
                nitriding_hash_url,
                data=key_hash_b64.encode('utf-8'),
                method='POST'
            )
            
            response = urllib.request.urlopen(req, timeout=5)
            response_body = response.read().decode('utf-8')
            
            if response.getcode() == 200:
                logger.info("Successfully registered public key hash with nitriding")
                logger.info(f"Response: {response_body}")
                return True
            else:
                logger.error(f"Failed to register public key hash: HTTP {response.getcode()}")
                logger.error(f"Response: {response_body}")
                return False
                
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else "No error body"
            logger.error(f"HTTP Error {e.code}: {e.reason}")
            logger.error(f"Error response: {error_body}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error registering public key: {e}", exc_info=True)
            return False

    def sign_data(self, data: bytes) -> str:
        """Sign msg_hash bytes with RSA-PSS-SHA256, return base64 signature.

        Expects pre-computed bytes (e.g. the keccak256 msg_hash).
        Internally the RSA-PSS layer hashes again with SHA256 before signing,
        matching the double-hash pattern the on-chain verifier uses:
          keccak256(abi.encodePacked(inputHash, outputHash, timestamp))  → RSA-PSS-SHA256(msg_hash)
        """
        signature = self.private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode('utf-8')
    
    def get_public_key(self) -> str:
        """Return public key in PEM format"""
        return self.public_key_pem

# Initialize key manager (skip nitriding registration when TEE_ENABLED=false)
tee_keys = TEEKeyManager(register_nitriding=TEE_ENABLED)

# Nitriding readiness signal
nitriding_url = "http://127.0.0.1:8080/enclave/ready"

def signal_ready():
    """Signal to nitriding that enclave is ready"""
    if not TEE_ENABLED:
        logger.info("TEE disabled - skipping nitriding ready signal")
        return
    try:
        r = urllib.request.urlopen(nitriding_url)
        if r.getcode() != 200:
            raise Exception(f"Expected status code 200 but got {r.getcode()}")
        logger.info("Successfully signaled ready to nitriding")
    except Exception as e:
        logger.warning(f"Could not signal nitriding (may not be in TEE): {e}")


# Pydantic Models
class Message(BaseModel):
    role: str
    content: str
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class Tool(BaseModel):
    type: str = "function"
    function: Dict[str, Any]


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[str] = "auto"


class CompletionResponse(BaseModel):
    completion: str
    model: str
    timestamp: int       # unix seconds (uint256) — matches on-chain timestamp
    signature: str       # base64 RSA-PSS-SHA256(keccak256(inputHash‖outputHash‖timestamp))
    request_hash: str    # hex keccak256(request_bytes) — inputHash for on-chain verification
    output_hash: str     # hex keccak256(response_content) — outputHash for on-chain verification
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, str]] = None


class ChatResponse(BaseModel):
    finish_reason: str
    message: Dict[str, Any]
    model: str
    timestamp: int       # unix seconds (uint256) — matches on-chain timestamp
    signature: str       # base64 RSA-PSS-SHA256(keccak256(inputHash‖outputHash‖timestamp))
    request_hash: str    # hex keccak256(request_bytes) — inputHash for on-chain verification
    output_hash: str     # hex keccak256(response_content) — outputHash for on-chain verification
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, str]] = None


class AttestationResponse(BaseModel):
    """TEE attestation document"""
    public_key: str
    timestamp: str
    enclave_info: Dict[str, Any]
    measurements: Optional[Dict] = None


class ProviderKeysRequest(BaseModel):
    """One-time API key injection request"""
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    xai_api_key: Optional[str] = None


# Helper functions
def compute_tee_msg_hash(
    request_bytes: bytes,
    response_content: str,
    timestamp: int,
) -> tuple:
    """Compute msg_hash matching the on-chain verifier:
      keccak256(abi.encodePacked(inputHash, outputHash, timestamp))
    where inputHash and outputHash are each keccak256 bytes32 values
    and timestamp is uint256 (big-endian 32 bytes).

    Returns (msg_hash_bytes, input_hash_hex, output_hash_hex).
    """
    input_hash  = keccak(request_bytes)                        # bytes32
    output_hash = keccak(response_content.encode('utf-8'))     # bytes32
    msg_hash    = keccak(input_hash + output_hash + timestamp.to_bytes(32, 'big'))
    return msg_hash, input_hash.hex(), output_hash.hex()


def get_provider_from_model(model: str) -> str:
    """Infer provider from model name"""
    model_lower = model.lower()
    
    if "gpt" in model_lower or model.startswith("openai/") or model_lower.startswith("o3") or model_lower.startswith("o4"):
        return "openai"
    elif "claude" in model_lower or model.startswith("anthropic/"):
        return "anthropic"
    elif "gemini" in model_lower or model.startswith("google/") or "google" in model_lower:
        return "google"
    elif "grok" in model_lower or model.startswith("x-ai/"):
        return "x-ai"
    else:
        return "openai"


@lru_cache(maxsize=32)
def get_chat_model_cached(model: str, temperature: float, max_tokens: int):
    """
    Get cached chat model instance using environment API keys.
    Models are cached by (model, temperature, max_tokens) tuple.
    """
    provider = get_provider_from_model(model)
    
    logger.info(f"Creating cached chat model - Provider: {provider}, Model: {model}")
    
    if provider in ["google", "gemini"]:
        alias_map = {
            "gemini-2.5-flash":         "gemini-2.5-flash",
            "gemini-2.5-flash-lite":    "gemini-2.5-flash-lite",
            "gemini-2.5-pro":           "gemini-2.5-pro",
            "gemini-3-pro-preview":     "gemini-3-pro-preview",
            "gemini-3-flash-preview":   "gemini-3-flash-preview",
        }
        resolved_model = alias_map.get(model, model)
        thinking_budget = None
        if "2.5-flash" in model or "flash-lite" in model:
            thinking_budget = 0
        elif "2.5-pro" in model:
            thinking_budget = 128
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
            
        return ChatGoogleGenerativeAI(
            model=resolved_model,
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
            thinking_budget=thinking_budget,
            include_thoughts=False if thinking_budget is not None else None,
        )
        
    elif provider == "openai":
        model_temp = 1.0 if model in ["o4-mini", "o3", "o4", "o4-5"] else temperature 
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=model_temp,
            max_tokens=max_tokens,
            http_async_client=openai_http_client,
            streaming=True,
            stream_usage=True,
        )
        
    elif provider == "anthropic":
        alias_map = {
            "claude-3.7-sonnet":    "claude-3-7-sonnet-latest",
            "claude-3.5-haiku":     "claude-3-5-haiku-latest",
            "claude-4.0-sonnet":    "claude-sonnet-4-0",
            "claude-sonnet-4-5":    "claude-sonnet-4-5",
            "claude-sonnet-4-6":    "claude-sonnet-4-6",
            "claude-haiku-4-5":     "claude-haiku-4-5-20251001",
            "claude-opus-4-5":      "claude-opus-4-5-20251101",
            "claude-opus-4-6":      "claude-opus-4-6",
        }
        anthropic_model = alias_map.get(model, model)
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
            
        return ChatAnthropic(
            model=anthropic_model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=ANTHROPIC_TIMEOUT,
            streaming=True,
            stream_usage=True,
        )
        
    elif provider == "x-ai":
        alias_map = {
            "grok-3-mini-beta":                 "grok-3-mini",
            "grok-3-beta":                      "grok-3-latest",
            "grok-2-1212":                      "grok-2-latest",
            "grok-4.1-fast":                    "grok-4-1-fast",
            "grok-4-fast":                      "grok-4-fast",
            "grok-4":                           "grok-4",
            "grok-4-1-fast-non-reasoning":      "grok-4-1-fast-non-reasoning",
        }
        xai_model = alias_map.get(model, model)
        
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY not found in environment")
            
        return ChatXAI(
            model=xai_model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            http_async_client=xai_http_client,
            streaming=True,
            stream_usage=True,
        )
        
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def convert_messages(messages: List[Message]) -> List[Any]:
    """Convert API messages to LangChain message objects"""
    langchain_messages = []
    
    for msg in messages:
        role = msg.role.lower()
        content = msg.content
        
        if role == "system":
            langchain_messages.append(SystemMessage(content=content))
            logger.info(f"Added SystemMessage: {content[:100]}...")
            
        elif role == "user":
            langchain_messages.append(HumanMessage(content=content))
            logger.info(f"Added HumanMessage: {content[:100]}...")
            
        elif role == "assistant":
            if msg.tool_calls:
                logger.info(f"Processing assistant message with {len(msg.tool_calls)} tool calls")
                
                langchain_tool_calls = []
                for tc in msg.tool_calls:
                    logger.info(f"Tool call: id={tc.get('id')}, type={tc.get('type')}, function={tc.get('function', {}).get('name')}")
                    
                    args = tc.get('function', {}).get('arguments', '{}')
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            logger.warning(f"Could not parse tool arguments as JSON: {args}")
                            args = {}
                    
                    langchain_tool_calls.append({
                        "name": tc.get('function', {}).get('name', ''),
                        "args": args,
                        "id": tc.get('id', ''),
                        "type": tc.get('type', 'function')
                    })
                
                ai_msg = AIMessage(
                    content=content or "",
                    tool_calls=langchain_tool_calls
                )
                logger.info(f"Created AIMessage with tool_calls: {langchain_tool_calls}")
            else:
                ai_msg = AIMessage(content=content)
                logger.info(f"Added AIMessage (no tools): {content[:100]}...")
            
            langchain_messages.append(ai_msg)
            
        elif role == "tool":
            logger.info(f"Adding ToolMessage: tool_call_id={msg.tool_call_id}, name={msg.name}, content={content[:100]}...")
            langchain_messages.append(
                ToolMessage(
                    content=content,
                    tool_call_id=msg.tool_call_id or "",
                    name=msg.name or ""
                )
            )
    
    logger.info(f"Converted {len(messages)} messages to {len(langchain_messages)} LangChain messages")
    return langchain_messages


def extract_usage(response: AIMessage) -> Optional[Dict[str, int]]:
    """Extract token usage information from response"""
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        usage_metadata = response.usage_metadata
        return {
            "prompt_tokens": usage_metadata.get("input_tokens", 0),
            "completion_tokens": usage_metadata.get("output_tokens", 0),
            "total_tokens": usage_metadata.get("total_tokens", 0),
        }
    return None


# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    process = psutil.Process()
    connections = process.connections()
    conn_states = {}
    for conn in connections:
        state = str(conn.status)
        conn_states[state] = conn_states.get(state, 0) + 1
    
    return {
        "status": "healthy",
        "version": "1.0.0",
        "tee_enabled": TEE_ENABLED,
        "uptime_seconds": time.time() - process.create_time(),
        "memory_mb": process.memory_info().rss / 1024 / 1024,
        "threads": process.num_threads(),
        "open_files": len(process.open_files()),
        "num_fds": process.num_fds(),
        "connections": len(connections),
        "connection_states": conn_states,
        "gc_objects": len(gc.get_objects()),
    }


@app.get("/signing-key")
async def get_signing_key():
    """Return TEE attestation document with public key"""
    try:
        attestation = AttestationResponse(
            public_key=tee_keys.get_public_key(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            enclave_info={
                "platform": "aws-nitro",
                "instance_type": "tee-enabled",
                "version": "1.0.0"
            },
            measurements=None
        )
        logger.info("Attestation document requested")
        return attestation
    except Exception as e:
        logger.error(f"Attestation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/keys")
async def set_provider_keys(request: ProviderKeysRequest):
    """
    One-time endpoint to inject LLM provider API keys into the enclave.
    After the first successful call this endpoint returns 409 for all
    subsequent requests, ensuring keys cannot be overwritten at runtime.
    """
    global _keys_initialized, openai_http_client, xai_http_client

    with _keys_lock:
        if _keys_initialized:
            raise HTTPException(
                status_code=409,
                detail="Provider keys have already been initialized and cannot be changed",
            )

        if request.openai_api_key:
            os.environ["OPENAI_API_KEY"] = request.openai_api_key
        if request.google_api_key:
            os.environ["GOOGLE_API_KEY"] = request.google_api_key
        if request.anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = request.anthropic_api_key
        if request.xai_api_key:
            os.environ["XAI_API_KEY"] = request.xai_api_key

        # Recreate the shared HTTP clients so the new Authorization headers
        # are picked up (the original clients were built at import time with
        # empty keys from the Dockerfile).
        old_openai = openai_http_client
        old_xai = xai_http_client

        openai_http_client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"},
            timeout=TIMEOUT,
            limits=LIMITS,
            http2=True,
            follow_redirects=False,
        )
        xai_http_client = httpx.AsyncClient(
            base_url="https://api.x.ai/v1",
            headers={"Authorization": f"Bearer {os.environ.get('XAI_API_KEY', '')}"},
            timeout=TIMEOUT,
            limits=LIMITS,
            http2=True,
            follow_redirects=False,
        )

        # Close stale clients and invalidate the model cache so next request
        # creates fresh LangChain models with the correct credentials.
        await old_openai.aclose()
        await old_xai.aclose()
        get_chat_model_cached.cache_clear()

        _keys_initialized = True

    providers_set = [
        p for p, v in {
            "openai": request.openai_api_key,
            "google": request.google_api_key,
            "anthropic": request.anthropic_api_key,
            "xai": request.xai_api_key,
        }.items() if v
    ]
    logger.info("Provider API keys initialized for: %s", ", ".join(providers_set))
    return {"status": "ok", "providers_initialized": providers_set}


@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    """Create a text completion with TEE signing"""
    try:
        request_dict = request.dict()
        request_bytes = json.dumps(request_dict, sort_keys=True).encode('utf-8')

        model = get_chat_model_cached(
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens or 4096,
        )

        messages = [HumanMessage(content=request.prompt)]
        response = await asyncio.to_thread(model.invoke, messages)

        response_content = response.content or ""
        timestamp = int(datetime.now(timezone.utc).timestamp())
        usage = extract_usage(response)

        msg_hash, input_hash_hex, output_hash_hex = compute_tee_msg_hash(
            request_bytes, response_content, timestamp
        )
        signature = tee_keys.sign_data(msg_hash)

        return CompletionResponse(
            completion=response_content,
            model=request.model,
            timestamp=timestamp,
            signature=signature,
            request_hash=input_hash_hex,
            output_hash=output_hash_hex,
            usage=usage,
            metadata=None
        )

    except Exception as e:
        logger.error(f"Completion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def create_chat_completion(request: ChatRequest):
    """Create a chat completion with TEE signing"""
    try:
        logger.info(f"=" * 80)
        logger.info(f"Chat request for model: {request.model}")
        logger.info(f"Number of messages: {len(request.messages)}")
        
        if request.tools:
            logger.info(f"Tools provided: {len(request.tools)}")
            for tool in request.tools:
                logger.info(f"  Tool: {tool.function.get('name', 'unnamed')}")
        
        for i, msg in enumerate(request.messages):
            logger.info(f"Message {i}: role={msg.role}, has_tool_calls={bool(msg.tool_calls)}")
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    logger.info(f"  Tool call in message: {tc}")
        
        request_dict = request.dict()
        request_bytes = json.dumps(request_dict, sort_keys=True).encode('utf-8')

        model = get_chat_model_cached(
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens or 4096,
        )

        if request.tools:
            logger.info(f"Binding {len(request.tools)} tools")
            tools_list = [{"type": "function", "function": tool.function}
                         for tool in request.tools]
            model = model.bind_tools(tools_list)

        messages = convert_messages(request.messages)
        response = await asyncio.to_thread(model.invoke, messages)

        # Gemini can return content as a list of parts, e.g. [{"type": "text", "text": "..."}].
        # Normalize to a plain string so downstream hashing and JSON serialization are consistent
        # across all providers. This mirrors the same normalization in the streaming path.
        if isinstance(response.content, list):
            content_str = ''.join(
                item.get('text', '') if isinstance(item, dict) else str(item)
                for item in response.content
            )
        else:
            content_str = response.content or ""

        message_dict = {
            "role": "assistant",
            "content": content_str,
        }

        finish_reason = "stop"
        if hasattr(response, "tool_calls") and response.tool_calls:
            finish_reason = "tool_calls"
            message_dict["tool_calls"] = [
                {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": json.dumps(tc.get("args", {})),
                    }
                }
                for tc in response.tool_calls
            ]

        usage = extract_usage(response)

        # Build the canonical output string to hash.
        # When finish_reason == "tool_calls", message_dict["content"] is "" — the
        # meaningful output lives in the tool_calls list. We serialize it so the
        # signature actually covers which tools were called and with what arguments.
        # For ordinary text turns we hash the content string directly.
        if finish_reason == "tool_calls" and message_dict.get("tool_calls"):
            response_content = json.dumps(message_dict["tool_calls"], sort_keys=True)
        else:
            response_content = message_dict["content"]
        timestamp = int(datetime.now(timezone.utc).timestamp())
        msg_hash, input_hash_hex, output_hash_hex = compute_tee_msg_hash(
            request_bytes, response_content, timestamp
        )
        signature = tee_keys.sign_data(msg_hash)

        return ChatResponse(
            finish_reason=finish_reason,
            message=message_dict,
            model=request.model,
            timestamp=timestamp,
            signature=signature,
            request_hash=input_hash_hex,
            output_hash=output_hash_hex,
            usage=usage,
            metadata=None
        )
        
    except Exception as e:
        logger.error(f"Chat completion error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions/stream")
async def create_chat_completion_stream(request: ChatRequest):
    """Create a streaming chat completion with tool calls and final token usage"""
    try:
        provider = get_provider_from_model(request.model)
        buffer_tool_calls = provider in ["openai", "anthropic"]

        # Capture request bytes before streaming so we can sign at the end
        request_bytes = json.dumps(request.dict(), sort_keys=True).encode('utf-8')

        model = get_chat_model_cached(
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens or 4096,
        )

        if request.tools:
            tools_list = [{"type": "function", "function": tool.function}
                         for tool in request.tools]
            model = model.bind_tools(tools_list)

        messages = convert_messages(request.messages)

        async def generate():
            try:
                full_content = ""
                final_usage = None
                buffered_tool_calls = {}
                finish_reason = "stop"

                async for chunk in model.astream(messages):
                    if chunk.content:
                        if isinstance(chunk.content, str):
                            content_str = chunk.content
                        elif isinstance(chunk.content, list):
                            content_str = ''.join(
                                item.get('text', '') if isinstance(item, dict) else str(item)
                                for item in chunk.content
                            )
                        else:
                            content_str = str(chunk.content)

                        if content_str:
                            full_content += content_str
                            data = {
                                "choices": [{
                                    "delta": {"content": content_str, "role": "assistant"},
                                    "index": 0,
                                    "finish_reason": None
                                }],
                                "model": request.model
                            }
                            yield f"data: {json.dumps(data)}\n\n"
                            await asyncio.sleep(0)

                    if hasattr(chunk, 'tool_call_chunks') and chunk.tool_call_chunks:
                        finish_reason = "tool_calls"
                        for tc_chunk in chunk.tool_call_chunks:
                            tc_index = tc_chunk.get('index', 0)

                            if tc_index not in buffered_tool_calls:
                                buffered_tool_calls[tc_index] = {
                                    'id': tc_chunk.get('id', ''),
                                    'type': 'function',
                                    'function': {'name': tc_chunk.get('name', ''), 'arguments': ''}
                                }

                            if 'id' in tc_chunk and tc_chunk['id']:
                                buffered_tool_calls[tc_index]['id'] = tc_chunk['id']
                            if 'name' in tc_chunk and tc_chunk['name']:
                                buffered_tool_calls[tc_index]['function']['name'] = tc_chunk['name']
                            if 'args' in tc_chunk and tc_chunk['args']:
                                args_value = tc_chunk['args']
                                if isinstance(args_value, dict):
                                    args_str = json.dumps(args_value)
                                elif isinstance(args_value, str):
                                    args_str = args_value
                                else:
                                    args_str = str(args_value)
                                buffered_tool_calls[tc_index]['function']['arguments'] += args_str

                            if not buffer_tool_calls:
                                delta = {"role": "assistant", "tool_calls": [{"index": tc_index, "type": "function", "function": {}}]}
                                if tc_chunk.get('id'):
                                    delta["tool_calls"][0]["id"] = tc_chunk['id']
                                if tc_chunk.get('name'):
                                    delta["tool_calls"][0]["function"]["name"] = tc_chunk['name']
                                if tc_chunk.get('args'):
                                    args_value = tc_chunk['args']
                                    if isinstance(args_value, dict):
                                        delta["tool_calls"][0]["function"]["arguments"] = json.dumps(args_value)
                                    elif isinstance(args_value, str):
                                        delta["tool_calls"][0]["function"]["arguments"] = args_value
                                    else:
                                        delta["tool_calls"][0]["function"]["arguments"] = str(args_value)

                                    if not delta["tool_calls"][0]["function"]:
                                        del delta["tool_calls"][0]["function"]

                                data = {
                                    "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
                                    "model": request.model
                                }
                                yield f"data: {json.dumps(data)}\n\n"
                                await asyncio.sleep(0)

                    if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                        final_usage = chunk.usage_metadata

                if buffer_tool_calls and buffered_tool_calls:
                    for tc_index, tc in buffered_tool_calls.items():
                        delta = {
                            "role": "assistant",
                            "tool_calls": [{
                                "index": tc_index,
                                "id": tc['id'],
                                "type": "function",
                                "function": {"name": tc['function']['name'], "arguments": tc['function']['arguments']}
                            }]
                        }
                        data = {"choices": [{"delta": delta, "index": 0, "finish_reason": None}], "model": request.model}
                        yield f"data: {json.dumps(data)}\n\n"
                        await asyncio.sleep(0)

                # Sign the completed response for TEE attestation.
                # The middleware extracts tee_signature from the *last* SSE event
                # (the final chunk before [DONE]), so we embed it there.
                #
                # Matches the on-chain verifier pattern:
                #   keccak256(abi.encodePacked(inputHash, outputHash, timestamp))
                # where timestamp is uint256 unix seconds.
                timestamp = int(datetime.now(timezone.utc).timestamp())
                # For tool-call responses full_content is "" — hash the fully-buffered
                # tool calls list instead so the signature covers the actual tool
                # invocations. Sort by index key for a deterministic canonical form.
                if finish_reason == "tool_calls" and buffered_tool_calls:
                    tool_calls_list = [buffered_tool_calls[k] for k in sorted(buffered_tool_calls.keys())]
                    output_content = json.dumps(tool_calls_list, sort_keys=True)
                else:
                    output_content = full_content
                msg_hash, input_hash_hex, output_hash_hex = compute_tee_msg_hash(
                    request_bytes, output_content, timestamp
                )
                tee_signature = tee_keys.sign_data(msg_hash)

                final_data = {
                    "choices": [{"delta": {}, "index": 0, "finish_reason": finish_reason}],
                    "model": request.model,
                    "tee_signature": tee_signature,
                    "tee_timestamp": timestamp,
                    "tee_request_hash": input_hash_hex,
                    "tee_output_hash": output_hash_hex,
                }
                if final_usage:
                    final_data["usage"] = {
                        "prompt_tokens": final_usage.get("input_tokens", 0),
                        "completion_tokens": final_usage.get("output_tokens", 0),
                        "total_tokens": final_usage.get("total_tokens", 0)
                    }
                    logger.info(f"Stream completed - Usage: {final_data['usage']}, Finish: {finish_reason}, inputHash: {input_hash_hex[:16]}..., outputHash: {output_hash_hex[:16]}...")

                yield f"data: {json.dumps(final_data)}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.error(f"Streaming error: {str(e)}", exc_info=True)
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    except Exception as e:
        logger.error(f"Stream setup error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    return {
        "data": [
            # OpenAI
            {"id": "openai/gpt-4.1-2025-04-14",    "provider": "openai"},
            {"id": "openai/o4-mini",                "provider": "openai"},
            {"id": "openai/gpt-5",                  "provider": "openai"},
            {"id": "openai/gpt-5-mini",             "provider": "openai"},
            # Anthropic
            {"id": "anthropic/claude-sonnet-4-5",   "provider": "anthropic"},
            {"id": "anthropic/claude-sonnet-4-6",   "provider": "anthropic"},
            {"id": "anthropic/claude-haiku-4-5",    "provider": "anthropic"},
            {"id": "anthropic/claude-opus-4-5",     "provider": "anthropic"},
            {"id": "anthropic/claude-opus-4-6",     "provider": "anthropic"},
            # Google
            {"id": "google/gemini-2.5-flash",       "provider": "google"},
            {"id": "google/gemini-2.5-pro",         "provider": "google"},
            {"id": "google/gemini-2.5-flash-lite",  "provider": "google"},
            {"id": "google/gemini-3-pro-preview",   "provider": "google"},
            {"id": "google/gemini-3-flash-preview", "provider": "google"},
            # xAI
            {"id": "x-ai/grok-4",                           "provider": "x-ai"},
            {"id": "x-ai/grok-4-fast",                      "provider": "x-ai"},
            {"id": "x-ai/grok-4-1-fast",                    "provider": "x-ai"},
            {"id": "x-ai/grok-4-1-fast-non-reasoning",      "provider": "x-ai"},
        ]
    }


@app.get("/")
async def root():
    return {
        "name": "TEE LLM Backend",
        "version": "1.0.0",
        "tee_enabled": TEE_ENABLED,
    }


if __name__ == "__main__":
    logger.info(f"Starting TEE LLM Router Server (TEE_ENABLED={TEE_ENABLED})...")
    
    # Only signal readiness when TEE is enabled
    signal_ready()
    
    port = int(os.getenv("LLM_SERVER_PORT", "8000"))
    host = os.getenv("LLM_SERVER_HOST", "127.0.0.1")
    
    logger.info(f"Server starting on {host}:{port}")
    logger.info(f"Public Key (first 100 chars): {tee_keys.get_public_key()[:100]}...")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        limit_concurrency=100,
        timeout_keep_alive=30,
        timeout_graceful_shutdown=5,
    )
