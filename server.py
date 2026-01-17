#!/usr/bin/env python3
"""
TEE LLM Router Server
Runs within a Nitro Enclave with remote attestation and request signing.
Based on LangChain routing with async support and cryptographic verification.
"""

import os
import logging
import json
import hashlib
import base64
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime, UTC
import urllib.request

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
import psutil
import gc
import time
import httpx

from functools import lru_cache
import hashlib

from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_xai import ChatXAI
from langchain.chat_models import init_chat_model

# LLM Client request timeout
TIMEOUT = 120

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
    def __init__(self):
        self.private_key = None
        self.public_key = None
        self.public_key_pem = None
        self._initialize_keys()
    
    def _initialize_keys(self):
        """Generate RSA key pair for signing inference results"""
        logger.info("Generating TEE RSA key pair...")
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
        
        # Serialize public key for attestation
        self.public_key_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')
        
        logger.info("TEE key pair generated successfully")

        # Register with nitriding
        self.register_with_nitriding()

    def register_with_nitriding(self):
        """Register public key hash with nitriding"""
        try:
            # Get public key in DER format
            public_key_der = self.public_key.public_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            # Compute SHA256 hash of the public key
            key_hash = hashlib.sha256(public_key_der).digest()
            
            # Base64 encode the hash (this is what nitriding expects!)
            key_hash_b64 = base64.b64encode(key_hash).decode('utf-8')
            
            logger.info(f"Public key DER length: {len(public_key_der)} bytes")
            logger.info(f"Public key SHA256 hash (hex): {key_hash.hex()}")
            logger.info(f"Public key SHA256 hash (base64): {key_hash_b64}")
            
            # POST the BASE64-ENCODED HASH to nitriding
            nitriding_hash_url = "http://127.0.0.1:8080/enclave/hash"
            
            req = urllib.request.Request(
                nitriding_hash_url,
                data=key_hash_b64.encode('utf-8'),  # Send base64-encoded hash as UTF-8 string
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

    def sign_data(self, data: str) -> str:
        """Sign data with private key and return base64 signature"""
        data_bytes = data.encode('utf-8')
        signature = self.private_key.sign(
            data_bytes,
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

# Initialize key manager
tee_keys = TEEKeyManager()

# Nitriding readiness signal
nitriding_url = "http://127.0.0.1:8080/enclave/ready"

def signal_ready():
    """Signal to nitriding that enclave is ready"""
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
    usage: Optional[Dict] = None
    timestamp: Optional[str] = None
    signature: Optional[str] = None  # TEE signature
    request_hash: Optional[str] = None  # Hash of request


class ChatResponse(BaseModel):
    finish_reason: str
    message: Dict[str, Any]
    model: str
    usage: Optional[Dict] = None
    timestamp: Optional[str] = None
    signature: Optional[str] = None  # TEE signature
    request_hash: Optional[str] = None  # Hash of request


class AttestationResponse(BaseModel):
    """TEE attestation document"""
    public_key: str
    timestamp: str
    enclave_info: Dict[str, Any]
    measurements: Optional[Dict] = None


# Helper functions
def compute_request_hash(request_data: Dict) -> str:
    """Compute SHA256 hash of request data"""
    request_json = json.dumps(request_data, sort_keys=True)
    return hashlib.sha256(request_json.encode('utf-8')).hexdigest()


def get_api_key(model: str, auth_header: Optional[str]) -> Optional[str]:
    """Extract API key from authorization header or environment variables"""
    if auth_header and auth_header.startswith("Bearer "):
        logger.info(f"Using API key from Authorization header for {model}")
        return auth_header.replace("Bearer ", "")
    
    provider = get_provider_from_model(model)
    env_var_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "x-ai": "XAI_API_KEY",
        "cohere": "COHERE_API_KEY",
        "groq": "GROQ_API_KEY",
        "together": "TOGETHER_API_KEY",
    }
    
    env_var = env_var_map.get(provider)
    if env_var:
        api_key = os.getenv(env_var)
        if api_key:
            logger.info(f"Using {env_var} from environment for {model}")
            return api_key
    
    return None


@lru_cache(maxsize=32)
def _get_chat_model_from_env(model: str, temperature: float, max_tokens: int):
    """
    Internal cached function for environment-based API keys.
    Only caches models using environment variables.
    """
    provider = get_provider_from_model(model)
    env_var_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "x-ai": "XAI_API_KEY",
    }
    
    env_var = env_var_map.get(provider)
    if not env_var:
        raise ValueError(f"Unknown provider: {provider}")
    
    api_key = os.getenv(env_var)
    if not api_key:
        raise ValueError(f"Missing {env_var} in environment")
    
    return get_chat_model(model, api_key, temperature, max_tokens)


def get_chat_model_cached(model: str, temperature: float, max_tokens: int, api_key: Optional[str] = None):
    """
    Get chat model instance, using cache for environment keys.
    
    - If api_key is None: Use cached model with environment variable
    - If api_key is provided: Create fresh model instance (no caching)
    """
    if api_key:
        # User-provided key - create fresh instance, don't cache
        logger.info(f"Creating fresh model instance with user-provided API key for {model}")
        return get_chat_model(model, api_key, temperature, max_tokens)
    else:
        # Environment key - use cached instance
        return _get_chat_model_from_env(model, temperature, max_tokens)


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
    elif "command" in model_lower or model.startswith("cohere/"):
        return "cohere"
    elif model.startswith("groq/"):
        return "groq"
    elif model.startswith("together/"):
        return "together"
    else:
        return "openai"


def get_chat_model(model: str, api_key: str, temperature: float = 0.7, max_tokens: int = 100):
    """Get the appropriate chat model instance based on the model name"""
    provider = get_provider_from_model(model)
    
    logger.info(f"Creating chat model - Provider: {provider}, Model: {model}")
    
    # Common HTTP client configuration to limit connections
    http_client_config = {
        "max_connections": 10,
        "max_keepalive_connections": 5,
        "timeout": TIMEOUT,
    }
    
    if provider in ["google", "gemini"]:
        thinking_budget = None
        if "2.5-flash" in model or "flash-lite" in model:
            thinking_budget = 0
        elif "2.5-pro" in model:
            thinking_budget = 128
            
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
            thinking_budget=thinking_budget,
            include_thoughts=False if thinking_budget is not None else None,
            http_client_config=http_client_config,
        )
    elif provider == "openai":
        model_temp = 1.0 if model in ["o4-mini", "o3"] else temperature
        
        http_client = httpx.Client(
            limits=httpx.Limits(
                max_connections=http_client_config["max_connections"],
                max_keepalive_connections=http_client_config["max_keepalive_connections"]
            ),
            timeout=httpx.Timeout(http_client_config["timeout"])
        )

        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=model_temp,
            max_tokens=max_tokens,
            http_client=http_client,
        )
    elif provider == "anthropic":
        anthropic_model = model
        if model == "claude-3.7-sonnet":
            anthropic_model = "claude-3-7-sonnet-latest"
        elif model == "claude-3.5-haiku":
            anthropic_model = "claude-3-5-haiku-latest"
        elif model == "claude-4.0-sonnet":
            anthropic_model = "claude-sonnet-4-0"
            
        return ChatAnthropic(
            model=anthropic_model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=TIMEOUT,
        )
    elif provider == "x-ai":
        xai_model = model
        if model == "grok-3-mini-beta":
            xai_model = "grok-3-mini"
        elif model == "grok-3-beta":
            xai_model = "grok-3-latest"
        elif model == "grok-2-1212":
            xai_model = "grok-2-latest"
        elif model == "grok-4.1-fast":
            xai_model = "grok-4-1-fast"
            
        return ChatXAI(
            model=xai_model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            stream_usage=True,
            timeout=TIMEOUT,
        )
    else:
        logger.warning(f"Using fallback initialization for model: {model}")
        return init_chat_model(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )


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
                
                # Convert tool_calls from dict format to LangChain's expected format
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
                
                # Create AIMessage with tool_calls as direct attribute
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

def extract_usage(response: AIMessage) -> Optional[Dict]:
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
        "tee_enabled": True,
        "uptime_seconds": time.time() - process.create_time(),
        "memory_mb": process.memory_info().rss / 1024 / 1024,
        "threads": process.num_threads(),
        "open_files": len(process.open_files()),
        "num_fds": process.num_fds(),
        "connections": len(connections),
        "connection_states": conn_states,
        "gc_objects": len(gc.get_objects()),
    }


@app.get("/attestation")
async def get_attestation():
    """Return TEE attestation document with public key"""
    try:
        # In a real Nitro Enclave, you'd retrieve actual PCR measurements
        # For now, we'll return a mock attestation structure
        attestation = AttestationResponse(
            public_key=tee_keys.get_public_key(),
            timestamp=datetime.now(UTC).isoformat(),
            enclave_info={
                "platform": "aws-nitro",
                "instance_type": "tee-enabled",
                "version": "1.0.0"
            },
            measurements=None  # Would contain PCR values in real deployment
        )
        
        logger.info("Attestation document requested")
        return attestation
    except Exception as e:
        logger.error(f"Attestation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(
    request: CompletionRequest,
    authorization: Optional[str] = Header(None)
):
    """Create a text completion with TEE signing"""
    try:
        # Compute request hash
        request_dict = request.dict()
        request_hash = compute_request_hash(request_dict)
        
        api_key = get_api_key(request.model, authorization)
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail=f"No API key provided for {request.model}"
            )
        
        # Get model instance
        model = get_chat_model_cached(
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens or 4096,
            api_key=api_key,
        )
        
        # Convert prompt to message
        messages = [HumanMessage(content=request.prompt)]
        
        # Invoke model
        response = await asyncio.to_thread(model.invoke, messages)
        
        # Create response data for signing
        timestamp = datetime.now(UTC).isoformat()
        response_data = {
            "completion": response.content,
            "model": request.model,
            "request_hash": request_hash,
            "timestamp": timestamp
        }
        
        # Sign the response
        signature = tee_keys.sign_data(json.dumps(response_data, sort_keys=True))
        
        return CompletionResponse(
            completion=response.content,
            model=request.model,
            usage=extract_usage(response),
            timestamp=timestamp,
            signature=signature,
            request_hash=request_hash
        )
        
    except Exception as e:
        logger.error(f"Completion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def create_chat_completion(
    request: ChatRequest,
    authorization: Optional[str] = Header(None)
):
    """Create a chat completion with TEE signing"""
    try:
        logger.info(f"=" * 80)
        logger.info(f"Chat request for model: {request.model}")
        logger.info(f"Number of messages: {len(request.messages)}")
        
        # Log tool information
        if request.tools:
            logger.info(f"Tools provided: {len(request.tools)}")
            for tool in request.tools:
                logger.info(f"  Tool: {tool.function.get('name', 'unnamed')}")
        
        # Log messages with tool calls
        for i, msg in enumerate(request.messages):
            logger.info(f"Message {i}: role={msg.role}, has_tool_calls={bool(msg.tool_calls)}")
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    logger.info(f"  Tool call in message: {tc}")
        
        # Compute request hash
        request_dict = request.dict()
        request_hash = compute_request_hash(request_dict)
        
        api_key = get_api_key(request.model, authorization)
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail=f"No API key provided for {request.model}"
            )
        
        # Get model instance
        model = get_chat_model_cached(
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens or 4096,
            api_key=api_key,
        )
        
        # Bind tools if provided
        if request.tools:
            logger.info(f"Binding {len(request.tools)} tools")
            tools_list = [{"type": "function", "function": tool.function} 
                         for tool in request.tools]
            model = model.bind_tools(tools_list)
        
        # Convert messages
        messages = convert_messages(request.messages)
        
        # Invoke model asynchronously
        response = await asyncio.to_thread(model.invoke, messages)
        
        # Extract message content
        message_dict = {
            "role": "assistant",
            "content": response.content or "",
        }
        
        # Check for tool calls
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
        
        # Create response data for signing
        timestamp = datetime.now(UTC).isoformat()
        response_data = {
            "message": message_dict,
            "model": request.model,
            "finish_reason": finish_reason,
            "request_hash": request_hash,
            "timestamp": timestamp
        }
        
        # Sign the response
        signature = tee_keys.sign_data(json.dumps(response_data, sort_keys=True))
        
        return ChatResponse(
            finish_reason=finish_reason,
            message=message_dict,
            model=request.model,
            usage=extract_usage(response),
            timestamp=timestamp,
            signature=signature,
            request_hash=request_hash
        )
        
    except Exception as e:
        logger.error(f"Chat completion error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions/stream")
async def create_chat_completion_stream(
    request: ChatRequest,
    authorization: Optional[str] = Header(None)
):
    """Create a streaming chat completion with final token usage"""
    try:
        api_key = get_api_key(request.model, authorization)
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail=f"No API key provided for {request.model}"
            )
        
        # Get model instance
        model = get_chat_model_cached(
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens or 4096,
            api_key=api_key,
        )
        
        # Bind tools if provided
        if request.tools:
            tools_list = [{"type": "function", "function": tool.function} 
                         for tool in request.tools]
            model = model.bind_tools(tools_list)
        
        # Convert messages
        messages = convert_messages(request.messages)
        
        async def generate():
            """Generate SSE stream with token usage"""
            stream = None
            try:
                # Track accumulated content and usage
                full_content = ""
                final_usage = None
                
                stream = model.astream(messages)
                async for chunk in stream:
                    # Accumulate content
                    if chunk.content:
                        full_content += chunk.content
                    
                    # Stream content delta
                    data = {
                        "choices": [{
                            "delta": {
                                "content": chunk.content or "",
                                "role": "assistant"
                            },
                            "index": 0,     # This maintains compatability with OpenAI API format
                            "finish_reason": None
                        }],
                        "model": request.model
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    
                    # Capture usage metadata from the last chunk
                    if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                        final_usage = chunk.usage_metadata
                
                # Send final chunk with finish_reason and usage
                final_data = {
                    "choices": [{
                        "delta": {},
                        "index": 0,
                        "finish_reason": "stop"
                    }],
                    "model": request.model
                }
                
                # Add usage information if available
                if final_usage:
                    final_data["usage"] = {
                        "prompt_tokens": final_usage.get("input_tokens", 0),
                        "completion_tokens": final_usage.get("output_tokens", 0),
                        "total_tokens": final_usage.get("total_tokens", 0)
                    }
                    logger.info(f"Stream completed - Usage: {final_data['usage']}")
                
                yield f"data: {json.dumps(final_data)}\n\n"
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                logger.error(f"Streaming error: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                # Ensure stream is properly closed
                if stream is not None:
                    try:
                        await stream.aclose()
                    except:
                        pass
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        
    except Exception as e:
        logger.error(f"Stream setup error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "data": [
            {"id": "gpt-4o", "provider": "openai"},
            {"id": "gpt-4o-mini", "provider": "openai"},
            {"id": "gpt-4-turbo", "provider": "openai"},
            {"id": "o4-mini", "provider": "openai"},
            {"id": "o3", "provider": "openai"},
            {"id": "claude-3-5-sonnet-20240620", "provider": "anthropic"},
            {"id": "claude-3-opus-20240229", "provider": "anthropic"},
            {"id": "claude-3.7-sonnet", "provider": "anthropic"},
            {"id": "claude-4.0-sonnet", "provider": "anthropic"},
            {"id": "gemini-2.0-flash-exp", "provider": "google"},
            {"id": "gemini-2.5-flash-preview", "provider": "google"},
            {"id": "gemini-2.5-pro-preview", "provider": "google"},
            {"id": "gemini-1.5-pro", "provider": "google"},
            {"id": "gemini-2.5-flash-lite", "provider": "google"},
            {"id": "grok-3-mini-beta", "provider": "x-ai"},
            {"id": "grok-3-beta", "provider": "x-ai"},
            {"id": "grok-2-1212", "provider": "x-ai"},
            {"id": "grok-4.1-fast", "provider": "x-ai"},
            {"id": "grok-4-1-fast-non-reasoning", "provider": "x-ai"},
        ]
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "TEE LLM Router",
        "version": "1.0.0",
        "tee_enabled": True,
        "endpoints": {
            "attestation": "/attestation",
            "completion": "/v1/completions",
            "chat": "/v1/chat/completions",
            "chat_stream": "/v1/chat/completions/stream",
            "models": "/v1/models",
            "health": "/health"
        },
        "features": [
            "Remote Attestation",
            "Cryptographic Request Signing",
            "Multi-provider LLM Routing",
            "Async Processing"
        ]
    }


if __name__ == "__main__":
    logger.info("Starting TEE LLM Router Server...")
    
    # Signal readiness to nitriding
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
    )
