"""Pydantic models from TEE routing API to be used for facilitator."""
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

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
    timestamp: str
    signature: str  # TEE signature
    request_hash: str  # Hash of request
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, str]] = None


class ChatResponse(BaseModel):
    finish_reason: str
    message: Dict[str, Any]
    model: str
    timestamp: str
    signature: str  # TEE signature
    request_hash: str  # Hash of request
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, str]] = None


class AttestationResponse(BaseModel):
    """TEE attestation document"""
    public_key: str
    timestamp: str
    enclave_info: Dict[str, Any]
    measurements: Optional[Dict] = None