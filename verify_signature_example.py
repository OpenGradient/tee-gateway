"""
TEE Signature Verification Example

Demonstrates how to verify signatures produced by the TEE LLM Router.
The signing format is compatible with TEERegistry.verifySignature on-chain:

  messageHash = keccak256(abi.encodePacked(inputHash, outputHash, timestamp))
  signature   = RSA-PSS-SHA256(messageHash)

Off-chain verification follows the same steps, using the public key
obtained from the /attestation endpoint.
"""

from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
from Crypto.Hash import keccak as keccak_mod
import base64
import json


# ============================================
# HELPERS (matching server.py and TEERegistry)
# ============================================

def keccak256(data: bytes) -> bytes:
    """Compute keccak256 hash (Ethereum-compatible, matching Solidity's keccak256)"""
    k = keccak_mod.new(digest_bits=256, data=data)
    return k.digest()


def compute_input_hash(request_data: dict) -> bytes:
    """Compute keccak256 of request data (bytes32), matching server.py"""
    request_json = json.dumps(request_data, sort_keys=True)
    return keccak256(request_json.encode('utf-8'))


def compute_output_hash(response_data: dict) -> bytes:
    """Compute keccak256 of response data (bytes32), matching server.py"""
    response_json = json.dumps(response_data, sort_keys=True)
    return keccak256(response_json.encode('utf-8'))


def compute_message_hash(input_hash: bytes, output_hash: bytes, timestamp: int) -> bytes:
    """Compute the message hash that was signed.

    Matches TEERegistry.computeMessageHash:
      keccak256(abi.encodePacked(inputHash, outputHash, timestamp))
    """
    # abi.encodePacked(bytes32, bytes32, uint256) = raw concatenation
    packed = input_hash + output_hash + timestamp.to_bytes(32, byteorder='big')
    return keccak256(packed)


# ============================================
# EXAMPLE: Parse a response and verify
# ============================================

# Your public key (from /attestation endpoint)
public_key_pem = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAlZfwKzVJcc2am/006eVR
07J7W6jpdFiuB3HMSxnB+to7bMR49RX2cVJpKar+1i0+qfzDYTo3zsJTKy4JcNuL
O9uiJZgZ6DsAGEROwKihPtGF0Z2dG1uSwdSHyPUDtUEGNfkXbWmORFhO1O4Gurvc
Lnf9LWRnVpiep26KyQlN+2JUVbpr17bEV/NiBe7u1klCs22BBYo3w0ZBT9FjhamO
1EtWS9Cz+tMqYryPRYL4cQahGLEYgLm1gWsChlxmsBixV4Iv+AqUvvbQf5vKuxlz
mjlJv08ZCKnCc0034NsdKbriJ0G3AlQHRk0TrPMkriGMJmAcDUyxmArh4b1/0aSZ
CwIDAQAB
-----END PUBLIC KEY-----"""

public_key = serialization.load_pem_public_key(
    public_key_pem.encode('utf-8'),
    backend=default_backend()
)

# Example response from the TEE LLM Router (replace with actual values)
response = {
    "finish_reason": "stop",
    "message": {"role": "assistant", "content": "Hello! How can I assist you today?"},
    "model": "gpt-4o",
    "timestamp": "2025-12-16T08:20:26.491885+00:00",
    "timestamp_unix": 1734337226,
    "input_hash": "0xabcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
    "output_hash": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
    "signature": "<base64-encoded-signature>",
    "tee_id": "0x9876543210fedcba9876543210fedcba9876543210fedcba9876543210fedcba",
}

# Example original request (what you sent to the server)
original_request = {
    "model": "gpt-4o",
    "messages": [
        {
            "role": "user",
            "content": "Hello!",
            "tool_calls": None,
            "tool_call_id": None,
            "name": None
        }
    ],
    "max_tokens": 100,
    "temperature": 0.9,
    "stop": None,
    "tools": None,
    "tool_choice": "auto"
}

print("=" * 70)
print("TEE SIGNATURE VERIFICATION (On-Chain Compatible)")
print("=" * 70)

# ============================================
# STEP 1: VERIFY INPUT HASH
# ============================================

print("\n[1] VERIFYING INPUT HASH")
print("-" * 70)

computed_input_hash = compute_input_hash(original_request)
response_input_hash = bytes.fromhex(response["input_hash"][2:])  # strip 0x prefix

print(f"Input hash from response: {response['input_hash']}")
print(f"Computed from request:    0x{computed_input_hash.hex()}")

if computed_input_hash == response_input_hash:
    print("OK: Input hash verified - request was not tampered with")
else:
    print("FAIL: Input hash mismatch - request may have been modified!")

# ============================================
# STEP 2: VERIFY OUTPUT HASH
# ============================================

print("\n[2] VERIFYING OUTPUT HASH")
print("-" * 70)

# Reconstruct the output data that was hashed (must match server.py)
output_data = {
    "finish_reason": response["finish_reason"],
    "message": response["message"],
    "model": response["model"],
}
computed_output_hash = compute_output_hash(output_data)
response_output_hash = bytes.fromhex(response["output_hash"][2:])

print(f"Output hash from response: {response['output_hash']}")
print(f"Computed from response:    0x{computed_output_hash.hex()}")

if computed_output_hash == response_output_hash:
    print("OK: Output hash verified - response content matches")
else:
    print("FAIL: Output hash mismatch - response may have been modified!")

# ============================================
# STEP 3: VERIFY RSA-PSS SIGNATURE
# ============================================

print("\n[3] VERIFYING RSA-PSS SIGNATURE (TEERegistry-compatible)")
print("-" * 70)

timestamp_unix = response["timestamp_unix"]
input_hash_bytes = bytes.fromhex(response["input_hash"][2:])
output_hash_bytes = bytes.fromhex(response["output_hash"][2:])

# Compute the message hash (same as TEERegistry.computeMessageHash)
message_hash = compute_message_hash(input_hash_bytes, output_hash_bytes, timestamp_unix)
print(f"messageHash = keccak256(abi.encodePacked(inputHash, outputHash, timestamp))")
print(f"           = 0x{message_hash.hex()}")

# The signature is RSA-PSS over SHA256(messageHash)
# Python's verify() with hashes.SHA256() will internally compute SHA256(message_hash)
signature_bytes = base64.b64decode(response["signature"])

try:
    public_key.verify(
        signature_bytes,
        message_hash,  # 32 bytes; library computes SHA256(message_hash) internally
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )

    print("\n" + "=" * 70)
    print("SIGNATURE VERIFIED SUCCESSFULLY")
    print("=" * 70)
    print("\nThis response was:")
    print("  1. Generated inside the TEE enclave")
    print("  2. Signed with the attested private key")
    print("  3. Not tampered with in transit")
    print("  4. Verifiable on-chain via TEERegistry.verifySignature()")
    print(f"\nTEE ID: {response['tee_id']}")
    print(f"Message: \"{response['message']['content']}\"")

    print("\n" + "=" * 70)
    print("For on-chain verification, call:")
    print(f"  TEERegistry.verifySignature(")
    print(f"    teeId:     {response['tee_id']},")
    print(f"    inputHash:  {response['input_hash']},")
    print(f"    outputHash: {response['output_hash']},")
    print(f"    timestamp:  {timestamp_unix},")
    print(f"    signature:  <raw signature bytes>")
    print(f"  )")
    print("=" * 70)

except Exception as e:
    print(f"\nSIGNATURE VERIFICATION FAILED!")
    print(f"Error: {e}")
