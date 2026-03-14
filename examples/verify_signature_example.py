from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
import base64
import json
import hashlib
import ast

# ============================================
# PARSE YOUR RESULT
# ============================================

# Your public key
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

# Parse system_fingerprint from your result
system_fingerprint = "{'response_signature': 'PLyCgScL1Jr6OSb7wazEbor4yhBYJpauuqmsZJBoRNrpYl0sJ3ct472IminGRcfGGF1sBNB9YU6lKiWsJRnygJIufQ+yKt6a14QxrtjYp0F2LKCIvjIzveVnHs6oQQa9hz8VJFqSO/QLa4quw1GjYJHo+2fy8JPOPSBCXtmbHhBj4/7vSK53kQwJ0jld+LnpaAlURMxSaR49KOsbmAFCB9iR1pv292g0QOIY0hvlsNjH7HWaz+X1e2+Yytcl3eLP2IYIQqyVgJkm/U2zGb8ZZW10xxY+DbN+QlnHU9/SAq38n36zDbjLdZUhkgVTtht4vdn1wgFDSuEGQx4X5/nIHw==', 'request_signature': '3cd5e62557ea16dc77aef5c2c66188d180be259ac00f482de19896e78ebbf429', 'response_timestamp': '2025-12-16T08:20:26.491885+00:00'}"

signatures = ast.literal_eval(system_fingerprint)
response_sig_b64 = signatures['response_signature']
request_hash = signatures['request_signature']
timestamp_iso = signatures['response_timestamp']

response_signature = base64.b64decode(response_sig_b64)

# Response data from your result (WITH QUESTION MARK!)
message_content = "Hello! How can I assist you today? 😊"
model = "gpt-4.1-2025-04-14"
finish_reason = "stop"

print("=" * 70)
print("TEE SIGNATURE VERIFICATION")
print("=" * 70)

# ============================================
# STEP 1: VERIFY REQUEST HASH
# ============================================

print("\n[1] VERIFYING REQUEST HASH")
print("-" * 70)

# Build original request (matching what was sent to the server)
original_request = {
    "model": "gpt-4.1-2025-04-14",
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

# Compute request hash (matching server's compute_request_hash function)
request_json = json.dumps(original_request, sort_keys=True)
computed_request_hash = hashlib.sha256(request_json.encode('utf-8')).hexdigest()

print(f"Request hash from response: {request_hash}")
print(f"Computed from request:      {computed_request_hash}")

if request_hash == computed_request_hash:
    print("✓ REQUEST HASH VERIFIED!")
    print("  The request was not tampered with")
else:
    print("✗ REQUEST HASH MISMATCH!")
    print("  WARNING: Request may have been modified!")

# ============================================
# STEP 2: VERIFY RESPONSE SIGNATURE
# ============================================

print("\n[2] VERIFYING RESPONSE SIGNATURE")
print("-" * 70)

print(f"Timestamp from response: {timestamp_iso}")
print(f"Signature length: {len(response_signature)} bytes")

# Reconstruct the EXACT signed data structure (from server.py line 505-511)
signed_data = {
    "finish_reason": finish_reason,
    "message": {
        "role": "assistant",
        "content": message_content
    },
    "model": model,
    "request_hash": request_hash,
    "timestamp": timestamp_iso
}

print("\nSigned data structure:")
print(json.dumps(signed_data, indent=2, sort_keys=True))

# Create the signed message (with sort_keys=True to match server)
signed_message = json.dumps(signed_data, sort_keys=True).encode('utf-8')

print("\nSigned message (first 200 chars):")
print(signed_message[:200].decode('utf-8', errors='replace'))

# Verify the signature using RSA-PSS
try:
    public_key.verify(
        response_signature,
        signed_message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    
    print("\n" + "=" * 70)
    print("✓✓✓ RESPONSE SIGNATURE VERIFIED! ✓✓✓")
    print("=" * 70)
    print("\nThis response was:")
    print("  1. Generated inside the TEE enclave")
    print("  2. Signed with the attested private key")
    print("  3. Not tampered with in transit")
    print("\nYou can cryptographically trust this message:")
    print(f'  "{message_content}"')
    
    print("\n" + "=" * 70)
    print("✓✓✓ FULL TEE VERIFICATION SUCCESSFUL ✓✓✓")
    print("=" * 70)
    
except Exception as e:
    print("\n✗ SIGNATURE VERIFICATION FAILED!")
    print(f"Error: {e}")