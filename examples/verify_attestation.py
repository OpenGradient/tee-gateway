import base64
import cbor2
import cose
import json
import logging
from collections import namedtuple
import subprocess
import sys
import hashlib

from cose import EC2, CoseAlgorithms, CoseEllipticCurves
from Crypto.Util.number import long_to_bytes
from OpenSSL import crypto

measurement_path = "measurements.txt"
root_cert_path = "aws_root_cert.pem"
enclave_url = "https://3.133.152.176/enclave/attestation"
nonce = "0123456789abcdef0123456789abcdef01234567"

logging.basicConfig(
    filename='verification_logs.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

PCR_tuple = namedtuple("PCRs", ["PCR0", "PCR1", "PCR2"])

# This library is based on richardfan1126: nitro-enclave-python-demo 
# (https://github.com/richardfan1126/nitro-enclave-python-demo/tree/master)
def get_pcrs() -> PCR_tuple:
    """
    Gets expected PCR values from enclave measurements JSON, returns a tuple of the PCR values.
    """
    with open(measurement_path, 'r') as file:
        json_measurement_data = file.read()

    try:
        measurement_data = json.loads(json_measurement_data)
        PCRs = PCR_tuple(measurement_data["Measurements"]["PCR0"],
                         measurement_data["Measurements"]["PCR1"],
                         measurement_data["Measurements"]["PCR2"])
        logging.debug("Given PCR measurements:\n"
                      "PCR0 %s\n"
                      "PCR1 %s\n"
                      "PCR2 %s\n",
                      PCRs.PCR0,
                      PCRs.PCR1,
                      PCRs.PCR2)
    except json.JSONDecodeError as e:
        raise ValueError("Error reading measurement file for PCRs: %s" % e)
    
    return PCRs

def get_root_cert_pem() -> str:
    with open(root_cert_path, 'r') as file:
        return file.read()

def get_attestation(url: str, nonce: str) -> str:
    # Construct curl command
    curl_command = [
        "curl",
        "-k",
        "-G",
        url,
        "--data-urlencode",
        f"nonce={nonce}"
    ]

    # Run the curl command and capture the output
    result = subprocess.run(curl_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Check if the command was successful
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None

    # Return the output of the curl command
    if result.stdout == None:
        print(f"Curl command result was None")

    return result.stdout

def verify_attestation_doc(attestation_string: str) -> None:
    """
    Verify the attestation document

    This uses the expected PCR values stored in measurements.txt,
    and the root_cert_pem provided by AWS Nitro Attestation PKI.

    If invalid, raise an exception
    """
    # Load in PCR and root cert pem
    logging.debug("Loading expected PCR values and root cert pem")
    expected_pcrs = get_pcrs()
    root_cert_pem = get_root_cert_pem()

    # Decode CBOR attestation document
    decoded_data = base64.b64decode(attestation_string)
    data = cbor2.loads(decoded_data)

    # Load and decode document payload
    doc = data[2]
    doc_obj = cbor2.loads(doc)
    logging.debug("Loaded an attestation document")

    # Expose timestamp
    timestamp_ms = doc_obj['timestamp']
    timestamp_s = timestamp_ms / 1000
    logging.info("Attestation document timestamp (ms): %d", timestamp_ms)
    logging.info("Attestation document timestamp (UTC): %s", 
                __import__('datetime').datetime.utcfromtimestamp(timestamp_s).isoformat())

    # Get PCRs from attestation document
    document_pcrs_arr = doc_obj['pcrs']

    # Expose user data
    user_data = doc_obj['user_data']
    logging.debug("Enclave generated user data: %s", user_data)
    prefix_length = 2  # Taken from Nitriding documentation
    hash_length = hashlib.sha256().digest_size  # 32 bytes for a SHA256 hash

    # Extract the tlsKeyHash
    tls_key_start = prefix_length
    tls_key_end = tls_key_start + hash_length
    tls_key_hash = user_data[tls_key_start:tls_key_end]

    # Extract the appKeyHash
    app_key_start = tls_key_end + prefix_length
    app_key_end = app_key_start + hash_length
    app_key_hash = user_data[app_key_start:app_key_end]

    logging.info("Enclave returned TLS key: %s", base64.b64encode(tls_key_hash).decode('utf-8'))
    logging.info("Enclave returned app key: %s", base64.b64encode(app_key_hash).decode('utf-8'))

    # Expose public key
    # TODO (kyle):  Write API to expose public key to inference node
    #               This will be needed for the sequencer to encrypt
    #               input data.
    public_key = doc_obj['public_key']
    logging.debug("Enclave generated public key: %s", public_key)

    ## Validating Attestation document ##
    # 1. Validate PCRs and Nonce
    logging.debug("Validating PCRs")
    for index, expected_pcr in enumerate(expected_pcrs):
        # Attestation document doesn't have specified PCR, raise exception
        if index not in document_pcrs_arr or document_pcrs_arr[index] is None:
            raise Exception("PCR%s not found" % index)

        # Get PCR hexcode
        doc_pcr = document_pcrs_arr[index].hex()
        logging.debug("PCR%s:\n"
                      "Attestation value: %s\n"
                      "Expected PCR: %s",
                      index,
                      doc_pcr,
                      expected_pcr)

        # Check if PCR match
        if expected_pcr != doc_pcr:
            logging.warn("PCRs do not match:\n"
                         "Attestation PCR%s: %s\n"
                         "Expected PCR%s: %s",
                         index, doc_pcr,
                         index, expected_pcr)
            raise Exception("PCR%s does not match" % index)

    logging.debug("Validating nonce")
    # Check that nonce matches
    attestation_nonce = doc_obj['nonce'].hex()
    logging.info("Received nonce is %s", attestation_nonce)
    logging.info("Given nonce is %s", nonce)
    if attestation_nonce != nonce:
        raise Exception(f"Attestation nonce: {attestation_nonce}, did not match given nonce: {nonce}")

    # 2. Validate Signature 
    logging.debug("Validating signature of attestation document")
    # Get signing certificate from attestation document
    logging.debug("Getting signing certificate from attestation document:\n %s", doc_obj['certificate'])
    cert = crypto.load_certificate(crypto.FILETYPE_ASN1, doc_obj['certificate'])

    # Get the key parameters from the cert public key
    logging.debug("Creating EC2 key from the signing certificates public key")
    cert_public_numbers = cert.get_pubkey().to_cryptography_key().public_numbers()
    x = cert_public_numbers.x
    y = cert_public_numbers.y
    curve = cert_public_numbers.curve

    x = long_to_bytes(x)
    y = long_to_bytes(y)

    # Create the EC2 key from public key parameters
    key = EC2(alg = CoseAlgorithms.ES384, x = x, y = y, crv = CoseEllipticCurves.P_384)

    # Get the protected header from attestation document
    phdr = cbor2.loads(data[0])

    # Construct the Sign1 message
    logging.debug("Constructing Sign1 message from the attestation document")
    msg = cose.Sign1Message(phdr = phdr, uhdr = data[1], payload = doc)
    msg.signature = data[3]

    # Verify the signature using the EC2 key
    logging.debug("Comparing EC2 key against the Sign1")
    if not msg.verify_signature(key):
        raise Exception("Wrong signature")
    logging.debug("Signature of attestation document verified")

    # 3. Validate signing certificate PKI
    logging.debug("Verifying the certificate of the attestation document "
                  "is signed by the root certificate of the AWS Nitro Attestation PKI")
    if root_cert_pem is not None:
        # Create an X509Store object for the CA bundles
        store = crypto.X509Store()

        # Create the CA cert object from PEM string, and store into X509Store
        _cert = crypto.load_certificate(crypto.FILETYPE_PEM, root_cert_pem)
        store.add_cert(_cert)

        # Get the CA bundle from attestation document and store into X509Store
        # Except the first certificate, which is the root certificate
        for _cert_binary in doc_obj['cabundle'][1:]:
            _cert = crypto.load_certificate(crypto.FILETYPE_ASN1, _cert_binary)
            store.add_cert(_cert)

        # Get the X509Store context
        store_ctx = crypto.X509StoreContext(store, cert)
        
        # Validate the certificate
        # If the cert is invalid, it will raise exception
        store_ctx.verify_certificate()
        logging.debug("Certificate verified by AWS Nitro Attestation PKI")

    print("Verification successful")
    return

if __name__ == "__main__":
    print("Starting verification")

    attestation_str = get_attestation(enclave_url, nonce)
    print("Attestation string:\n", attestation_str)

    verify_attestation_doc(attestation_str)