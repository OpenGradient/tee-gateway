prog := tee-llm-router
version := 1.0.0
image_tag := $(prog):$(version)
image_tar := $(prog)-$(version)-kaniko.tar
image_eif := $(image_tar:%.tar=%.eif)

SOURCE_DATE_EPOCH ?= 1700006400

ARCH ?= $(shell uname -m)
ifeq ($(ARCH),aarch64)
	override ARCH=arm64
endif
ifeq ($(ARCH),x86_64)
	override ARCH=amd64
endif

.PHONY: all
all: run

.PHONY: image
image: $(image_tar)

$(image_tar): Dockerfile scripts/start.sh requirements.txt
	find tee_gateway -exec touch -t 202311150000 {} \;
	touch -t 202311150000 scripts/start.sh Dockerfile requirements.txt
	SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) docker run \
		-v $(PWD):/workspace \
		-e SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) \
		gcr.io/kaniko-project/executor:v1.9.2 \
		--reproducible \
		--no-push \
		--tarPath $(image_tar) \
		--destination $(image_tag) \
		--build-arg TARGETPLATFORM=linux/$(ARCH) \
		--build-arg TARGETOS=linux \
		--build-arg TARGETARCH=$(ARCH) \
		--custom-platform linux/$(ARCH)

$(image_eif): $(image_tar)
	docker load -i $<
	nitro-cli build-enclave \
		--docker-uri $(image_tag) \
		--output-file $(image_eif)

.PHONY: run
run: $(image_eif)
	nitro-cli terminate-enclave --all
	./scripts/run-enclave.sh $(image_eif)

.PHONY: clean
clean:
	rm -f $(image_tar) $(image_eif)

# ---------------------------------------------------------------------------
# Enclave health / attestation checks (no payment required)
# ---------------------------------------------------------------------------

.PHONY: health
health:
	curl -i -k https://localhost:443/health

.PHONY: get-signing-key
get-signing-key:
	@curl -sk https://localhost:443/signing-key | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['public_key']); print('tee_id:', d.get('tee_id','MISSING'))"

.PHONY: verify-tee-id
verify-tee-id:
	@curl -sk https://localhost:443/signing-key | python3 -c "import json,sys,base64; from eth_hash.auto import keccak; d=json.load(sys.stdin); pem=d['public_key']; der=base64.b64decode(''.join(pem.strip().splitlines()[1:-1])); computed='0x'+keccak(der).hex(); reported=d.get('tee_id','MISSING'); print('reported tee_id:',reported); print('computed tee_id:',computed); print('VALID' if reported==computed else 'MISMATCH')"

.PHONY: get-tls-cert
get-tls-cert:
	openssl s_client -connect localhost:443 </dev/null 2>/dev/null | openssl x509 -text

# ---------------------------------------------------------------------------
# Local development (no TEE, no payment middleware enforced by x402 clients)
# ---------------------------------------------------------------------------

.PHONY: test-local
test-local:
	# Run server locally without TEE (for development).
	# Set API keys via environment variables before running:
	#   export OPENAI_API_KEY=...  ANTHROPIC_API_KEY=...  etc.
	python3 -m tee_gateway

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make run            - Build image, build EIF, and run in Nitro Enclave"
	@echo "  make image          - Build reproducible Docker image TAR (Kaniko)"
	@echo "  make clean          - Remove build artifacts (.tar, .eif)"
	@echo ""
	@echo "  make health         - Check enclave health endpoint"
	@echo "  make get-signing-key - Print TEE public key and tee_id"
	@echo "  make verify-tee-id  - Verify tee_id matches the public key"
	@echo "  make get-tls-cert   - Print the nitriding TLS certificate"
	@echo ""
	@echo "  make test-local     - Run server locally without TEE (development)"
	@echo ""
	@echo "  LLM endpoints (/v1/chat/completions, /v1/completions) require x402"
	@echo "  payment headers. Use an x402-compatible client to call them."
