HOST ?= https://127.0.0.1:443
MODEL ?= gemini-2.5-flash-lite
PROMPT ?= "Describe to me the 7 layers of the network stack"
TEMPERATURE ?= 0.7
MAX_TOKENS ?= 150

prog := tee-llm-router
version := 1.0.0
image_tag := $(prog):$(version)
image_tar := $(prog)-$(version)-kaniko.tar
image_eif := $(image_tar:%.tar=%.eif)

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

$(image_tar): Dockerfile server.py start.sh requirements.txt
	docker run \
		-v $(PWD):/workspace \
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
	./run-enclave.sh $(image_eif)

.PHONY: clean
clean:
	rm -f $(image_tar) $(image_eif)

.PHONY: test-local
test-local:
	# Test locally without TEE (for development)
	python3 server.py

test-completion:
	curl -i -k -X POST $(HOST)/v1/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "$(MODEL)", "prompt": $(PROMPT), "temperature": $(TEMPERATURE)}'

test-chat:
	curl -i -k -X POST $(HOST)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "$(MODEL)", "messages": [{"role": "user", "content": $(PROMPT)}], "temperature": $(TEMPERATURE)}'

test-stream:
	curl -i -X POST ${HOST}/v1/chat/completions/stream \
 		-H "Content-Type: application/json" \
		-N \
		--insecure \
		-d '{"model": "${MODEL}","messages": [{"role": "user","content": ${PROMPT}}],"temperature": ${TEMPERATURE},"max_tokens": ${MAX_TOKENS}}'	
