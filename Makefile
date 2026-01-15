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

# Test models for each provider
OPENAI_MODEL ?= gpt-4o
ANTHROPIC_MODEL ?= claude-3.7-sonnet
GOOGLE_MODEL ?= gemini-2.5-flash-lite
XAI_MODEL ?= grok-3-mini-beta

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
	docker load -i $
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
	curl -i -X POST $(HOST)/v1/chat/completions/stream \
		-H "Content-Type: application/json" \
		-N \
		--insecure \
		-d '{"model": "$(MODEL)","messages": [{"role": "user","content": $(PROMPT)}],"temperature": $(TEMPERATURE),"max_tokens": $(MAX_TOKENS)}'

# Test chat completions for all providers
.PHONY: test-chat-all-models
test-chat-all-models: test-chat-openai test-chat-anthropic test-chat-google test-chat-xai

.PHONY: test-chat-openai
test-chat-openai:
	@echo "\n=========================================="
	@echo "Testing OpenAI: $(OPENAI_MODEL)"
	@echo "==========================================\n"
	curl -i -k -X POST $(HOST)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "$(OPENAI_MODEL)", "messages": [{"role": "user", "content": $(PROMPT)}], "temperature": $(TEMPERATURE), "max_tokens": $(MAX_TOKENS)}'

.PHONY: test-chat-anthropic
test-chat-anthropic:
	@echo "\n=========================================="
	@echo "Testing Anthropic: $(ANTHROPIC_MODEL)"
	@echo "==========================================\n"
	curl -i -k -X POST $(HOST)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "$(ANTHROPIC_MODEL)", "messages": [{"role": "user", "content": $(PROMPT)}], "temperature": $(TEMPERATURE), "max_tokens": $(MAX_TOKENS)}'

.PHONY: test-chat-google
test-chat-google:
	@echo "\n=========================================="
	@echo "Testing Google: $(GOOGLE_MODEL)"
	@echo "==========================================\n"
	curl -i -k -X POST $(HOST)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "$(GOOGLE_MODEL)", "messages": [{"role": "user", "content": $(PROMPT)}], "temperature": $(TEMPERATURE), "max_tokens": $(MAX_TOKENS)}'

.PHONY: test-chat-xai
test-chat-xai:
	@echo "\n=========================================="
	@echo "Testing xAI: $(XAI_MODEL)"
	@echo "==========================================\n"
	curl -i -k -X POST $(HOST)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "$(XAI_MODEL)", "messages": [{"role": "user", "content": $(PROMPT)}], "temperature": $(TEMPERATURE), "max_tokens": $(MAX_TOKENS)}'

# Test streaming for all providers
.PHONY: test-stream-all-models
test-stream-all-models: test-stream-openai test-stream-anthropic test-stream-google test-stream-xai

.PHONY: test-stream-openai
test-stream-openai:
	@echo "\n=========================================="
	@echo "Testing OpenAI Stream: $(OPENAI_MODEL)"
	@echo "==========================================\n"
	curl -i -X POST $(HOST)/v1/chat/completions/stream \
		-H "Content-Type: application/json" \
		-N \
		--insecure \
		-d '{"model": "$(OPENAI_MODEL)","messages": [{"role": "user","content": $(PROMPT)}],"temperature": $(TEMPERATURE),"max_tokens": $(MAX_TOKENS)}'

.PHONY: test-stream-anthropic
test-stream-anthropic:
	@echo "\n=========================================="
	@echo "Testing Anthropic Stream: $(ANTHROPIC_MODEL)"
	@echo "==========================================\n"
	curl -i -X POST $(HOST)/v1/chat/completions/stream \
		-H "Content-Type: application/json" \
		-N \
		--insecure \
		-d '{"model": "$(ANTHROPIC_MODEL)","messages": [{"role": "user","content": $(PROMPT)}],"temperature": $(TEMPERATURE),"max_tokens": $(MAX_TOKENS)}'

.PHONY: test-stream-google
test-stream-google:
	@echo "\n=========================================="
	@echo "Testing Google Stream: $(GOOGLE_MODEL)"
	@echo "==========================================\n"
	curl -i -X POST $(HOST)/v1/chat/completions/stream \
		-H "Content-Type: application/json" \
		-N \
		--insecure \
		-d '{"model": "$(GOOGLE_MODEL)","messages": [{"role": "user","content": $(PROMPT)}],"temperature": $(TEMPERATURE),"max_tokens": $(MAX_TOKENS)}'

.PHONY: test-stream-xai
test-stream-xai:
	@echo "\n=========================================="
	@echo "Testing xAI Stream: $(XAI_MODEL)"
	@echo "==========================================\n"
	curl -i -X POST $(HOST)/v1/chat/completions/stream \
		-H "Content-Type: application/json" \
		-N \
		--insecure \
		-d '{"model": "$(XAI_MODEL)","messages": [{"role": "user","content": $(PROMPT)}],"temperature": $(TEMPERATURE),"max_tokens": $(MAX_TOKENS)}'

# Test completions for all providers (only OpenAI/xAI support this)
.PHONY: test-completion-all-models
test-completion-all-models: test-completion-openai test-completion-xai

.PHONY: test-completion-openai
test-completion-openai:
	@echo "\n=========================================="
	@echo "Testing OpenAI Completion: $(OPENAI_MODEL)"
	@echo "==========================================\n"
	curl -i -k -X POST $(HOST)/v1/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "$(OPENAI_MODEL)", "prompt": $(PROMPT), "temperature": $(TEMPERATURE), "max_tokens": $(MAX_TOKENS)}'

.PHONY: test-completion-xai
test-completion-xai:
	@echo "\n=========================================="
	@echo "Testing xAI Completion: $(XAI_MODEL)"
	@echo "==========================================\n"
	curl -i -k -X POST $(HOST)/v1/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "$(XAI_MODEL)", "prompt": $(PROMPT), "temperature": $(TEMPERATURE), "max_tokens": $(MAX_TOKENS)}'

# Comprehensive test suite
.PHONY: test-all
test-all: test-chat-all-models test-stream-all-models test-completion-all-models
	@echo "\n=========================================="
	@echo "All tests completed!"
	@echo "==========================================\n"

# Quick test with just one model per provider
.PHONY: test-quick
test-quick:
	@echo "Running quick test with one model per provider..."
	@$(MAKE) test-chat-openai
	@$(MAKE) test-chat-google
