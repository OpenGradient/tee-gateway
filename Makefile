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

SOURCE_DATE_EPOCH ?= 1700006400

ARCH ?= $(shell uname -m)
ifeq ($(ARCH),aarch64)
	override ARCH=arm64
endif
ifeq ($(ARCH),x86_64)
	override ARCH=amd64
endif

# Test models for each provider
OPENAI_MODEL ?= gpt-4.1
ANTHROPIC_MODEL ?= claude-3.7-sonnet
GOOGLE_MODEL ?= gemini-2.5-flash-lite
XAI_MODEL ?= grok-3-mini-beta

.PHONY: all
all: run

.PHONY: image
image: $(image_tar)

$(image_tar): Dockerfile scripts/start.sh requirements.txt
	find openapi_server -exec touch -t 202311150000 {} \;
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

.PHONY: test-local
test-local:
	# Test locally without TEE (for development)
	python3 -m openapi_server

test-completion:
	curl -i -k -X POST $(HOST)/v1/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "$(MODEL)", "prompt": $(PROMPT), "temperature": $(TEMPERATURE)}'

test-chat:
	curl -i -k -X POST $(HOST)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "$(MODEL)", "messages": [{"role": "user", "content": $(PROMPT)}], "temperature": $(TEMPERATURE)}'

test-stream:
	curl -i -k -X POST $(HOST)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-N \
		-d '{"model": "$(MODEL)","messages": [{"role": "user","content": $(PROMPT)}],"temperature": $(TEMPERATURE),"max_tokens": $(MAX_TOKENS),"stream": true}'

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
	curl -i -k -X POST $(HOST)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-N \
		-d '{"model": "$(OPENAI_MODEL)","messages": [{"role": "user","content": $(PROMPT)}],"temperature": $(TEMPERATURE),"max_tokens": $(MAX_TOKENS),"stream": true}'

.PHONY: test-stream-anthropic
test-stream-anthropic:
	@echo "\n=========================================="
	@echo "Testing Anthropic Stream: $(ANTHROPIC_MODEL)"
	@echo "==========================================\n"
	curl -i -k -X POST $(HOST)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-N \
		-d '{"model": "$(ANTHROPIC_MODEL)","messages": [{"role": "user","content": $(PROMPT)}],"temperature": $(TEMPERATURE),"max_tokens": $(MAX_TOKENS),"stream": true}'

.PHONY: test-stream-google
test-stream-google:
	@echo "\n=========================================="
	@echo "Testing Google Stream: $(GOOGLE_MODEL)"
	@echo "==========================================\n"
	curl -i -k -X POST $(HOST)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-N \
		-d '{"model": "$(GOOGLE_MODEL)","messages": [{"role": "user","content": $(PROMPT)}],"temperature": $(TEMPERATURE),"max_tokens": $(MAX_TOKENS),"stream": true}'

.PHONY: test-stream-xai
test-stream-xai:
	@echo "\n=========================================="
	@echo "Testing xAI Stream: $(XAI_MODEL)"
	@echo "==========================================\n"
	curl -i -k -X POST $(HOST)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-N \
		-d '{"model": "$(XAI_MODEL)","messages": [{"role": "user","content": $(PROMPT)}],"temperature": $(TEMPERATURE),"max_tokens": $(MAX_TOKENS),"stream": true}'

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

# Tool calling tests for all providers
.PHONY: test-tools-all-models
test-tools-all-models: test-tools-openai test-tools-anthropic test-tools-google test-tools-xai

.PHONY: test-tools-openai
test-tools-openai:
	@echo "\n=========================================="
	@echo "Testing OpenAI Tools: $(OPENAI_MODEL)"
	@echo "==========================================\n"
	@echo '{"model":"$(OPENAI_MODEL)","messages":[{"role":"user","content":"What is the weather in San Francisco?"}],"temperature":$(TEMPERATURE),"max_tokens":$(MAX_TOKENS),"tools":[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a location","parameters":{"type":"object","properties":{"location":{"type":"string","description":"City name"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]}}}]}' | \
	curl -i -k -X POST $(HOST)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d @-

.PHONY: test-tools-anthropic
test-tools-anthropic:
	@echo "\n=========================================="
	@echo "Testing Anthropic Tools: $(ANTHROPIC_MODEL)"
	@echo "==========================================\n"
	@echo '{"model":"$(ANTHROPIC_MODEL)","messages":[{"role":"user","content":"Calculate 15 times 23 plus 47"}],"temperature":$(TEMPERATURE),"max_tokens":$(MAX_TOKENS),"tools":[{"type":"function","function":{"name":"calculator","description":"Perform basic arithmetic operations","parameters":{"type":"object","properties":{"operation":{"type":"string","enum":["add","subtract","multiply","divide"]},"a":{"type":"number"},"b":{"type":"number"}},"required":["operation","a","b"]}}}]}' | \
	curl -i -k -X POST $(HOST)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d @-

.PHONY: test-tools-google
test-tools-google:
	@echo "\n=========================================="
	@echo "Testing Google Tools: $(GOOGLE_MODEL)"
	@echo "==========================================\n"
	@echo '{"model":"$(GOOGLE_MODEL)","messages":[{"role":"user","content":"Search for recent AI news"}],"temperature":$(TEMPERATURE),"max_tokens":$(MAX_TOKENS),"tools":[{"type":"function","function":{"name":"web_search","description":"Search the web for information","parameters":{"type":"object","properties":{"query":{"type":"string","description":"Search query"},"num_results":{"type":"integer","description":"Number of results","default":5}},"required":["query"]}}}]}' | \
	curl -i -k -X POST $(HOST)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d @-

.PHONY: test-tools-xai
test-tools-xai:
	@echo "\n=========================================="
	@echo "Testing xAI Tools: $(XAI_MODEL)"
	@echo "==========================================\n"
	@echo '{"model":"$(XAI_MODEL)","messages":[{"role":"user","content":"Send an email to john@example.com about the meeting tomorrow at 2pm"}],"temperature":$(TEMPERATURE),"max_tokens":$(MAX_TOKENS),"tools":[{"type":"function","function":{"name":"send_email","description":"Send an email to a recipient","parameters":{"type":"object","properties":{"to":{"type":"string","description":"Recipient email"},"subject":{"type":"string","description":"Email subject"},"body":{"type":"string","description":"Email body"}},"required":["to","subject","body"]}}}]}' | \
	curl -i -k -X POST $(HOST)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d @-

# Multi-turn tool calling test (weather example)
.PHONY: test-tools-multiturn
test-tools-multiturn:
	@echo "\n=========================================="
	@echo "Testing Multi-turn Tool Calling: $(GOOGLE_MODEL)"
	@echo "==========================================\n"
	@echo "Step 1: Initial request with tool..."
	@echo '{"model":"$(GOOGLE_MODEL)","messages":[{"role":"user","content":"What is the weather in Tokyo?"}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a location","parameters":{"type":"object","properties":{"location":{"type":"string"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]}}}]}' | \
	curl -s -k -X POST $(HOST)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d @- > /tmp/tool_response.json
	@echo "\n"
	@cat /tmp/tool_response.json | python3 -m json.tool
	@echo "\n\nStep 2: Sending tool result back..."
	@python3 -c 'import json; resp = json.load(open("/tmp/tool_response.json")); tc = resp["choices"][0]["message"]["tool_calls"][0]; req = {"model": "$(GOOGLE_MODEL)", "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}, {"role": "assistant", "content": "", "tool_calls": [tc]}, {"role": "tool", "tool_call_id": tc["id"], "name": "get_weather", "content": "{\"temperature\": 18, \"condition\": \"cloudy\", \"humidity\": 75}"}], "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a location", "parameters": {"type": "object", "properties": {"location": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]}}}]}; print(json.dumps(req))' | \
	curl -s -k -X POST $(HOST)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d @- | python3 -m json.tool

# Database query tool test (complex parameters)
.PHONY: test-tools-complex
test-tools-complex:
	@echo "\n=========================================="
	@echo "Testing Complex Tool Parameters: $(ANTHROPIC_MODEL)"
	@echo "==========================================\n"
	@echo '{"model":"$(ANTHROPIC_MODEL)","messages":[{"role":"user","content":"Find all users who signed up in the last 7 days from California"}],"temperature":$(TEMPERATURE),"max_tokens":$(MAX_TOKENS),"tools":[{"type":"function","function":{"name":"query_database","description":"Query the user database with filters","parameters":{"type":"object","properties":{"table":{"type":"string","description":"Database table name"},"filters":{"type":"array","description":"Filter conditions","items":{"type":"object","properties":{"field":{"type":"string"},"operator":{"type":"string","enum":["equals","greater_than","less_than","contains"]},"value":{}},"required":["field","operator","value"]}},"limit":{"type":"integer","description":"Max results"}},"required":["table","filters"]}}}]}' | \
	curl -i -k -X POST $(HOST)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d @-

# Streaming tool calling tests for all providers
.PHONY: test-stream-tools-all-models
test-stream-tools-all-models: test-stream-tools-openai test-stream-tools-anthropic test-stream-tools-google test-stream-tools-xai

.PHONY: test-stream-tools-openai
test-stream-tools-openai:
	@echo "\n=========================================="
	@echo "Testing OpenAI Streaming Tools: $(OPENAI_MODEL)"
	@echo "==========================================\n"
	@echo '{"model":"$(OPENAI_MODEL)","messages":[{"role":"user","content":"What is the weather in San Francisco and New York?"}],"stream":true,"temperature":$(TEMPERATURE),"max_tokens":$(MAX_TOKENS),"tools":[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a location","parameters":{"type":"object","properties":{"location":{"type":"string","description":"City name"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]}}}]}' | \
	curl -N -k -X POST $(HOST)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d @-

.PHONY: test-stream-tools-anthropic
test-stream-tools-anthropic:
	@echo "\n=========================================="
	@echo "Testing Anthropic Streaming Tools: $(ANTHROPIC_MODEL)"
	@echo "==========================================\n"
	@echo '{"model":"$(ANTHROPIC_MODEL)","messages":[{"role":"user","content":"Calculate 15 times 23, then add 47 to the result"}],"stream":true,"temperature":$(TEMPERATURE),"max_tokens":$(MAX_TOKENS),"tools":[{"type":"function","function":{"name":"calculator","description":"Perform basic arithmetic operations","parameters":{"type":"object","properties":{"operation":{"type":"string","enum":["add","subtract","multiply","divide"]},"a":{"type":"number"},"b":{"type":"number"}},"required":["operation","a","b"]}}}]}' | \
	curl -N -k -X POST $(HOST)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d @-

.PHONY: test-stream-tools-google
test-stream-tools-google:
	@echo "\n=========================================="
	@echo "Testing Google Streaming Tools: $(GOOGLE_MODEL)"
	@echo "==========================================\n"
	@echo '{"model":"$(GOOGLE_MODEL)","messages":[{"role":"user","content":"Search for AI news and get the weather in Tokyo"}],"stream":true,"temperature":$(TEMPERATURE),"max_tokens":$(MAX_TOKENS),"tools":[{"type":"function","function":{"name":"web_search","description":"Search the web for information","parameters":{"type":"object","properties":{"query":{"type":"string","description":"Search query"},"num_results":{"type":"integer","description":"Number of results","default":5}},"required":["query"]}}},{"type":"function","function":{"name":"get_weather","description":"Get current weather for a location","parameters":{"type":"object","properties":{"location":{"type":"string"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]}}}]}' | \
	curl -N -k -X POST $(HOST)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d @-

.PHONY: test-stream-tools-xai
test-stream-tools-xai:
	@echo "\n=========================================="
	@echo "Testing xAI Streaming Tools: $(XAI_MODEL)"
	@echo "==========================================\n"
	@echo '{"model":"$(XAI_MODEL)","messages":[{"role":"user","content":"Send emails to john@example.com and jane@example.com about tomorrows meeting at 2pm"}],"stream":true,"temperature":$(TEMPERATURE),"max_tokens":$(MAX_TOKENS),"tools":[{"type":"function","function":{"name":"send_email","description":"Send an email to a recipient","parameters":{"type":"object","properties":{"to":{"type":"string","description":"Recipient email"},"subject":{"type":"string","description":"Email subject"},"body":{"type":"string","description":"Email body"}},"required":["to","subject","body"]}}}]}' | \
	curl -N -k -X POST $(HOST)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d @-

# Quick streaming tool test
.PHONY: test-stream-tools-quick
test-stream-tools-quick:
	@echo "Running quick streaming tool test..."
	@$(MAKE) test-stream-tools-openai
	@$(MAKE) test-stream-tools-google

# Update test-all to include streaming tools
.PHONY: test-all
test-all: test-chat-all-models test-stream-all-models test-completion-all-models test-tools-all-models test-stream-tools-all-models
	@echo "\n=========================================="
	@echo "All tests completed!"
	@echo "==========================================\n"

# Update test-quick to include a streaming tool test
.PHONY: test-quick
test-quick:
	@echo "Running quick test suite..."
	@$(MAKE) test-chat-openai
	@$(MAKE) test-stream-google
	@$(MAKE) test-stream-tools-openai

# Quick tool test with one model per provider
.PHONY: test-tools-quick
test-tools-quick:
	@echo "Running quick tool test with one model per provider..."
	@$(MAKE) test-tools-google
	@$(MAKE) test-tools-openai
	@$(MAKE) test-tools-xai
	@$(MAKE) test-tools-anthropic


.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make run                         - Build and run enclave"
	@echo "  make test-local                  - Run server locally without TEE (dev)"
	@echo "  make test-chat                   - Test basic chat completion"
	@echo "  make test-stream                 - Test streaming completion"
	@echo "  make test-chat-all-models        - Test chat with all providers"
	@echo "  make test-stream-all-models      - Test streaming with all providers"
	@echo "  make test-tools-all-models       - Test tool calling with all providers"
	@echo "  make test-stream-tools-all-models - Test streaming tools with all providers"
	@echo "  make test-stream-tools-quick     - Quick streaming tool test (OpenAI + Google)"
	@echo "  make test-tools-multiturn        - Test multi-turn tool conversation"
	@echo "  make test-tools-complex          - Test complex tool parameters"
	@echo "  make test-tools-quick            - Quick tool test (Google + OpenAI)"
	@echo "  make test-all                    - Run all tests"
	@echo "  make test-quick                  - Quick test (chat, stream, streaming tools)"
	@echo ""
	@echo "Environment variables:"
	@echo "  HOST                             - API endpoint (default: https://127.0.0.1:443)"
	@echo "  MODEL                            - Model to use (default: gemini-2.5-flash-lite)"
	@echo "  OPENAI_MODEL                     - OpenAI model (default: gpt-4.1)"
	@echo "  ANTHROPIC_MODEL                  - Anthropic model (default: claude-3.7-sonnet)"
	@echo "  GOOGLE_MODEL                     - Google model (default: gemini-2.5-flash-lite)"
	@echo "  XAI_MODEL                        - xAI model (default: grok-3-mini-beta)"
	@echo ""
	@echo "Testing from the enclave host (loopback only, no payment required):"
	@echo "  make test-chat HOST=http://127.0.0.1:8000"
	@echo "  make test-chat-all-models HOST=http://127.0.0.1:8000"
	@echo "  Note: port 8000 is bound to 127.0.0.1 only and not reachable from the internet."
