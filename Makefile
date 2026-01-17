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

# Tool calling tests for all providers
.PHONY: test-tools-all-models
test-tools-all-models: test-tools-openai test-tools-anthropic test-tools-google test-tools-xai

.PHONY: test-tools-openai
test-tools-openai:
	@echo "\n=========================================="
	@echo "Testing OpenAI Tools: $(OPENAI_MODEL)"
	@echo "==========================================\n"
	curl -i -k -X POST $(HOST)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{ \
			"model": "$(OPENAI_MODEL)", \
			"messages": [{"role": "user", "content": "What is the weather in San Francisco?"}], \
			"temperature": $(TEMPERATURE), \
			"max_tokens": $(MAX_TOKENS), \
			"tools": [{ \
				"type": "function", \
				"function": { \
					"name": "get_weather", \
					"description": "Get current weather for a location", \
					"parameters": { \
						"type": "object", \
						"properties": { \
							"location": {"type": "string", "description": "City name"}, \
							"unit": {"type": "string", "enum": ["celsius", "fahrenheit"]} \
						}, \
						"required": ["location"] \
					} \
				} \
			}] \
		}'

.PHONY: test-tools-anthropic
test-tools-anthropic:
	@echo "\n=========================================="
	@echo "Testing Anthropic Tools: $(ANTHROPIC_MODEL)"
	@echo "==========================================\n"
	curl -i -k -X POST $(HOST)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{ \
			"model": "$(ANTHROPIC_MODEL)", \
			"messages": [{"role": "user", "content": "Calculate 15 times 23 plus 47"}], \
			"temperature": $(TEMPERATURE), \
			"max_tokens": $(MAX_TOKENS), \
			"tools": [{ \
				"type": "function", \
				"function": { \
					"name": "calculator", \
					"description": "Perform basic arithmetic operations", \
					"parameters": { \
						"type": "object", \
						"properties": { \
							"operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]}, \
							"a": {"type": "number"}, \
							"b": {"type": "number"} \
						}, \
						"required": ["operation", "a", "b"] \
					} \
				} \
			}] \
		}'

.PHONY: test-tools-google
test-tools-google:
	@echo "\n=========================================="
	@echo "Testing Google Tools: $(GOOGLE_MODEL)"
	@echo "==========================================\n"
	curl -i -k -X POST $(HOST)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{ \
			"model": "$(GOOGLE_MODEL)", \
			"messages": [{"role": "user", "content": "Search for recent AI news"}], \
			"temperature": $(TEMPERATURE), \
			"max_tokens": $(MAX_TOKENS), \
			"tools": [{ \
				"type": "function", \
				"function": { \
					"name": "web_search", \
					"description": "Search the web for information", \
					"parameters": { \
						"type": "object", \
						"properties": { \
							"query": {"type": "string", "description": "Search query"}, \
							"num_results": {"type": "integer", "description": "Number of results", "default": 5} \
						}, \
						"required": ["query"] \
					} \
				} \
			}] \
		}'

.PHONY: test-tools-xai
test-tools-xai:
	@echo "\n=========================================="
	@echo "Testing xAI Tools: $(XAI_MODEL)"
	@echo "==========================================\n"
	curl -i -k -X POST $(HOST)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{ \
			"model": "$(XAI_MODEL)", \
			"messages": [{"role": "user", "content": "Send an email to john@example.com about the meeting tomorrow at 2pm"}], \
			"temperature": $(TEMPERATURE), \
			"max_tokens": $(MAX_TOKENS), \
			"tools": [{ \
				"type": "function", \
				"function": { \
					"name": "send_email", \
					"description": "Send an email to a recipient", \
					"parameters": { \
						"type": "object", \
						"properties": { \
							"to": {"type": "string", "description": "Recipient email"}, \
							"subject": {"type": "string", "description": "Email subject"}, \
							"body": {"type": "string", "description": "Email body"} \
						}, \
						"required": ["to", "subject", "body"] \
					} \
				} \
			}] \
		}'

# Multi-turn tool calling test (weather example)
.PHONY: test-tools-multiturn
test-tools-multiturn:
	@echo "\n=========================================="
	@echo "Testing Multi-turn Tool Calling: $(GOOGLE_MODEL)"
	@echo "==========================================\n"
	@echo "Step 1: Initial request with tool..."
	@curl -s -k -X POST $(HOST)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{ \
			"model": "$(GOOGLE_MODEL)", \
			"messages": [{"role": "user", "content": "What is the weather in Tokyo?"}], \
			"tools": [{ \
				"type": "function", \
				"function": { \
					"name": "get_weather", \
					"description": "Get current weather for a location", \
					"parameters": { \
						"type": "object", \
						"properties": { \
							"location": {"type": "string"}, \
							"unit": {"type": "string", "enum": ["celsius", "fahrenheit"]} \
						}, \
						"required": ["location"] \
					} \
				} \
			}] \
		}' > /tmp/tool_response.json
	@echo "\n"
	@cat /tmp/tool_response.json | python3 -m json.tool
	@echo "\n\nStep 2: Sending tool result back..."
	@# Extract tool_call_id and construct second request
	@python3 -c ' \
import json; \
resp = json.load(open("/tmp/tool_response.json")); \
tc = resp["message"]["tool_calls"][0]; \
req = { \
	"model": "$(GOOGLE_MODEL)", \
	"messages": [ \
		{"role": "user", "content": "What is the weather in Tokyo?"}, \
		{"role": "assistant", "content": "", "tool_calls": [tc]}, \
		{"role": "tool", "tool_call_id": tc["id"], "name": "get_weather", "content": "{\"temperature\": 18, \"condition\": \"cloudy\", \"humidity\": 75}"} \
	], \
	"tools": [{ \
		"type": "function", \
		"function": { \
			"name": "get_weather", \
			"description": "Get current weather for a location", \
			"parameters": { \
				"type": "object", \
				"properties": { \
					"location": {"type": "string"}, \
					"unit": {"type": "string", "enum": ["celsius", "fahrenheit"]} \
				}, \
				"required": ["location"] \
			} \
		} \
	}] \
}; \
print(json.dumps(req))' | \
	curl -s -k -X POST $(HOST)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d @- | python3 -m json.tool

# Database query tool test (complex parameters)
.PHONY: test-tools-complex
test-tools-complex:
	@echo "\n=========================================="
	@echo "Testing Complex Tool Parameters: $(ANTHROPIC_MODEL)"
	@echo "==========================================\n"
	curl -i -k -X POST $(HOST)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{ \
			"model": "$(ANTHROPIC_MODEL)", \
			"messages": [{"role": "user", "content": "Find all users who signed up in the last 7 days from California"}], \
			"temperature": $(TEMPERATURE), \
			"max_tokens": $(MAX_TOKENS), \
			"tools": [{ \
				"type": "function", \
				"function": { \
					"name": "query_database", \
					"description": "Query the user database with filters", \
					"parameters": { \
						"type": "object", \
						"properties": { \
							"table": {"type": "string", "description": "Database table name"}, \
							"filters": { \
								"type": "array", \
								"description": "Filter conditions", \
								"items": { \
									"type": "object", \
									"properties": { \
										"field": {"type": "string"}, \
										"operator": {"type": "string", "enum": ["equals", "greater_than", "less_than", "contains"]}, \
										"value": {} \
									}, \
									"required": ["field", "operator", "value"] \
								} \
							}, \
							"limit": {"type": "integer", "description": "Max results"} \
						}, \
						"required": ["table", "filters"] \
					} \
				} \
			}] \
		}'

# Comprehensive test suite
.PHONY: test-all
test-all: test-chat-all-models test-stream-all-models test-completion-all-models test-tools-all-models
	@echo "\n=========================================="
	@echo "All tests completed!"
	@echo "==========================================\n"

# Quick test with just one model per provider
.PHONY: test-quick
test-quick:
	@echo "Running quick test with one model per provider..."
	@$(MAKE) test-chat-openai
	@$(MAKE) test-chat-google

# Quick tool test with one model per provider
.PHONY: test-tools-quick
test-tools-quick:
	@echo "Running quick tool test with one model per provider..."
	@$(MAKE) test-tools-google
	@$(MAKE) test-tools-openai

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make run                      - Build and run enclave"
	@echo "  make test-chat                - Test basic chat completion"
	@echo "  make test-stream              - Test streaming completion"
	@echo "  make test-chat-all-models     - Test chat with all providers"
	@echo "  make test-stream-all-models   - Test streaming with all providers"
	@echo "  make test-tools-all-models    - Test tool calling with all providers"
	@echo "  make test-tools-multiturn     - Test multi-turn tool conversation"
	@echo "  make test-tools-complex       - Test complex tool parameters"
	@echo "  make test-tools-quick         - Quick tool test (Google + OpenAI)"
	@echo "  make test-all                 - Run all tests"
	@echo "  make test-quick               - Quick test (chat only)"
	@echo ""
	@echo "Environment variables:"
	@echo "  HOST                          - API endpoint (default: https://127.0.0.1:443)"
	@echo "  MODEL                         - Model to use (default: gemini-2.5-flash-lite)"
	@echo "  OPENAI_MODEL                  - OpenAI model (default: gpt-4o)"
	@echo "  ANTHROPIC_MODEL               - Anthropic model (default: claude-3.7-sonnet)"
	@echo "  GOOGLE_MODEL                  - Google model (default: gemini-2.5-flash-lite)"
	@echo "  XAI_MODEL                     - xAI model (default: grok-3-mini-beta)"
