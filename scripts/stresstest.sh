#!/bin/bash

# Stress test script for TEE LLM Router
# Usage: ./stresstest.sh [num_requests] [parallel_jobs]

# Configuration
HOST="${HOST:-https://3.133.152.176}"
NUM_REQUESTS="${1:-100}"
PARALLEL_JOBS="${2:-10}"
MODEL="${MODEL:-gemini-2.0-flash}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo "========================================="
echo "TEE LLM Router Stress Test"
echo "========================================="
echo "Host: $HOST"
echo "Requests: $NUM_REQUESTS"
echo "Parallel Jobs: $PARALLEL_JOBS"
echo "Model: $MODEL"
echo "========================================="
echo ""

# Function to send a single request
send_request() {
    local request_id=$1

    response=$(curl -X POST "$HOST/v1/chat/completions" -k \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $GOOGLE_API_KEY" \
        -w "\nHTTP_CODE:%{http_code}\nTIME_TOTAL:%{time_total}\n" \
        -s \
        -d '{
          "model": "'"$MODEL"'",
          "messages": [
            {"role": "user", "content": "What is the weather in San Francisco?"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "abc123", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\": \"San Francisco\"}"}}]},
            {"role": "tool", "tool_call_id": "abc123", "content": "72°F and sunny"}
          ],
          "tools": [
            {"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}
          ]
        }' 2>&1)

    http_code=$(echo "$response" | grep "HTTP_CODE:" | cut -d: -f2)
    time_total=$(echo "$response" | grep "TIME_TOTAL:" | cut -d: -f2)

    if [ "$http_code" = "200" ]; then
        echo -e "${GREEN}✓${NC} Request $request_id: Success (${time_total}s)"
    else
        echo -e "${RED}✗${NC} Request $request_id: Failed (HTTP $http_code)"
    fi
}

export -f send_request
export HOST MODEL GREEN RED NC

# Start timer
overall_start=$(date +%s)

echo "Starting stress test..."
echo ""

seq 1 "$NUM_REQUESTS" | xargs -P "$PARALLEL_JOBS" -I {} bash -c 'send_request "$@"' _ {}

# End timer
overall_end=$(date +%s)
overall_duration=$((overall_end - overall_start))

echo ""
echo "========================================="
echo "Stress Test Complete"
echo "========================================="
echo "Total Requests:     $NUM_REQUESTS"
echo "Total Duration:     ${overall_duration}s"
if [ "$overall_duration" -gt 0 ]; then
    rps=$(echo "scale=2; $NUM_REQUESTS / $overall_duration" | bc)
    echo "Requests/Second:    $rps"
fi
echo "========================================="
