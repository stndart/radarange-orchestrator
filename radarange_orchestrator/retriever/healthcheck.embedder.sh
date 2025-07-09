#!/bin/bash

API_KEY="EMPTY"
EMBEDDER_API_BASE="http://localhost:12400"

EMBEDDER_MODEL=$(curl -s -X GET "$EMBEDDER_API_BASE/v1/models" -H "Authorization: Bearer $API_KEY" | jq -r '.data[0].id')

# Exit if the model is not found
if [[ -z "$EMBEDDER_MODEL" || "$EMBEDDER_MODEL" == "null" ]]; then
    exit 1  # Unhealthy
fi

# Send an embedding request
EMBEDDER_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$EMBEDDER_API_BASE/v1/embeddings" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$EMBEDDER_MODEL"'",
        "input": ["Привет!"]
    }')

# Check if the response is HTTP 200
if [[ "$EMBEDDER_RESPONSE" == "200" ]]; then
    exit 0  # Healthy
else
    exit 1  # Unhealthy
fi
