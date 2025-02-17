#!/bin/bash

# This script starts the multi-agent debate simulation.

# Set default values for the arguments
API_KEY=""
SHARE=false
SERVER_PORT=8080
INBROWSER=false
CONCURRENCY_COUNT=1

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --api-key) API_KEY="$2"; shift ;;
        --share) SHARE=true ;;
        --server-port) SERVER_PORT="$2"; shift ;;
        --inbrowser) INBROWSER=true ;;
        --concurrency-count) CONCURRENCY_COUNT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Export the API key as an environment variable
export OPENAI_API_KEY=$API_KEY

# Start the multi-agent debate simulation
python3 -m apps.agents.agents \
    --api-key $API_KEY \
    --share $SHARE \
    --server-port $SERVER_PORT \
    --inbrowser $INBROWSER \
    --concurrency-count $CONCURRENCY_COUNT
