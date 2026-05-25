#!/bin/bash
# Load API keys from keys.txt file

if [ -f "keys.txt" ]; then
    echo "[Info] Loading API keys from keys.txt"
    source keys.txt
    echo "[Info] API keys loaded successfully"
else
    echo "[Warning] keys.txt not found, using environment variables"
fi
