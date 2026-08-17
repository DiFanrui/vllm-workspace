#!/bin/bash
set -u

cd /root/InfiniLM
source .venv/bin/activate

python python/infinilm/server/inference_server.py \
  --device nvidia \
  --model=/root/autodl-tmp/models/Llama-3.1-8B-Instruct \
  --port 8102 \
  --tp 1 \
  --max-new-tokens 4096 \
  --num-blocks 64 \
  --max-batch-size 64 \
  --enable-graph \
  --enable-paged-attn \
  --attn flash-attn \
  --ignore-eos
