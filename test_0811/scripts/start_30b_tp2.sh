#!/bin/bash
# 双卡 TP=2 启动 Qwen3-VL-30B-A3B
# 用法: ./start_30b_tp2.sh [端口] [日志文件]
set -euo pipefail

MODEL=${MODEL:-/data/models/Qwen3-VL-30B-A3B-Instruct}
PORT=${1:-8001}
LOG=${2:-/tmp/vllm_qwen3vl_30b.log}

echo ">>> 启动 $MODEL  TP=2 @ :$PORT  (日志: $LOG)"
ASCEND_RT_VISIBLE_DEVICES=0,1 vllm serve "$MODEL" \
  --port "$PORT" --trust-remote-code \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 --max-model-len 8192 \
  --limit-mm-per-prompt '{"image": 1}' \
  > "$LOG" 2>&1 &

PID=$!
echo ">>> PID=$PID  等待服务就绪..."
for i in $(seq 1 60); do
  if curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo ">>> 服务已就绪 (约 ${i}0s)"
    break
  fi
  sleep 10
done
echo ">>> 测试: python test_vision.py --url http://127.0.0.1:$PORT"
