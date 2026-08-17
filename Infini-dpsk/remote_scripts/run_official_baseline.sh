#!/bin/bash
set -u

source /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/vllm-bench312-conda/

model_path="/root/autodl-tmp/models/Llama-3.1-8B-Instruct"
run_dir="/root/autodl-tmp/bench-runs/official-8b-4090-baseline-20260711"
result_dir="${run_dir}/results"
summary_file="${run_dir}/summary.tsv"
port=8102

mkdir -p "$result_dir"

if [[ ! -f "$summary_file" ]]; then
  printf 'timestamp\tmodel\tgpu\tcommit\tconcurrency\tinput_len\toutput_len\tnum_prompts\tseed\tlog_file\tstatus\n' > "$summary_file"
fi

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
commit=$(cd /root/InfiniLM && git rev-parse --short HEAD 2>/dev/null || echo unknown)
batch_size_list=(1 4 16 64)
random_output_len_list=(256 1024 4096)

for batch_size in "${batch_size_list[@]}"; do
  if [[ "$batch_size" == "1" || "$batch_size" == "4" ]]; then
    random_input_len_list=(32 256 4096)
  else
    random_input_len_list=(32 256)
  fi

  for random_input_len in "${random_input_len_list[@]}"; do
    for random_output_len in "${random_output_len_list[@]}"; do
      sleep 1
      seed=$(date +%s)
      num_prompts=$((batch_size * 10))
      if [[ "$num_prompts" -lt 20 ]]; then
        num_prompts=20
      fi

      log_file="${result_dir}/bench_8b_con=${batch_size}_in=${random_input_len}_out=${random_output_len}_seed=${seed}.log"
      status_file="${result_dir}/current_status.txt"

      {
        echo "=================================================="
        echo "Running official 8B baseline"
        echo "timestamp=$(date '+%F %T')"
        echo "gpu=${gpu_name}"
        echo "commit=${commit}"
        echo "batch_size/max_concurrency=${batch_size}"
        echo "input_len=${random_input_len}"
        echo "output_len=${random_output_len}"
        echo "num_prompts=${num_prompts}"
        echo "seed=${seed}"
        echo "model_path=${model_path}"
        echo "result_dir=${result_dir}"
        echo "=================================================="
      } | tee -a "$log_file"

      echo "running con=${batch_size} in=${random_input_len} out=${random_output_len} seed=${seed} log=${log_file}" > "$status_file"

      if vllm bench serve \
        --model "$model_path" \
        --port "$port" \
        --backend openai-chat \
        --tokenizer "$model_path" \
        --endpoint /v1/chat/completions \
        --request-rate inf \
        --num-prompts "$num_prompts" \
        --max-concurrency "$batch_size" \
        --random-input-len "$random_input_len" \
        --random-output-len "$random_output_len" \
        --ignore-eos \
        --save-result \
        --result-dir "$result_dir" \
        --seed "$seed" >> "$log_file" 2>&1; then
        status=ok
      else
        status=failed
      fi

      echo "Finished: con=${batch_size}, in=${random_input_len}, out=${random_output_len}, status=${status}" | tee -a "$log_file"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "Llama-3.1-8B-Instruct" "$gpu_name" "$commit" "$batch_size" "$random_input_len" "$random_output_len" "$num_prompts" "$seed" "$log_file" "$status" >> "$summary_file"
    done
  done
done

echo "done $(date '+%F %T')" > "${result_dir}/current_status.txt"
