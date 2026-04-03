#!/bin/bash
export MASTER_PORT=29509
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,1

# model
model_path=ckpt/MSA-4B

# data
# bench_name:batch_size
# Adjust the batch size appropriately based on the number of GPUs.
benchmarks=(
    "ms_100M:16"
    "hipporag_narrative:16"
    "nature_questions:16"
    "2wikimultihopqa:16"
    "hotpotqa:16"
    "musique:16"
    "hipporag_popqa:16"
    "triviaqa_06M:16"
    "triviaqa_10M:2"
    "dureader:16"
    "msmarco_v1:16"
)

top_p=0.9
temperature=0.0
max_length=2048
template=QWEN3_INSTRUCT_TEMPLATE

# Create log directory (use first argument as custom name, otherwise use timestamp)
log_name="${1:-eval_logs_$(date +%Y%m%d_%H%M%S)}"
log_dir="./src/evaluation/outputs/${log_name}"
if [ -d "$log_dir" ]; then
    echo "Error: log directory already exists: $log_dir"
    exit 1
fi
mkdir -p "$log_dir"

# Statistics
total_benchmarks=${#benchmarks[@]}
current=0
success_count=0
fail_count=0
failed_benchmarks=()

echo "=========================================="
echo "Start running all benchmark evaluations"
echo "Total: ${total_benchmarks} benchmarks"
echo "Log directory: $log_dir"
echo "=========================================="
echo ""

# Run each benchmark
for entry in "${benchmarks[@]}"; do
    benchmark="${entry%%:*}"
    batch_size="${entry##*:}"
    current=$((current + 1))
    echo "[$current/$total_benchmarks] Running: $benchmark (batch_size=$batch_size)"
    echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"

    # Create separate log file for each benchmark
    log_file="$log_dir/${benchmark}.log"
    json_file="$log_dir/${benchmark}.json"

    # Run evaluation and record logs
    python -u src/app/benchmark.py \
        --benchmark "$benchmark" \
        --model_path "$model_path" \
        --top_p "$top_p" \
        --temperature "$temperature" \
        --max_length "$max_length" \
        --template "$template" \
        --output_file "$json_file" \
        --max_batch_size "$batch_size"  \
        --max_chunk_per_block 16384 \
        --block_size 2048 \
        2>&1 | tee $log_file

    # Check exit status
    exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        echo "[$current/$total_benchmarks] $benchmark finished (success)"
        # Print benchmark name and metrics
        echo "========== $benchmark Results =========="
        python -c "import json; d=json.load(open('$json_file')); [print(f'  {k}: {v}') for k,v in d.get(list(d.keys())[0],{}).get('precision',{}).get('metrics',{}).items()]" 2>/dev/null || echo "  (failed to parse metrics)"
        echo "========================================"
        success_count=$((success_count + 1))
    else
        echo "[$current/$total_benchmarks] $benchmark failed (exit code: $exit_code)"
        fail_count=$((fail_count + 1))
        failed_benchmarks+=("$benchmark")
    fi

    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "----------------------------------------"
    echo ""
done

# Summary
echo "=========================================="
echo "All benchmark evaluations completed"
echo "=========================================="
echo "Total: $total_benchmarks"
echo "Success: $success_count"
echo "Failed: $fail_count"
echo ""

if [ $fail_count -gt 0 ]; then
    echo "Failed benchmarks:"
    for failed in "${failed_benchmarks[@]}"; do
        echo "  - $failed"
    done
    echo ""
fi

echo "All logs saved in: $log_dir"
echo "=========================================="
