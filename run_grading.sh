#!/bin/bash
#SBATCH --job-name=medic-grading
#SBATCH --output=logs/grading_%j.out
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --account=<your-slurm-account>

# Grade MedExpert responses with each LLM Judge reported in the paper.
#
# Local judges are served with vLLM; set SERVER to your endpoint. Start one server per
# model and run this script once per model, or edit MODELS below to loop.
#
# Usage:  sbatch run_grading.sh            (cluster)
#         bash   run_grading.sh            (local, with a reachable SERVER)

set -euo pipefail

source .env
module load anaconda3/2024.02-1
conda activate medic-complete

export PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
export PYTHONPATH="${PROJECT_DIR}/code:${PYTHONPATH:-}"
export CACHE_DIR="${PROJECT_DIR}/cache"
export SERVER="http://localhost:8000/v1"
export GRADER_API_KEY="${GRADER_API_KEY:-EMPTY}"
export BATCH_SIZE=8

mkdir -p logs

# The two primary metrics. Add completeness-medexpert for the zero-shot appendix variant.
METRICS="completeness-medexpert-fss completeness-icl-healthbench"

# model_id : max_tokens : temperature   (per-model settings from the paper)
MODELS=(
  "meta-llama/Llama-3.3-70B-Instruct:4096:0"
  "aaditya/Llama3-OpenBioLLM-70B:1024:0"
  "Qwen/Qwen3.6-35B-A3B:32768:0"
  "gpt-5-mini-2025-08-07:8192:1.0"
  "prometheus-eval/prometheus-7b-v2.0:4096:0"
)

for entry in "${MODELS[@]}"; do
  IFS=":" read -r MODEL_ID MAX_TOK TEMP <<< "${entry}"

  export MODEL="${MODEL_ID}"
  export MAX_TOKENS="${MAX_TOK}"
  export DECODING_TEMPERATURE="${TEMP}"

  # Prometheus uses its own prompt format and is excluded from Criteria-Checklist.
  if [[ "${MODEL_ID}" == prometheus-eval/* ]]; then
    RUN_METRICS="completeness-medexpert-prometheus"
  else
    RUN_METRICS="${METRICS}"
  fi

  # OpenAI models go to the API rather than a local vLLM server.
  if [[ "${MODEL_ID}" == gpt-* ]]; then
    export SERVER="https://api.openai.com/v1"
    export GRADER_API_KEY="${OPENAI_API_KEY}"
  fi

  echo "=== ${MODEL_ID} : ${RUN_METRICS} ==="
  python -m medscore_completeness.scripts.grade_responses \
    --dataset MedExpert \
    --dataset-kwargs '{"n_examples": -1}' \
    --metrics ${RUN_METRICS} \
    --output-dir "${PROJECT_DIR}/results/reframe/${MODEL_ID}/MedExpert" \
    --config config.yaml
done
