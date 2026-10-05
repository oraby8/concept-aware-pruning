#!/usr/bin/env bash
# Roughly reproduces the validated Llama-3.2-1B / 40% / beta=2.0 result.
# Known reference numbers (lm_eval limit=200, see README):
#   dense                 Lambada-OpenAI PPL 5.62
#   baseline (beta=0)     Lambada-OpenAI PPL 88.52
#   concept-aware (beta=2) Lambada-OpenAI PPL ~24-30
set -euo pipefail
cd "$(dirname "$0")/.."

concept-prune run \
  --model unsloth/Llama-3.2-1B \
  --concepts examples/concepts_ioi_sva.yaml \
  --holdout-concepts examples/concepts_ioi_sva_holdout.yaml \
  --compression 0.4 \
  --beta 2.0 \
  --benchmarks lambada_openai \
  --limit 200 \
  --output llama_example_results.json
