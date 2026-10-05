#!/usr/bin/env bash
# Roughly reproduces the validated Gemma4 Arabic-translation result
# (real chrF/BLEU on held-out WMT24++-style data, 20% compression, beta=2.0
# -- see README "Arabic translation result"). Requires your own EN->AR (or
# any language pair) source/target JSONL; this script does not ship one.
#
# Usage: ./run_translation_example.sh path/to/en_ar_data.jsonl
set -euo pipefail
cd "$(dirname "$0")/.."

DATA="${1:?usage: run_translation_example.sh path/to/source_target.jsonl}"
MODEL="google/gemma-4-E2B-it"
SYSTEM_PROMPT="You are a professional translator. Translate the user text from English to Arabic (Modern Standard Arabic). Output only the translation, no extra text."

python examples/build_translation_pairs.py \
  --model "$MODEL" \
  --data "$DATA" --source-key source --target-key target \
  --system-prompt "$SYSTEM_PROMPT" \
  --n-pairs 50 --concept-name translation_ar \
  --out translation_pairs.jsonl

concept-prune run \
  --model "$MODEL" \
  --concepts examples/concepts_translation.yaml \
  --compression 0.2 \
  --beta 2.0 \
  --bf16 \
  --output gemma4_translation_results.json

echo "Pruned models saved scores only -- see gemma4_translation_results.json."
echo "For real translation-quality scoring (chrF/BLEU on generated text, not"
echo "just next-token accuracy), you'll need a small custom script using"
echo "concept_prune.evaluation.evaluate_custom() -- see README 'Concept = task"
echo "quality' section for the pattern used to score the Gemma4 result."
