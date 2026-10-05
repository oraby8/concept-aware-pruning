# concept-prune

Concept-aware structured (MLP-width) pruning for Hugging Face causal LMs.

Instead of pruning purely by weight magnitude, this blends in an
attribution-based "does this neuron matter for concept X?" signal, so you
can protect specific capabilities (a linguistic pattern, a fact type, or
even something task-shaped like "translates correctly") while still hitting
your target compression ratio.

Validated on Llama-3.2-1B, Qwen2.5-1.5B, Qwen3-0.6B, and
`google/gemma-4-E2B-it` (a production Arabic-translation model) — see
"What's actually validated" below for the numbers. This README covers
install + usage; it is not the research writeup.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
# or: pip install -r requirements.txt && pip install -e . --no-deps
```

Gated models (e.g. `meta-llama/Llama-3.2-1B`) need `huggingface-cli login`
first, or use a public mirror (`unsloth/Llama-3.2-1B` is an unquantized,
ungated copy of the same weights, used throughout development).

## Quickstart

```bash
concept-prune run \
  --model unsloth/Llama-3.2-1B \
  --concepts examples/concepts_ioi_sva.yaml \
  --holdout-concepts examples/concepts_ioi_sva_holdout.yaml \
  --compression 0.4 \
  --beta 2.0 \
  --benchmarks lambada_openai \
  --limit 200 \
  --output results.json
```

This prunes the model two ways — plain magnitude-only (baseline) and
concept-aware — at 40% MLP-width compression, evaluates both (plus the
dense model) on `lambada_openai` and on held-out concept accuracy, and
writes a comparison JSON. See `examples/run_llama_example.sh` for the same
thing as a script, and `examples/run_translation_example.sh` for the
translation-quality variant.

### CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--model` | required | HF model name or local path |
| `--concepts` | required | YAML describing which concepts to protect (see below) |
| `--holdout-concepts` | none | Separate YAML for a non-circular held-out accuracy read |
| `--compression` | 0.4 | Fraction of MLP width removed per layer |
| `--beta` | 2.0 | Concept-coverage blend weight (0 = plain magnitude pruning) |
| `--aggregate` | max | How multiple concepts combine (`max` or `sum`) |
| `--m-steps` | 4 | Integrated-gradients interpolation steps |
| `--n-per-concept` | 60 | Probe pairs per concept |
| `--benchmarks` | none | Comma-separated `lm_eval` tasks, e.g. `arc_easy,hellaswag,piqa,winogrande,lambada_openai` |
| `--limit` | none | Per-task example limit for `lm_eval` (use ≥200 for anything you'll draw a conclusion from — see "Common pitfalls") |
| `--bf16` | off | Load in bfloat16 instead of float16 |
| `--output` | `concept_prune_results.json` | Where to write results |

## Defining concepts

A **concept** is a name plus a list of clean/corrupted prompt pairs, where
the model should predict a specific token after the *clean* prompt but a
different context after the *corrupted* one. The two prompts must tokenize
to the same length (required by the attribution hooks).

`--concepts` points at a YAML file:

```yaml
concepts:
  ioi:
    template: builtin:ioi       # one of the built-in generators (see below)
  my_custom_concept:
    pairs_file: my_pairs.jsonl  # or: your own clean/corrupted/label triples
```

**Built-in templates** (`concept_prune/examples_templates.py`): `ioi`
(indirect object identification), `sva` (subject-verb agreement), `factual`
(country/capital recall). Write your own by copying the same signature —
`def generate_x(n_candidates: int) -> list[ConceptPair]` — and referencing
it via a custom loader, or just build a `pairs_file` directly.

**`pairs_file` format** — JSONL, one row per pair:

```json
{"concept": "my_concept", "clean": "...", "corrupted": "...", "label": " Paris"}
```

`label` can be a string (tokenized alone, first token used — fine for
simple cases) or a pre-resolved integer token id (safer when the label
token's identity depends on surrounding context — see the translation
pattern below).

### Concept = task quality (e.g. translation)

A concept doesn't have to be a narrow linguistic probe — it can be "the
model produces the right output for this task". For translation: clean =
(source text, correct target's first token), corrupted = the same target
paired with a *different* source. `examples/build_translation_pairs.py`
builds this from a source/target JSONL:

```bash
python examples/build_translation_pairs.py \
  --model google/gemma-4-E2B-it \
  --data your_data.jsonl --source-key source --target-key target \
  --system-prompt "You are a professional translator. ..." \
  --n-pairs 50 --concept-name translation_ar --out translation_pairs.jsonl
```

Then reference `translation_pairs.jsonl` from a `pairs_file` concept (see
`examples/concepts_translation.yaml`). For the actual translation-quality
number (chrF/BLEU on generated text, not next-token accuracy), write a
small script using `concept_prune.evaluation.evaluate_custom()` — the CLI's
built-in eval only does next-token concept accuracy and `lm_eval` tasks.

## Python API

The CLI is a thin wrapper; use the package directly for anything custom:

```python
from concept_prune import load_model, find_decoder_layers
from concept_prune.concepts import concepts_from_templates
from concept_prune.examples_templates import generate_ioi, generate_sva
from concept_prune.attribution import compute_concept_importance, combine_concepts
from concept_prune.pruning import prune_model
from concept_prune.evaluation import evaluate_concept_accuracy, evaluate_lm_eval

model, tokenizer = load_model("unsloth/Llama-3.2-1B")
layers = find_decoder_layers(model)  # handles wrapped/multimodal models too

concepts = concepts_from_templates(tokenizer, {"ioi": generate_ioi, "sva": generate_sva}, n_per_concept=60)
importance = {name: compute_concept_importance(model, pairs, layers=layers) for name, pairs in concepts.items()}
coverage = combine_concepts(importance)

baseline = prune_model(model, 0.4, coverage_by_layer=None, beta=0.0)      # magnitude-only
concept_aware = prune_model(model, 0.4, coverage_by_layer=coverage, beta=2.0)

print(evaluate_lm_eval(concept_aware, tokenizer, ["lambada_openai"], limit=200))
print(evaluate_concept_accuracy(concept_aware, concepts))
```
