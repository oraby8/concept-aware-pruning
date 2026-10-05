"""`concept-prune run` -- CLI entry point.

Runs baseline (magnitude-only) and concept-aware pruning side by side on one
model, evaluates both (plus the dense model) on held-out concept accuracy
and/or standard lm_eval benchmarks, and writes a comparison JSON.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time

import torch
import yaml

from .attribution import combine_concepts, compute_concept_importance
from .concepts import ConceptPair, concepts_from_pairs_file, concepts_from_templates
from .evaluation import evaluate_concept_accuracy, evaluate_lm_eval
from .model_utils import find_decoder_layers, load_model
from .pruning import count_parameters, prune_model


def _load_concepts_yaml(path):
    """Load a concepts.yaml describing template-generated pairs, e.g.:

        concepts:
          ioi:
            template: builtin:ioi
          sva:
            template: builtin:sva
    or, for pre-built pair files:
          my_concept:
            pairs_file: my_pairs.jsonl

    ``builtin:*`` names map to the example generators shipped in
    concept_prune.examples_templates (ioi, sva, greaterthan, factual) --
    see examples/concepts_ioi_sva.yaml for the reference file.
    """
    with open(path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    return spec["concepts"]


def _builtin_generator(name):
    from . import examples_templates

    return getattr(examples_templates, f"generate_{name}")


def _build_resolved_concepts(tokenizer, concepts_spec, n_per_concept, seed):
    random.seed(seed)
    generators = {}
    file_concepts = {}
    for name, cfg in concepts_spec.items():
        if "template" in cfg:
            builtin_name = cfg["template"].split(":", 1)[1]
            generators[name] = _builtin_generator(builtin_name)
        elif "pairs_file" in cfg:
            file_concepts.update(concepts_from_pairs_file(tokenizer, cfg["pairs_file"], n_per_concept))
        else:
            raise ValueError(f"concept {name!r} needs a 'template' or 'pairs_file' key")
    resolved = {}
    if generators:
        resolved.update(concepts_from_templates(tokenizer, generators, n_per_concept))
    resolved.update(file_concepts)
    return resolved


def run(args):
    t0 = time.time()
    model, tokenizer = load_model(args.model, dtype=torch.bfloat16 if args.bf16 else torch.float16)
    layers = find_decoder_layers(model)
    print(f"[time] model load: {time.time()-t0:.1f}s  layers={len(layers)}")

    concepts_spec = _load_concepts_yaml(args.concepts)
    train_resolved = _build_resolved_concepts(tokenizer, concepts_spec, args.n_per_concept, seed=0)

    holdout_resolved = None
    if args.holdout_concepts:
        holdout_spec = _load_concepts_yaml(args.holdout_concepts)
        holdout_resolved = _build_resolved_concepts(tokenizer, holdout_spec, args.n_per_concept, seed=1)

    print("=== computing per-concept attribution ===")
    t0 = time.time()
    concept_importance = {
        name: compute_concept_importance(model, pairs, m_steps=args.m_steps, layers=layers)
        for name, pairs in train_resolved.items()
    }
    coverage_by_layer = combine_concepts(concept_importance, aggregate=args.aggregate)
    print(f"[time] attribution: {time.time()-t0:.1f}s")

    benchmarks = [t.strip() for t in args.benchmarks.split(",")] if args.benchmarks else []

    def _evaluate(m, label):
        result = {}
        if benchmarks:
            result["lm_eval"] = evaluate_lm_eval(m, tokenizer, benchmarks, limit=args.limit)
        result["concept_acc_train"] = evaluate_concept_accuracy(m, train_resolved)
        if holdout_resolved:
            result["concept_acc_holdout"] = evaluate_concept_accuracy(m, holdout_resolved)
        print(f"=== {label} === {json.dumps(result, indent=2, default=str)}")
        return result

    summary = {"config": vars(args)}
    summary["dense"] = _evaluate(model, "DENSE")
    summary["dense"]["params"] = count_parameters(model)

    print("=== baseline (magnitude-only) pruning ===")
    model_baseline = prune_model(model, args.compression, coverage_by_layer=None, beta=0.0, layers=None)
    summary["baseline_pruned"] = _evaluate(model_baseline, "BASELINE PRUNED")
    summary["baseline_pruned"]["params"] = count_parameters(model_baseline)
    del model_baseline
    torch.cuda.empty_cache()

    print("=== concept-aware pruning ===")
    model_concept = prune_model(model, args.compression, coverage_by_layer=coverage_by_layer, beta=args.beta, layers=None)
    summary["concept_aware_pruned"] = _evaluate(model_concept, "CONCEPT-AWARE PRUNED")
    summary["concept_aware_pruned"]["params"] = count_parameters(model_concept)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nSaved results to {args.output}")


def main():
    parser = argparse.ArgumentParser(prog="concept-prune")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Prune a model both ways (baseline + concept-aware) and compare.")
    p_run.add_argument("--model", required=True, help="HF model name or local path")
    p_run.add_argument("--concepts", required=True, help="Path to a concepts.yaml (attribution/train pairs)")
    p_run.add_argument("--holdout-concepts", default=None, help="Optional concepts.yaml for non-circular held-out eval")
    p_run.add_argument("--compression", type=float, default=0.4, help="Fraction of MLP width to remove (default 0.4)")
    p_run.add_argument("--beta", type=float, default=2.0, help="Concept-coverage blend weight (default 2.0, the validated default)")
    p_run.add_argument("--aggregate", choices=["max", "sum"], default="max", help="How to combine multiple concepts' coverage")
    p_run.add_argument("--m-steps", type=int, default=4, help="Integrated-gradients interpolation steps")
    p_run.add_argument("--n-per-concept", type=int, default=60, help="Probe pairs per concept")
    p_run.add_argument("--benchmarks", default="", help="Comma-separated lm_eval task names, e.g. arc_easy,hellaswag,lambada_openai")
    p_run.add_argument("--limit", type=int, default=None, help="lm_eval per-task example limit")
    p_run.add_argument("--bf16", action="store_true", help="Load in bfloat16 (default float16)")
    p_run.add_argument("--output", default="concept_prune_results.json")
    p_run.set_defaults(func=run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
