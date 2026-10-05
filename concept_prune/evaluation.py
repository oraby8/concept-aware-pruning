"""Evaluation helpers: held-out concept accuracy, standard lm_eval benchmarks,
and a generic hook for task-specific metrics (e.g. chrF/BLEU for translation).
"""
from __future__ import annotations

import torch


@torch.no_grad()
def evaluate_concept_accuracy(model, resolved_pairs_by_concept: dict) -> dict:
    """Next-token argmax accuracy on each concept's (held-out, ideally)
    resolved pairs. ``resolved_pairs_by_concept`` has the same shape as
    concepts_from_templates()'s return value:
    ``{concept_name: [(clean_ids, corrupted_ids, label_id), ...]}``.
    """
    device = next(model.parameters()).device
    results = {}
    for concept, pairs in resolved_pairs_by_concept.items():
        correct = 0
        for clean_ids, _corr_ids, label_id in pairs:
            clean_ids = clean_ids.to(device)
            attn_mask = torch.ones_like(clean_ids)
            out = model(input_ids=clean_ids, attention_mask=attn_mask)
            pred = out.logits[:, -1, :].argmax(dim=-1).item()
            if pred == label_id:
                correct += 1
        results[concept] = correct / max(len(pairs), 1)
    return results


def evaluate_lm_eval(model, tokenizer, tasks: list[str], limit: int | None = None, batch_size: int = 4) -> dict:
    """Run the standard lm-evaluation-harness over *tasks* (e.g.
    ["arc_easy", "hellaswag", "piqa", "winogrande", "lambada_openai"]).
    Thin wrapper so callers don't need to know the lm_eval API directly.
    """
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    results = evaluator.simple_evaluate(model=lm, tasks=tasks, limit=limit, num_fewshot=0)
    return results.get("results", results)


def evaluate_custom(model, tokenizer, eval_fn, **kwargs):
    """Escape hatch for task-specific evaluation that doesn't fit
    evaluate_concept_accuracy or lm_eval (e.g. chrF/BLEU translation
    quality against a held-out reference set). ``eval_fn(model, tokenizer,
    **kwargs) -> dict`` is called directly and its return value passed
    through -- see examples/run_translation_example.sh for the reference
    pattern used to score the Gemma4 Arabic-translation result.
    """
    return eval_fn(model, tokenizer, **kwargs)
