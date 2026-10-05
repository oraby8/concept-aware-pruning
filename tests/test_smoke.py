"""Fast end-to-end smoke test: tiny model + tiny synthetic concept + tiny
compression, asserts the full pipeline (attribution -> blended pruning ->
concept accuracy eval) runs without error and produces a genuinely smaller,
still-loadable model. Not a correctness/quality test -- see README for how
the method's actual effectiveness was validated.
"""
import torch

from concept_prune.attribution import combine_concepts, compute_concept_importance
from concept_prune.concepts import concepts_from_templates
from concept_prune.evaluation import evaluate_concept_accuracy
from concept_prune.examples_templates import generate_ioi
from concept_prune.model_utils import find_decoder_layers, load_model
from concept_prune.pruning import count_parameters, prune_model

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def test_smoke_end_to_end():
    model, tokenizer = load_model(TINY_MODEL, dtype=torch.float32, device_map=None)
    layers = find_decoder_layers(model)
    assert len(layers) > 0

    resolved = concepts_from_templates(tokenizer, {"ioi": generate_ioi}, n_per_concept=4)
    assert len(resolved["ioi"]) > 0, "no valid IOI pairs resolved for the tiny model's tokenizer"

    importance = {"ioi": compute_concept_importance(model, resolved["ioi"], m_steps=2, layers=layers)}
    coverage = combine_concepts(importance)
    assert set(coverage.keys()) == set(range(len(layers)))

    orig_params = count_parameters(model)
    pruned = prune_model(model, prune_percent=0.3, coverage_by_layer=coverage, beta=2.0, layers=None)
    pruned_params = count_parameters(pruned)
    assert pruned_params < orig_params, "pruned model should have fewer parameters than dense"

    baseline = prune_model(model, prune_percent=0.3, coverage_by_layer=None, beta=0.0, layers=None)
    assert count_parameters(baseline) == pruned_params, "baseline and concept-aware should prune the same count"

    acc = evaluate_concept_accuracy(pruned, resolved)
    assert "ioi" in acc and 0.0 <= acc["ioi"] <= 1.0

    # sanity: a forward pass on the pruned model shouldn't error
    input_ids = tokenizer("hello world", return_tensors="pt").input_ids
    with torch.no_grad():
        out = pruned(input_ids=input_ids)
    assert out.logits.shape[0] == 1


if __name__ == "__main__":
    test_smoke_end_to_end()
    print("smoke test passed")
