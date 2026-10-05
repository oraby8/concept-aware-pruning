"""Blend magnitude-based neuron importance with concept-coverage importance
into a single pruning score.

Validated formula (found, via a full sweep across beta in {0, 0.25, 0.5, 1,
2, 4} and compression ratios 20-60%, to give a clean win over magnitude-only
pruning at every ratio tested with beta=2.0 as the default -- see README):
min-max normalize both terms to [0, 1], then ``score = magnitude_norm +
beta * coverage_norm``. beta=0 exactly reproduces magnitude-only pruning
(useful as a sanity check / baseline run).
"""
from __future__ import annotations

import torch


def compute_magnitude_importance(gate_weight: torch.Tensor, up_weight: torch.Tensor) -> torch.Tensor:
    """Peak-to-peak magnitude (MAW / PPM) importance per neuron pair: this is
    optipfair's own default width-pruning criterion, reimplemented here so it
    can be blended with the concept-coverage term. gate_weight/up_weight are
    the [intermediate_size, hidden_size] weight matrices for one layer's
    gate_proj / up_proj.
    """
    gate_max_abs = torch.max(gate_weight, dim=1).values + torch.abs(torch.min(gate_weight, dim=1).values)
    up_max_abs = torch.max(up_weight, dim=1).values + torch.abs(torch.min(up_weight, dim=1).values)
    return gate_max_abs + up_max_abs


def blend_scores(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    concept_coverage: torch.Tensor | None,
    beta: float = 2.0,
) -> torch.Tensor:
    """Per-layer blended pruning score. ``concept_coverage`` is that layer's
    entry from combine_concepts()'s output (or None / beta=0 for plain
    magnitude-only baseline pruning). Higher score = more important = kept.
    """
    mag = compute_magnitude_importance(gate_weight, up_weight)
    if concept_coverage is None or beta == 0:
        return mag
    mag_n = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    cov = concept_coverage.to(mag.device).float()
    assert cov.shape[0] == mag.shape[0], (
        f"coverage/importance shape mismatch: {cov.shape} vs {mag.shape} "
        "-- did you pass a global intermediate_size instead of this layer's "
        "own width? (see model_utils.get_layer_mlp_widths for architectures "
        "with non-uniform per-layer MLP width, e.g. Gemma4)"
    )
    cov_n = (cov - cov.min()) / (cov.max() - cov.min() + 1e-8)
    return mag_n + beta * cov_n
