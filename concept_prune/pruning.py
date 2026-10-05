"""Structured MLP-width pruning driven by a per-layer scoring function.

Reimplements the neuron-pair removal mechanics directly (rather than calling
into optipfair) so a custom (blended) importance score can be used in place
of optipfair's built-in magnitude criterion -- optipfair's public API does
not expose a hook for supplying external per-neuron scores, so this mirrors
its GLU neuron-pair pruning logic (keep top-k gate/up/down neuron triples by
score) directly against the layer's own Linear modules.
"""
from __future__ import annotations

import copy

import torch
from torch import nn

from .model_utils import find_decoder_layers, get_layer_mlp_widths
from .scoring import blend_scores


def _prune_neuron_pairs(mlp, prune_percent: float, importance_scores: torch.Tensor, device, dtype):
    gate_weight = mlp.gate_proj.weight.data.float()
    up_weight = mlp.up_proj.weight.data.float()
    original_size = gate_weight.size(0)
    neurons_to_prune = min(int(prune_percent * original_size), original_size - 1)
    k = original_size - neurons_to_prune
    _, indices_to_keep = torch.topk(importance_scores, k, largest=True, sorted=True)
    indices_to_keep = indices_to_keep.sort().values

    new_gate = nn.Linear(mlp.gate_proj.in_features, k, bias=False).to(device).to(dtype)
    new_up = nn.Linear(mlp.up_proj.in_features, k, bias=False).to(device).to(dtype)
    new_down = nn.Linear(k, mlp.down_proj.out_features, bias=False).to(device).to(dtype)
    new_gate.weight.data = mlp.gate_proj.weight.data[indices_to_keep, :]
    new_up.weight.data = mlp.up_proj.weight.data[indices_to_keep, :]
    new_down.weight.data = mlp.down_proj.weight.data[:, indices_to_keep]
    return new_gate, new_up, new_down, k


def prune_model(
    model,
    prune_percent: float,
    coverage_by_layer: dict[int, torch.Tensor] | None = None,
    beta: float = 2.0,
    layers=None,
    in_place: bool = False,
):
    """Prune *model*'s MLP width by ``prune_percent`` (e.g. 0.4 = remove 40%
    of gate/up/down neuron triples per layer, keeping the rest).

    ``coverage_by_layer=None`` (or ``beta=0``) gives plain magnitude-only
    ("baseline") pruning. Otherwise each layer's score is
    ``blend_scores(gate, up, coverage_by_layer[idx], beta)`` -- see
    scoring.py. Returns the pruned model (a deepcopy unless
    ``in_place=True``); also updates ``model.config`` intermediate-size
    fields where present so downstream code that reads them stays correct.
    """
    target = model if in_place else copy.deepcopy(model)
    if layers is None:
        layers = find_decoder_layers(target)
    device = next(target.parameters()).device
    dtype = target.dtype

    new_sizes = []
    for idx, layer in enumerate(layers):
        mlp = layer.mlp
        gate_weight = mlp.gate_proj.weight.data.float()
        up_weight = mlp.up_proj.weight.data.float()
        coverage = coverage_by_layer[idx] if coverage_by_layer is not None else None
        scores = blend_scores(gate_weight, up_weight, coverage, beta)
        new_gate, new_up, new_down, k = _prune_neuron_pairs(mlp, prune_percent, scores, device, dtype)
        mlp.gate_proj, mlp.up_proj, mlp.down_proj = new_gate, new_up, new_down
        new_sizes.append(k)

    # Best-effort config bookkeeping. Architectures with non-uniform
    # per-layer width (e.g. Gemma4) don't have one meaningful global
    # intermediate_size any more after pruning -- only set it when uniform.
    if len(set(new_sizes)) == 1 and hasattr(target.config, "intermediate_size"):
        target.config.intermediate_size = new_sizes[0]

    return target


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
