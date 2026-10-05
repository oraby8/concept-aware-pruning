"""Per-neuron concept attribution via integrated-gradients-style patching on
each MLP layer's down_proj input activation.

This is a targeted, node-level simplification of EAP-IG (Edge Attribution
Patching with Integrated Gradients), restricted to MLP intermediate
("down_proj input") neurons -- the exact granularity width pruning operates
at. It is NOT the full edge-level circuit method (which needs
attention-edge attribution over a TransformerLens-style graph); investigated
and confirmed unnecessary for width pruning specifically, since standard
EAP-IG tooling (e.g. hannamw/EAP-IG) represents each MLP as a single opaque
node over d_model, not d_mlp, and so cannot produce scores at this
granularity at all. See the project README for the full writeup.

Width-aware: each layer's activation/coverage tensor is sized from that
layer's own down_proj.in_features at attribution time, not a single global
intermediate_size -- some architectures (e.g. Gemma4) use a non-uniform
width across layers ("double-wide" MLPs on some layers only).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .model_utils import find_decoder_layers, get_layer_mlp_widths


class _AttributionHooks:
    """Registers forward-pre-hooks on every layer's mlp.down_proj that either
    capture the incoming activation (default) or replace it with a patched
    value (when one has been set for that layer index), then removes the
    hooks on exit -- use as a context manager.
    """

    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self._capture_cache = {}
        self._patch_values = {}
        self._handles = []

    def _make_pre_hook(self, layer_idx):
        def hook(module, inputs):
            x = inputs[0]
            if layer_idx in self._patch_values:
                return (self._patch_values[layer_idx],)
            self._capture_cache[layer_idx] = x.detach()
            return None

        return hook

    def __enter__(self):
        for idx, layer in enumerate(self.layers):
            h = layer.mlp.down_proj.register_forward_pre_hook(self._make_pre_hook(idx))
            self._handles.append(h)
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def capture_activations(self, input_ids, attn_mask):
        self._capture_cache = {}
        self._patch_values = {}
        with torch.no_grad():
            self.model(input_ids=input_ids, attention_mask=attn_mask)
        return {k: v.clone() for k, v in self._capture_cache.items()}

    def patched_forward_backward(self, input_ids, attn_mask, target_id, interp_values):
        for v in interp_values.values():
            v.requires_grad_(True)
        self._patch_values = interp_values
        self.model.zero_grad(set_to_none=True)
        out = self.model(input_ids=input_ids, attention_mask=attn_mask)
        logits = out.logits[:, -1, :]
        loss = F.cross_entropy(logits.float(), torch.tensor([target_id], device=logits.device))
        loss.backward()
        grads = {
            k: (v.grad.detach().clone() if v.grad is not None else torch.zeros_like(v))
            for k, v in self._patch_values.items()
        }
        self._patch_values = {}
        return grads


def _attribute_one_pair(hooks, clean_ids, corr_ids, label_id, m_steps, device, model_dtype):
    clean_ids = clean_ids.to(device)
    corr_ids = corr_ids.to(device)
    attn_mask = torch.ones_like(corr_ids)

    corrupted_acts = hooks.capture_activations(corr_ids, attn_mask)
    clean_acts = hooks.capture_activations(clean_ids, attn_mask)

    grad_accum = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in corrupted_acts.items()}
    for step in range(1, m_steps + 1):
        alpha = step / m_steps
        interp = {}
        for k in corrupted_acts:
            c = corrupted_acts[k].float()
            cl = clean_acts[k].float()
            iv = (c + alpha * (cl - c)).to(model_dtype).clone()
            interp[k] = iv
        grads = hooks.patched_forward_backward(corr_ids, attn_mask, label_id, interp)
        for k in grad_accum:
            grad_accum[k] += grads[k].float()

    attributions = {}
    for k in grad_accum:
        avg_grad = grad_accum[k] / m_steps
        diff = clean_acts[k].float() - corrupted_acts[k].float()
        attr = (avg_grad * diff).abs().sum(dim=1).squeeze(0)  # sum over seq -> [layer_width]
        attributions[k] = attr.detach().cpu()
    return attributions


def compute_concept_importance(model, resolved_pairs, m_steps: int = 4, layers=None, verbose=True):
    """Compute per-layer, per-neuron importance for one concept.

    ``resolved_pairs`` is a list of (clean_ids, corrupted_ids, label_id)
    tensors, as produced by concept_prune.concepts._resolve_and_filter (via
    concepts_from_templates / concepts_from_pairs / concepts_from_pairs_file).

    Returns ``{layer_idx: tensor[layer_width]}`` -- width-aware, so layers
    with different MLP widths (e.g. Gemma4's double-wide layers) each get a
    correctly-sized tensor.
    """
    if layers is None:
        layers = find_decoder_layers(model)
    layer_widths = get_layer_mlp_widths(layers)
    device = next(model.parameters()).device
    model_dtype = model.dtype

    acc = {idx: torch.zeros(w) for idx, w in enumerate(layer_widths)}
    n_ok = 0
    with _AttributionHooks(model, layers) as hooks:
        for clean_ids, corr_ids, label_id in resolved_pairs:
            try:
                attributions = _attribute_one_pair(hooks, clean_ids, corr_ids, label_id, m_steps, device, model_dtype)
            except Exception as e:  # noqa: BLE001 - surface and skip, don't abort the whole run
                if verbose:
                    print(f"[attribution] skipping pair due to: {e!r}")
                continue
            for idx in acc:
                acc[idx] += attributions[idx]
            n_ok += 1

    for idx in acc:
        acc[idx] /= max(n_ok, 1)
    if verbose:
        print(f"[attribution] {n_ok}/{len(resolved_pairs)} pairs used")
    return acc


def combine_concepts(concept_importances: dict[str, dict[int, torch.Tensor]], aggregate: str = "max"):
    """Combine several concepts' per-layer importance into one coverage
    tensor per layer (default: max across concepts -- protect a neuron if
    it's important to ANY included concept). ``aggregate`` can be "max" or
    "sum".
    """
    concept_names = list(concept_importances)
    if not concept_names:
        raise ValueError("combine_concepts() called with no concepts")
    layer_idxs = concept_importances[concept_names[0]].keys()
    coverage = {}
    for idx in layer_idxs:
        stacked = torch.stack([concept_importances[c][idx] for c in concept_names], dim=0)
        coverage[idx] = stacked.max(dim=0).values if aggregate == "max" else stacked.sum(dim=0)
    return coverage
