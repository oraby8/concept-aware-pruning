"""Generic model loading and decoder-layer discovery.

Plain HF causal LMs expose their transformer blocks at ``model.model.layers``.
Wrapped/multimodal models (e.g. Gemma4's image-text-to-text wrapper) nest the
actual text decoder one level deeper, e.g. ``model.model.language_model.layers``.
find_decoder_layers() tries the common cases first, then falls back to a
shallow recursive search for any ``.layers`` attribute containing modules that
look like transformer blocks (i.e. have an ``.mlp`` with gate/up/down
projections), so the rest of the package never has to special-case a
particular architecture.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _looks_like_transformer_layers(candidate) -> bool:
    try:
        first = candidate[0]
    except (TypeError, IndexError, KeyError):
        return False
    mlp = getattr(first, "mlp", None)
    if mlp is None:
        return False
    return all(hasattr(mlp, name) for name in ("gate_proj", "up_proj", "down_proj"))


def find_decoder_layers(model):
    """Locate the ``nn.ModuleList`` of transformer decoder blocks for *model*.

    Handles plain causal LMs (``model.model.layers``) and wrapped/multimodal
    models where the text decoder is nested further in (e.g.
    ``model.model.language_model.layers`` for Gemma4). Falls back to a
    shallow breadth-first search over the module tree for any ``.layers``
    attribute that looks like a list of transformer blocks (has an ``.mlp``
    with gate/up/down projections).

    Raises ``AttributeError`` if no matching layers are found.
    """
    candidates = []
    root = getattr(model, "model", model)
    candidates.append(getattr(root, "layers", None))
    lang_model = getattr(root, "language_model", None)
    if lang_model is not None:
        candidates.append(getattr(lang_model, "layers", None))

    for c in candidates:
        if c is not None and _looks_like_transformer_layers(c):
            return c

    # Shallow recursive fallback: BFS over named_modules, first
    # ``.layers``-bearing module whose contents look like transformer blocks.
    for _name, module in model.named_modules():
        layers = getattr(module, "layers", None)
        if layers is not None and _looks_like_transformer_layers(layers):
            return layers

    raise AttributeError(
        "Could not locate transformer decoder layers on this model. "
        "Expected model.model.layers, model.model.language_model.layers, "
        "or some module.layers whose blocks expose mlp.gate_proj/up_proj/down_proj."
    )


def get_layer_mlp_widths(layers) -> list[int]:
    """Per-layer MLP intermediate width (``down_proj.in_features``).

    Returned as a list, one entry per layer, since some architectures
    (e.g. Gemma4) use a non-uniform width across layers ("double-wide" MLPs
    on some layers only) -- callers must not assume a single global
    intermediate_size.
    """
    return [layer.mlp.down_proj.in_features for layer in layers]


def load_model(model_name: str, dtype=torch.bfloat16, device_map="auto", trust_remote_code=False):
    """Load a tokenizer + causal LM, trying the standard causal-LM class first
    and falling back to AutoModelForImageTextToText for multimodal-wrapped
    text models (e.g. Gemma4), which still work fine for text-only pruning
    once you reach the inner decoder via find_decoder_layers().
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map=device_map, trust_remote_code=trust_remote_code
        )
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype=dtype, device_map=device_map, trust_remote_code=trust_remote_code
        )
    model.eval()
    for attr in ("temperature", "top_p", "top_k"):
        if hasattr(model.generation_config, attr):
            setattr(model.generation_config, attr, None)
    return model, tokenizer


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
