"""concept_prune: concept-aware structured (width) pruning for HF causal LMs.

Protects MLP intermediate neurons important to user-defined "concepts"
(probed via clean/corrupted contrastive text pairs) while pruning the rest
by magnitude, instead of pure magnitude-only pruning.
"""
from .concepts import Concept, concepts_from_templates, concepts_from_pairs
from .model_utils import find_decoder_layers, load_model
from .attribution import compute_concept_importance
from .scoring import blend_scores
from .pruning import prune_model

__all__ = [
    "Concept",
    "concepts_from_templates",
    "concepts_from_pairs",
    "find_decoder_layers",
    "load_model",
    "compute_concept_importance",
    "blend_scores",
    "prune_model",
]
