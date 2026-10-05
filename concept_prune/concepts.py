"""Concept definitions: a Concept is a name plus a set of clean/corrupted
contrastive text pairs, each with the token that should follow the clean
prompt (the attribution target).

A "concept" is deliberately generic: it can be a narrow linguistic
phenomenon (subject-verb agreement, indirect-object identification), a
factual-recall probe, or something task-shaped like "produces a correct
translation" (source text as prompt, correct vs. corrupted reference
continuation) -- see examples/concepts_translation.yaml for that pattern.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field


@dataclass
class ConceptPair:
    clean_prompt: str
    corrupted_prompt: str
    label: str | int  # label token text (tokenized in isolation) or a pre-resolved token id


@dataclass
class Concept:
    name: str
    pairs: list[ConceptPair] = field(default_factory=list)

    def __len__(self):
        return len(self.pairs)


def _resolve_and_filter(tokenizer, pairs: list[ConceptPair], n_target: int):
    """Tokenize each pair, keep only those where clean/corrupted have equal
    token length (required for the attribution hooks, which patch one
    activation tensor into another of the same shape) and the label resolves
    to a single token id. Returns a list of
    (clean_ids, corrupted_ids, label_id) tensors, capped at n_target.
    """
    out = []
    for p in pairs:
        clean_ids = tokenizer(p.clean_prompt, return_tensors="pt").input_ids
        corr_ids = tokenizer(p.corrupted_prompt, return_tensors="pt").input_ids
        if clean_ids.shape[1] != corr_ids.shape[1]:
            continue
        if isinstance(p.label, str):
            label_ids = tokenizer(p.label, add_special_tokens=False).input_ids
            if len(label_ids) == 0:
                continue
            label_id = label_ids[0]
        else:
            label_id = p.label
        out.append((clean_ids, corr_ids, label_id))
        if len(out) >= n_target:
            break
    return out


def concepts_from_templates(tokenizer, generators: dict, n_per_concept: int = 60, oversample: int = 3):
    """Build resolved Concept probe sets from a dict of
    ``{concept_name: callable() -> list[ConceptPair]}`` template generators.

    Each generator is called once and should itself produce randomized
    pairs (e.g. via ``random.sample`` over a name/place/item pool) -- call it
    with a fresh RNG seed beforehand if you need reproducibility. Generators
    are over-sampled (default 3x n_per_concept) since some candidate pairs
    get filtered out by tokenization-length mismatches.
    """
    resolved = {}
    for name, gen_fn in generators.items():
        candidates = gen_fn(n_per_concept * oversample)
        resolved[name] = _resolve_and_filter(tokenizer, candidates, n_per_concept)
        print(f"[concepts] {name}: {len(resolved[name])} valid pairs (of {len(candidates)} candidates)")
    return resolved


def concepts_from_pairs_file(tokenizer, path: str, n_per_concept: int | None = None):
    """Load concept pairs from a JSONL file with rows
    ``{"concept": str, "clean": str, "corrupted": str, "label": str}``.
    Groups by "concept" and resolves each group the same way as
    concepts_from_templates.
    """
    by_concept: dict[str, list[ConceptPair]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_concept.setdefault(row["concept"], []).append(
                ConceptPair(row["clean"], row["corrupted"], row["label"])
            )
    resolved = {}
    for name, pairs in by_concept.items():
        target = n_per_concept or len(pairs)
        resolved[name] = _resolve_and_filter(tokenizer, pairs, target)
        print(f"[concepts] {name}: {len(resolved[name])} valid pairs (of {len(pairs)} candidates)")
    return resolved


def concepts_from_pairs(tokenizer, name: str, pairs: list[ConceptPair], n_per_concept: int | None = None):
    """Resolve a single in-memory list of ConceptPair into one concept's
    probe set. Useful for the "concept = task quality" pattern (e.g.
    translation), where pairs are built from a source dataset + a
    corruption function rather than a template.
    """
    target = n_per_concept or len(pairs)
    resolved = _resolve_and_filter(tokenizer, pairs, target)
    print(f"[concepts] {name}: {len(resolved)} valid pairs (of {len(pairs)} candidates)")
    return {name: resolved}


def make_corrupted_pairs_from_dataset(rows, source_key, target_key, corrupt_fn, n_pairs, seed=0):
    """Generic helper for the "task-quality" concept pattern: given a list of
    dataset rows (e.g. translation source/target pairs), build ConceptPair
    objects where the clean prompt is a formatted prompt over
    ``row[source_key]`` and the corrupted prompt swaps in a different row's
    source text (or apply ``corrupt_fn`` to build the corrupted continuation
    directly -- see examples/concepts_translation.yaml for the reference
    pattern used for the Arabic-translation concept). ``corrupt_fn`` receives
    ``(prompt_fn, row, other_row)`` and returns a ConceptPair.
    """
    rng = random.Random(seed)
    idxs = list(range(len(rows)))
    out = []
    for i in rng.sample(idxs, min(n_pairs, len(idxs))):
        j = rng.choice([x for x in idxs if x != i])
        out.append(corrupt_fn(rows[i], rows[j]))
    return out
