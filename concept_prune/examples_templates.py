"""Built-in example concept-pair generators, referenced from a concepts.yaml
via ``template: builtin:<name>``. These are the exact probes validated
throughout this project's research phase (see README) -- IOI and SVA were
found to be the efficiency sweet spot (2 concepts) at 40% compression on
Llama-3.2-1B; factual recall is included as a third example since it was
informative on other models/ratios. ("greater-than", a fourth probe used
during research, is deliberately NOT included here -- it was found to
saturate at ~100% accuracy for every model/method tested, i.e. it doesn't
discriminate at the compression ratios this project tested, so it isn't a
useful default. Nothing stops you from writing your own template with the
same generator signature: ``def generate_x(n_candidates) -> list[ConceptPair]``.)
"""
from __future__ import annotations

import random

from .concepts import ConceptPair

_NAMES = ["John", "Mary", "James", "Linda", "Robert", "Patricia", "Michael", "Jennifer",
          "William", "Elizabeth", "David", "Susan", "Richard", "Jessica", "Thomas", "Karen"]
_PLACES = ["store", "park", "market", "library", "cafe", "station", "garden", "museum"]
_ITEMS = ["drink", "book", "ball", "letter", "gift", "key", "flower", "coin"]

_NOUN_PAIRS = [("keys", "key"), ("dogs", "dog"), ("books", "book"), ("cars", "car"),
               ("birds", "bird"), ("students", "student"), ("teachers", "teacher"), ("phones", "phone")]
_CONTAINERS = ["cabinet", "house", "table", "shelf", "room", "box", "garden", "office"]

_COUNTRY_CAPITAL = [("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"),
                     ("Japan", "Tokyo"), ("Egypt", "Cairo"), ("Canada", "Ottawa"), ("Russia", "Moscow"),
                     ("China", "Beijing"), ("Portugal", "Lisbon"), ("Poland", "Warsaw"), ("Greece", "Athens")]


def generate_ioi(n_candidates: int) -> list[ConceptPair]:
    """Indirect Object Identification: "A and B went to the X. A gave a Y to"
    -> B. Corrupted swaps which name is grammatically the giver, changing
    which name should complete the sentence.
    """
    out = []
    for _ in range(n_candidates):
        a, b = random.sample(_NAMES, 2)
        place = random.choice(_PLACES)
        item = random.choice(_ITEMS)
        clean = f"{a} and {b} went to the {place}. {a} gave a {item} to"
        corrupted = f"{a} and {b} went to the {place}. {b} gave a {item} to"
        out.append(ConceptPair(clean, corrupted, " " + b))
    return out


def generate_sva(n_candidates: int) -> list[ConceptPair]:
    """Subject-verb agreement: "The keys in the cabinet" -> are (plural
    subject). Corrupted uses the singular noun, which should take "is"
    instead -- same label token either way, so this probes whether the
    model still tracks the (now-singular) subject's number correctly.
    """
    out = []
    for _ in range(n_candidates):
        plural, singular = random.choice(_NOUN_PAIRS)
        container = random.choice(_CONTAINERS)
        clean = f"The {plural} in the {container}"
        corrupted = f"The {singular} in the {container}"
        out.append(ConceptPair(clean, corrupted, " are"))
    return out


def generate_factual(n_candidates: int) -> list[ConceptPair]:
    """Factual recall: "The capital of France is" -> Paris. Corrupted asks
    about a different country, so the model must retrieve the fact tied to
    the specific entity in THIS prompt, not just pattern-match the template.
    """
    out = []
    for _ in range(n_candidates):
        c1, cap1 = random.choice(_COUNTRY_CAPITAL)
        c2, _cap2 = random.choice(_COUNTRY_CAPITAL)
        if c2 == c1:
            continue
        clean = f"The capital of {c1} is"
        corrupted = f"The capital of {c2} is"
        out.append(ConceptPair(clean, corrupted, " " + cap1))
    return out
