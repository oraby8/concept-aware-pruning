#!/usr/bin/env python3
"""Build a concept_prune pairs_file (JSONL) for a "task quality" concept --
here, translation quality -- from a source/target dataset.

This is the pattern validated on google/gemma-4-E2B-it's real Arabic
translation quality (see README "Concept = task quality" section): the
"concept" isn't a narrow linguistic probe, it's "produces the correct
continuation for this translation prompt". clean = correct prompt+target
pair, corrupted = the same target paired with a DIFFERENT source, so the
label token is the correct target's first token and the corrupted prompt
forces the model toward the wrong source's context.

Usage:
    python build_translation_pairs.py \\
        --model google/gemma-4-E2B-it \\
        --data en-ar_SA.jsonl --source-key source --target-key target \\
        --system-prompt "You are a professional translator. Translate the user text from English to Arabic (Modern Standard Arabic). Output only the translation, no extra text." \\
        --n-pairs 50 --concept-name translation_ar --out translation_pairs.jsonl

The model's chat template is used to build the prompt exactly as your
production client would (pass --system-prompt matching your own client, or
omit it for a plain instruction-less prompt) -- using a different prompt
format than production is a real gotcha (see README "Common pitfalls"):
mismatched formatting during attribution silently produces a garbage
concept signal without erroring.
"""
from __future__ import annotations

import argparse
import json
import random

from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Tokenizer to use for chat template + label token resolution")
    ap.add_argument("--data", required=True, help="JSONL with source/target rows")
    ap.add_argument("--source-key", default="source")
    ap.add_argument("--target-key", default="target")
    ap.add_argument("--system-prompt", default=None)
    ap.add_argument("--n-pairs", type=int, default=50)
    ap.add_argument("--concept-name", default="task_quality")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)

    rows = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    def make_prompt(source_text):
        msgs = []
        if args.system_prompt:
            msgs.append({"role": "system", "content": args.system_prompt})
        msgs.append({"role": "user", "content": source_text})
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    random.seed(args.seed)
    idxs = list(range(len(rows)))
    n = min(args.n_pairs, len(idxs))
    picked = random.sample(idxs, n)

    out_rows = []
    for i in picked:
        j = random.choice([x for x in idxs if x != i])
        clean_prompt = make_prompt(rows[i][args.source_key])
        corrupted_prompt = make_prompt(rows[j][args.source_key])
        label_id = tok(rows[i][args.target_key], add_special_tokens=False)["input_ids"][0]
        out_rows.append({
            "concept": args.concept_name,
            "clean": clean_prompt,
            "corrupted": corrupted_prompt,
            "label": label_id,  # pre-resolved int token id, not a string
        })

    with open(args.out, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(out_rows)} pairs to {args.out}")


if __name__ == "__main__":
    main()
