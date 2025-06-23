# create_dataset.py
# Author: you 🎈  Date: 2025-06-23
# Python 3.10+

import os, json, random, itertools, multiprocessing as mp
from pathlib import Path
from functools import partial

from datasets import load_dataset, interleave_datasets, Dataset
from transformers import AutoTokenizer
from datasketch import MinHash, MinHashLSH

######### CONFIG  #############################################################
TARGET_TOKENS = 30_000_000_000          # 30 B
SHARD_SIZE    = 2048 * 1024             # 2 048 sequences → ~1 Mi tokens / shard
TOKENIZER_ID  = "EleutherAI/gpt-neox-20b"  # any SP-compatible 32 k model
OUTPUT_DIR    = Path("pretrain_dataset")   # ⬅️ change as needed

# per-source budgets (must sum ≥ TARGET_TOKENS)
# ── 30 B-token maths-centric mixture ───────────────────────────────────────────
# 80 % maths / reasoning • 20 % clean general text
CORPORA = [
    # Dense informal maths (Apache-2.0, 14.7 B tok total)
    dict(hf_id="open-web-math/open-web-math",       split="train",
         cap=6_000_000_000),     # 6 B :contentReference[oaicite:0]{index=0}

    # Formal proofs & theorem libs (Apache-2.0, 8.3 B tok)
    dict(hf_id="hoskinson-center/proof-pile",       split="train",
         cap=2_000_000_000),     # 2 B :contentReference[oaicite:1]{index=1}

    # arXiv maths slice of Dolma (ODC-BY)
    dict(hf_id="allenai/dolma",                     split="train",
         cap=3_000_000_000,
         filter=lambda x: x["meta"].get("category") == "math"),  # 3 B :contentReference[oaicite:2]{index=2}

    # Permissive math-centric code (REMOVED - gated dataset)
    # dict(hf_id="bigcode/the-stack-v2",              split="data",
    #      cap=3_000_000_000,
    #      filter=lambda x: x.get("lang") in {"python", "julia", "rust"}),  # 3 B :contentReference[oaicite:3]{index=3}

    # Human-written CoT & PoT (MIT)
    dict(hf_id="TIGER-Lab/MathInstruct",            split="train",
         cap=2_000_000_000),     # 2 B :contentReference[oaicite:4]{index=4}

    # NVIDIA OpenMathReasoning (MIT; 3.2 M CoT traces)
    dict(hf_id="nvidia/OpenMathReasoning",          split="cot",
         cap=3_000_000_000),     # 3 B :contentReference[oaicite:5]{index=5}

    # Generic multi-domain CoT collection (CC-BY-4.0) — ↓ to 1 B
    dict(hf_id="pharaouk/CoT-Collection",           split="train",
         cap=1_000_000_000),     # 1 B :contentReference[oaicite:6]{index=6}

    # ★ NEW ★ DeepSeek-R1 distilled reasoning (CC-BY-NC-4.0)
    dict(hf_id="a-m-team/AM-DeepSeek-R1-Distilled-1.4M", split="train",
         cap=1_000_000_000),     # ~1 B (oversample allowed) :contentReference[oaicite:7]{index=7}

    # Dialogue / alignment (UltraChat 200 k, MIT)
    dict(hf_id="HuggingFaceH4/ultrachat_200k",      split="train_sft",
         cap=2_000_000_000),     # 2 B :contentReference[oaicite:8]{index=8}

    # High-quality web fluency buffer (FineWeb HQ, ODC-BY)
    dict(hf_id="HuggingFaceFW/fineweb",             split="hq",
         cap=7_000_000_000,
         filter=lambda x: x.get("language") == "en"),  # 7 B :contentReference[oaicite:9]{index=9}
]
# Total: 30 B tokens


###############################################################################

def token_counter(example, tok):
    return {"n_tokens": len(tok.encode(example["text"]))}

def yield_examples(dataset, tok, cap):
    total = 0
    for ex in dataset:
        text = ex["text"] if "text" in ex else ex.get("content", "")
        n_tokens = len(tok.encode(text))
        if total + n_tokens > cap:
            break
        yield dict(text=text, n_tokens=n_tokens)
        total += n_tokens

def dedup_iterator(records, seed=0, threshold=0.9):
    """MinHash near-dedup at ~90 % Jaccard similarity."""
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    for rec in records:
        mh = MinHash(num_perm=128)
        for token in rec["text"].split():
            mh.update(token.encode("utf-8"))
        if lsh.query(mh):
            continue              # near-duplicate
        lsh.insert(str(id(rec)), mh)
        yield rec

def shard_and_write(records, shard_tokens, out_dir, tokenizer):
    buf, buf_tokens, shard_id = [], 0, 0
    out_dir.mkdir(exist_ok=True, parents=True)
    for rec in records:
        buf.append(rec["text"])
        buf_tokens += rec["n_tokens"]
        if buf_tokens >= shard_tokens:
            shard_path = out_dir / f"shard_{shard_id:05d}.jsonl"
            with shard_path.open("w") as f:
                for line in buf:
                    f.write(json.dumps({"text": line}) + "\n")
            shard_id += 1
            buf, buf_tokens = [], 0

def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True)
    streams = []
    for spec in CORPORA:
        ds = load_dataset(spec["hf_id"], split=spec["split"], streaming=True, trust_remote_code=True)
        if spec.get("filter"):
            ds = ds.filter(spec["filter"])
        streams.append(
            ds.map(partial(token_counter, tok=tokenizer),
                   remove_columns=ds.features)
              .take(spec["cap"])        # hard-cap the iterator
        )

    merged = interleave_datasets(streams, probabilities=None, seed=42)
    deduped = dedup_iterator(merged)

    shard_and_write(deduped, SHARD_SIZE, OUTPUT_DIR, tokenizer)

if __name__ == "__main__":
    main()
