#!/usr/bin/env python3
"""
Driver for the native-MLX T5-family module (./t5.py).

Loads a model from its HF snapshot dir (discovered via glob) and greedily
generates from a text file.  transformers is used ONLY to build the tokenizer
at runtime -- the model itself runs entirely on MLX arrays.

Usage:
    python mlx_run.py --model lnote  --input /tmp/hs/sample_01.txt \
                      --output out_lynote_test.txt
    python mlx_run.py --model humaneyes --input /tmp/hs/sample_01.txt \
                      --output out_humaneyes_test.txt

Defaults to the Lynote/T5 path.
"""

import argparse
import glob
import os

import mlx.core as mx
from t5 import T5ConditionalMLX


# --------------------------------------------------------------------------- #
# per-model metadata
# --------------------------------------------------------------------------- #
MODELS = {
    "lynote": {
        "style": "t5",
        "snapshot_glob": (
            "/Users/terra/.cache/huggingface/hub/"
            "models--Lynote--humanize-text-model/snapshots/*/en"),
        "config": dict(d_model=512, d_ff=2048,
                       num_encoder_layers=6, num_decoder_layers=6,
                       num_heads=8, d_kv=64, vocab_size=32128,
                       positional="relative", rel_buckets=32,
                       rel_max_dist=128, scale_embedding=False,
                       layer_norm_eps=1e-6, add_final_layer_norm=True),
        "tokenizer": "T5Tokenizer",
        "max_enc": None,       # no hard position cap for T5
        "max_new": 220,
        "out": "out_lynote_test.txt",
    },
    "humaneyes": {
        "style": "pegasus",
        "snapshot_glob": (
            "/Users/terra/.cache/huggingface/hub/"
            "models--Eemansleepdeprived--Humaneyes/snapshots/*"),
        "config": dict(d_model=1024,
                       num_encoder_layers=16, num_decoder_layers=16,
                       num_heads=16, d_kv=64, vocab_size=96103,
                       positional="absolute", scale_embedding=True,
                       layer_norm_eps=1e-6, add_final_layer_norm=True,
                       attn_scale=1.0 / 8.0),
        "tokenizer": "PegasusTokenizer",
        # absolute learned peak positions are (60, 1024); cap both sides
        "max_enc": 55,
        "max_new": 42,
        "out": "out_humaneyes_test.txt",
    },
}


def resolve_snapshot(spec):
    hits = sorted(glob.glob(spec["snapshot_glob"]))
    env = spec.get("env", None)
    if env and os.path.isdir(env):
        hits.insert(0, env)
    if not hits:
        raise FileNotFoundError(f"no snapshot dirs matched {spec['snapshot_glob']}")
    # prefer a dir that directly holds the tokenizer/weights
    for h in hits:
        if os.path.isfile(os.path.join(h, "spiece.model")):
            return h
    return hits[0]


def make_tokenizer(spec, snapshot):
    from transformers import T5Tokenizer, PegasusTokenizer
    cls = {"T5Tokenizer": T5Tokenizer, "PegasusTokenizer": PegasusTokenizer}[
        spec["tokenizer"]]
    try:
        tok = cls.from_pretrained(snapshot)
    except Exception:
        tok = cls(snapshot + "/spiece.model")
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lynote",
                    choices=["lynote", "humaneyes"])
    ap.add_argument("--input", default="/tmp/hs/sample_01.txt")
    ap.add_argument("--output", default=None, help="defaults to per-model out_*.txt")
    ap.add_argument("--max-new", type=int, default=None)
    args = ap.parse_args()

    spec = MODELS[args.model]
    snapshot = resolve_snapshot(spec)
    print(f"[{args.model}] snapshot dir: {snapshot}", flush=True)

    cfg = dict(spec["config"])
    model = T5ConditionalMLX(
        snapshot, spec["style"],
        d_model=cfg["d_model"], d_ff=cfg.get("d_ff"),
        num_encoder_layers=cfg["num_encoder_layers"],
        num_decoder_layers=cfg["num_decoder_layers"],
        num_heads=cfg["num_heads"], d_kv=cfg["d_kv"],
        vocab_size=cfg["vocab_size"], positional=cfg["positional"],
        rel_buckets=cfg.get("rel_buckets", 32),
        rel_max_dist=cfg.get("rel_max_dist", 128),
        scale_embedding=cfg["scale_embedding"],
        layer_norm_eps=cfg["layer_norm_eps"],
        add_final_layer_norm=cfg["add_final_layer_norm"],
        attn_scale=cfg.get("attn_scale"),
    )
    print(f"[{args.model}] model built on MLX", flush=True)

    tok = make_tokenizer(spec, snapshot)
    raw = open(args.input, encoding="utf-8").read().strip()
    enc = tok(raw, return_tensors="np", max_length=512, truncation=True)
    input_ids = enc["input_ids"].tolist()[0]
    print(f"[{args.model}] input tokens: {len(input_ids)}", flush=True)

    if spec["max_enc"]:
        input_ids = input_ids[: spec["max_enc"]]
        print(f"[{args.model}] truncated encoder input to {len(input_ids)} tokens "
              f"(absolute-position cap)", flush=True)

    max_new = args.max_new or spec["max_new"]
    print(f"[{args.model}] generating up to {max_new} new tokens...", flush=True)
    out_ids = model.generate(input_ids, max_new_tokens=max_new, eos_id=1, start_id=0)
    text = tok.decode(out_ids, skip_special_tokens=True)
    print(f"[{args.model}] generated tokens (incl. eos if any): {len(out_ids)}", flush=True)

    output = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), spec["out"])
    with open(output, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"[{args.model}] wrote -> {output}", flush=True)
    print(f"[{args.model}] HEAD: {text[:250]}", flush=True)
    return text


if __name__ == "__main__":
    main()
