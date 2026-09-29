#!/usr/bin/env python3
"""declaude — local de-claude / humanizer tool (native MLX).

Modes (default = declaude):
  declaude   re-write AI-flavoured text naturally  (Lynote T5-small, full-length)
  summarize  condense to the core                  (Humaneyes Pegasus-base)
  auto       a light classifier routes each input -> declaude or summarize

The classifier is OFF unless --auto-mode (or --mode auto) is given.
Runs entirely on-device via MLX. No cloud, no API key.
"""
import argparse, re, sys, time

# MLX-driven model + loader helpers (transformer-free inference; tokenizer only)
from t5 import T5ConditionalMLX
from mlx_run import MODELS, resolve_snapshot, make_tokenizer


def build(spec):
    cfg = dict(spec["config"])
    m = T5ConditionalMLX(
        resolve_snapshot(spec), spec["style"],
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
    return m, make_tokenizer(spec, resolve_snapshot(spec))


def generate(m, tok, text, spec, max_new=None):
    enc = tok(text, return_tensors="np", max_length=512, truncation=True)
    ids = enc["input_ids"].tolist()[0]
    if spec["max_enc"]:
        ids = ids[: spec["max_enc"]]
    out = m.generate(ids, max_new_tokens=max_new or spec["max_new"],
                     eos_id=1, start_id=0)
    return tok.decode(out, skip_special_tokens=True).strip()


# ---- light routing classifier (auto-mode, off by default) ----------------
AI_CLICHE = [r"\bit is important\b", r"\bmoreover\b", r"\bfurthermore\b", r"\bin conclusion\b",
             r"\bhowever\b", r"\bleveraging\b", r"\bleverage\b", r"\brobust\b",
             r"\bunderscore\b", r"\bnavigate[ds]?\b", r"\bsignificant[ly]?\b", r"\bnotably\b",
             r"\btailored\b", r"\bseamless(?:ly)?\b",
             # --- transition-crutch tells (Terra feedback 2026-09-04, "Scaling Out") ---
             # formulaic bridges that read as AI even when the lexical vocab is clean
             r"\bI keep bumping into\b",
             r"\bLet me be clear about (?:the claim|this|something)\b",
             r"\bthat's kind of the point\b",
             r"\bthat's the whole trade\b",
             r"\bthe telling bit\b",
             r"\bthe quiet lesson\b",
             r"\bthe lesson (?:keeps landing|lands on the same spot)\b",
             r"\bthe shape holds\b",
             r"\bworth repeating\b",
             r"\bconnecting the dots\b",
             r"\bhere's the part I actually\b",
             r"\bthe part that matters here\b"]
DENSITY = [r"\bfirst\b", r"\bsecond\b", r"\bfinally\b", r"\bsteps?\b", r"\bpercent\b",
           r"\bquarters?\b", r"\breports?\b", r"\bguidelines?\b", r"\breview\b", r"\bmetrics?\b"]
_AI = [re.compile(p, re.I) for p in AI_CLICHE]
_DEN = [re.compile(p, re.I) for p in DENSITY]


def classify(text: str):
    """Return (decision, confidence), decision in {declaude, summarize}, conf in (0,1]."""
    wc = len(text.split())
    ai = sum(1 for p in _AI if p.search(text))
    den = sum(1 for p in _DEN if p.search(text))
    score = 0.0
    if ai > 0:
        score -= min(1.0, 0.25 + 0.15 * (ai - 1))      # clichés -> declaude it
    if wc > 220:
        score -= 0.20                                    # long -> lean summarize
    if den >= 3 and ai == 0:
        score += min(0.6, 0.2 * den)                     # dense, no clichés -> summarize
    if wc <= 40:
        score -= 0.4                                     # already tight -> light declaude
    decision = "summarize" if score > 0.3 else "declaude"
    return decision, round(abs(score), 2)


def fetch_models():
    """Download Lynote (en) + Humaneyes weights to the local HF cache."""
    from huggingface_hub import snapshot_download
    repos = [("Lynote/humanize-text-model", "en,zh"),
             ("Eemansleepdeprived/Humaneyes", None)]
    for repo, subs in repos:
        snap = snapshot_download(repo_id=repo, allow_patterns=["*"])
        print(f"[declaude] cached {repo} -> {snap}", flush=True)
        _ = subs
    print("[declaude] models ready", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="declaude",
                                 description="De-AI your text locally (MLX).")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("text", nargs="?", help="input text")
    src.add_argument("-f", "--file", help="read input from file")
    ap.add_argument("--mode", choices=["declaude", "summarize", "auto"],
                    help="mode (default declaude)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--summarize", action="store_true", help="shorthand for --mode summarize")
    g.add_argument("--declaude", action="store_true", help="shorthand for --mode declaude")
    g.add_argument("--auto-mode", action="store_true",
                   help="route input via the classifier (declaude vs summarize)")
    ap.add_argument("--max-new", type=int, default=None)
    ap.add_argument("--fetch-models", action="store_true",
                    help="download model weights to the local cache, then exit")
    a = ap.parse_args(argv)
    if a.fetch_models:
        fetch_models()
        return 0

    text = a.text
    if not text and a.file:
        text = open(a.file, encoding="utf-8").read().strip()
    if not text:
        text = sys.stdin.read().strip()
    if not text:
        ap.error("no input (pass text, -f FILE, or pipe stdin)")

    mode = a.mode or ("summarize" if a.summarize else
                      "declaude" if a.declaude else None)
    if a.auto_mode or a.mode == "auto":
        mode, conf = classify(text)
        if a.auto_mode or a.mode == "auto":
            sys.stderr.write(f"[declaude] auto -> {mode} (conf {conf})\n")
    mode = mode or "declaude"

    key = "lynote" if mode == "declaude" else "humaneyes"
    m, tok = build(MODELS[key])
    t0 = time.time()
    out = generate(m, tok, text, MODELS[key], a.max_new)
    sys.stderr.write(f"[declaude] {mode} via {key}  {time.time()-t0:.1f}s\n")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
