#!/usr/bin/env python3
"""declaude — local de-claude / humanizer tool (MLX).

Modes (default = declaude):
  declaude   re-write an AI-flavoured passage naturally (Lynote T5, full-length, preserves meaning)
  summarize  condense to the core (Humaneyes Pegasus)
  auto       a light classifier routes each input -> declaude or summarize (off by default)

The classifier is OFF unless --auto-mode is given.
"""
import argparse, sys, os, re, time

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")

# ---- light routing classifier (auto-mode) -------------------------------
AI_CLICHE = [r"\bit is important\b", r"\bmoreover\b", r"\bfurthermore\b", r"\bin conclusion\b",
             r"\bhowever\b", r"\bly[eo]ver?raging\b", r"\bly[eo]verage\b", r"\brobust\b",
             r"\bunderscore\b", r"\bnavigate[d]?\b", r"\bsignificant[ly]?\b", r"\bmerely\b",
             r"\bnotably\b", r"\bbusiness-critical\b", r"\btailored\b", r"\bseamless(?:ly)?\b"]
DENSITY_WORDS = [r"\bfirst\b", r"\bsecond\b", r"\bfinally\b", r"\bsteps?\b", r"\bstatistics?\b",
                 r"\bpercent\b", r"\bquarters?\b", r"\breports?\b", r"\bguidelines?\b", r"\breview\b"]
_re = lambda pats: [re.compile(p, re.I) for p in pats]
_AI = _re(AI_CLICHE); _DEN = _re(DENSITY_WORDS)

def classify(text: str):
    """Return (decision, confidence) where decision in {declaude, summarize}, confidence in (0,1]."""
    wc = len(text.split())
    ai = sum(1 for p in _AI if p.search(text))
    den = sum(1 for p in _DEN if p.search(text))
    # Signals:
    #  - AI clichés present -> the passage reads AI -> declaude it.
    #  - Very long & dense factual with few clichés -> summarize it.
    score = 0.0
    if ai > 0:
        score -= min(1.0, 0.25 + 0.15 * (ai - 1))      # declaude
    if wc > 220:
        score -= 0.20                                    # long -> lean summarize
    if den >= 3 and ai == 0:
        score += min(0.6, 0.2 * den)                      # dense factual, no clichés -> summarize
    if wc <= 40:
        score -= 0.4                                      # already tight -> declaude (light edit)
    decision = "summarize" if score > 0.3 else "declaude"
    return decision, round(abs(score), 2)


# ---- MLX generation ------------------------------------------------------
def make_gen(model_dir):
    from mlx_lm import load, generate as mlx_gen
    model, tokenizer = load(os.path.join(MODELS, model_dir), tokenizer_config={"trust_remote_code": True})
    def run(text, **kw):
        out = mlx_gen(model, tokenizer, prompt=text, verbose=False, **kw)
        return out
    return run

def main(argv=None):
    ap = argparse.ArgumentParser(prog="declaude", description="Local de-claude tool (MLX)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("text", nargs="?", help="input text")
    src.add_argument("-f", "--file", help="read input from file (else stdin)")
    ap.add_argument("--mode", choices=["declaude", "summarize", "auto"],
                    help="declare mode (default declaude)")
    ap.add_argument("--auto-mode", action="store_true",
                    help="route each input via the classifier (declaude vs summarize)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--models-dir", default=MODELS, help="override models dir")
    a = ap.parse_args(argv)

    text = a.text
    if not text and a.file:
        text = open(a.file).read().strip()
    if not text:
        text = sys.stdin.read().strip()
    if not text:
        ap.error("no input (pass text, -f FILE, or pipe stdin)")

    mode = a.mode
    auto = a.auto_mode or mode == "auto"
    if auto and mode == "auto":
        mode = None
    if mode is None:
        mode, conf = classify(text)
        if a.auto_mode:
            sys.stderr.write(f"[auto] classifier -> {mode} (conf {conf})\n")

    if mode == "summarize":
        gen = make_gen("humaneyes-mlx")
    else:
        gen = make_gen("lynote-mlx")

    t0 = time.time()
    out = gen(text, max_tokens=a.max_tokens)
    print(out.strip())
    sys.stderr.write(f"[{mode}] {time.time()-t0:.1f}s\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
