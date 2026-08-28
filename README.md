# declaude

**De-AI your writing. On your Mac. Offline. ~240 MB.**

You use an LLM to draft, then you paste the result somewhere it will be read by a human —
and it *screams* LLM. *"It is important to note," "Moreover," "In conclusion," "robust,"
"leveraging."* You know the sound. `declaude` is a tiny local tool that fixes exactly that:
it rewrites AI-flavoured text so it reads like a person wrote it, **on-device, with no cloud
and no API key, and no $80/month token bill.**

Under the hood it's two tiny open models run natively on Apple's **MLX** GPU runtime:

| mode | model | size | what it does |
|------|-------|------|--------------|
| **declaude** *(default)* | Lynote `humanize-text-model` (T5-small) | **~242 MB** | full-length, meaning-preserving rewrite that strips AI clichés — keeps *what* you said, fixes *how* it sounds |
| **summarize** | Humaneyes (Pegasus-base) | ~2.3 GB | condense a passage to its core point |
| **auto** | built-in classifier (off by default) | — | routes each input to declaude or summarize |

## Why *this* is better than "just prompt it harder"

We ran a **blind, 20-sample eval** — de-clauded output rated purely on "which reads most
human," with model labels shuffled per sample so you can't pattern-match:

- **Lynote wins 13 / 14** blind picks (one sample preferred the raw original).
- Humaneyes' summaries were consistently rejected as de-claude output (it truncates meaning).

The cost-to-run comparison (M4 Max, MLX GPU):

| | rewrite speed | peak memory |
|---|---|---|
| Lynote (declaude) | **~64 tok/s** | **242 MB** |
| Humaneyes (summarize) | ~19 tok/s | ~2.3 GB |

That speed and footprint is the whole point: it's a **tool you run in your editor / shell
pipeline**, not a heavy app.

## Install

```bash
# from a built release (macOS arm64 binary)
curl -L -o declaude https://github.com/terraboops/declaude/releases/latest/download/declaude-macos-arm64
chmod +x declaude && sudo mv declaude /usr/local/bin/
```

Or build from source (Python ≥3.11, MLX):

```bash
pip install mlx transformers-sentencepiece   # deps
git clone https://github.com/terraboops/declaude && cd declaude
./declaude --help
```

## Usage

```bash
echo "It is important to note that containerization offers a robust foundation..." | declaude            # declauded
cat draft.md | declaude --summarize                                                        # condensed
cat notes.txt | declaude --auto-mode                                                        # classifier picks
declaude -f draft.txt --max-tokens 320                                                    # from a file
```

Flags: `declaude` (default) · `--summarize` · `--auto-mode` (or `--mode auto`) · `-f/--file` ·
`--max-tokens` · `--models-dir`.

**Auto-mode is opt-in on purpose.** The classifier is a lightweight heuristic (AI-cliché +
density signals) tuned to be conservative — it's there to help, not to second-guess you. Run
`--auto-mode` when you want it, and usually you won't.

## Privacy

Zero telemetry, zero network, zero API key. Both models run on your device through MLX. Your
text — contracts, drafts, sensitive docs — never leaves your Mac. That's a feature, not a
compliance note.

## Technical note

`mlx-lm` does not (yet) ship the T5/Pegasus encoder-decoder architectures, so `declaude`
bundles a small native MLX implementation of the T5 family that loads the HuggingFace
weights directly. The models are MIT-licensed. This keeps the whole thing ~242 MB and fully
offline instead of requiring a ~7 GB instruction LLM to do the job.

## License

MIT. Models: Lynote `humanize-text-model` (MIT), Humaneyes (MIT).
