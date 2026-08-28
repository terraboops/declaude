#!/usr/bin/env python3
"""Convert Lynote (T5) + Humaneyes (Pegasus) HF checkpoints to MLX weights."""
import os, glob, sys
from mlx_lm import convert

HSNAP='/Users/terra/.cache/huggingface/hub'
DEST='/Users/terra/Developer/declaude/models'
os.makedirs(DEST, exist_ok=True)

ly = glob.glob(f'{HSNAP}/models--Lynote--humanize-text-model/snapshots/*/en')[0]
he = glob.glob(f'{HSNAP}/models--Eemansleepdeprived--Humaneyes/snapshots/*')[0]
print('Lynote(en):', ly, flush=True)
print('Humaneyes:', he, flush=True)

# Lynote t5-small -> mlx
try:
    convert(hf_path=ly, mlx_path=f'{DEST}/lynote-mlx', quantize=False)
    print('LYNOTE converted', flush=True)
except Exception as e:
    print('LYNOTE FAIL', type(e).__name__, e, flush=True)

# Humaneyes pegasus -> mlx
try:
    convert(hf_path=he, mlx_path=f'{DEST}/humaneyes-mlx', quantize=False)
    print('HUMANEYES converted', flush=True)
except Exception as e:
    print('HUMANEYES FAIL', type(e).__name__, e, flush=True)

print('DONE', flush=True)