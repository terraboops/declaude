import torch
from transformers import T5ForConditionalGeneration
snap = "/Users/terra/.cache/huggingface/hub/models--Lynote--humanize-text-model/snapshots/3b5497bbee8eeab31192294337dc9359ce68cd46/en"
ref = T5ForConditionalGeneration.from_pretrained(snap).to("cpu").eval()
ln = ref.encoder.block[0].layer[0].layer_norm
print("LN class:", type(ln).__name__)
print("has bias:", hasattr(ln,"bias"))
if hasattr(ln,"bias"):
    print("bias is None?", ln.bias is None)
    if ln.bias is not None:
        print("bias max abs:", ln.bias.detach().abs().max().item())
print("weight sum:", ln.weight.detach().sum().item())
print("num params pre-EPS: weight, bias:", ln.weight.numel())
print("eps:", getattr(ln,"variance_epsilon","n/a"))
