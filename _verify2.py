import mlx.core as mx, numpy as np, torch, math
from transformers import T5ForConditionalGeneration

snap = "/Users/terra/.cache/huggingface/hub/models--Lynote--humanize-text-model/snapshots/3b5497bbee8eeab31192294337dc9359ce68cd46/en"
ref = T5ForConditionalGeneration.from_pretrained(snap).to("cpu").eval()
m_ = None
SA = ref.encoder.block[0].layer[0].SelfAttention
ql=9
with torch.no_grad():
    pb = SA.compute_bias(ql, ql).numpy()[0]
    # recover HF bucket by scanning: for each (q,k) the bucket that reproduces pb[0,q,k] via lookup
    table = SA.relative_attention_bias.weight.detach().numpy()  # (32,8)
    diff = np.abs(pb.transpose(1,2,0)[...,None,:] - table[None,None,:,:])  # (q,k,32,8)
    best = diff.sum(-1).argmin(-1)  # (q,k) best bucket per head-mean
print("bf bucket shape", best.shape)
# my bucket
from t5 import relative_position_bucket, position_delta
b = np.asanyarray(relative_position_bucket(position_delta(ql,ql), True, 32, 128))
print("my bucket:\n", b)
print("hf bucket:\n", best)
print("equal?", np.array_equal(b, best))
print("mismatch where:", np.argwhere(b != best).tolist())
