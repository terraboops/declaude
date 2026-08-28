import mlx.core as mx, numpy as np, torch, math
from transformers import T5ForConditionalGeneration
from t5 import T5ConditionalMLX, relative_position_bucket, position_delta, bias_from_buckets

snap = "/Users/terra/.cache/huggingface/hub/models--Lynote--humanize-text-model/snapshots/3b5497bbee8eeab31192294337dc9359ce68cd46/en"
ref = T5ForConditionalGeneration.from_pretrained(snap).to("cpu").eval()
m = T5ConditionalMLX(snap, "t5", d_model=512, num_encoder_layers=6,
                     num_decoder_layers=6, num_heads=8, d_kv=64, vocab_size=32128)
ids = torch.tensor([[3, 14, 8, 7, 3, 1, 9, 22, 2]])
il = ids[0].tolist()
with torch.no_grad():
    emb_t = ref.shared(torch.tensor(ids)).numpy()[0]
    SA = ref.encoder.block[0].layer[0].SelfAttention
    ln0_out_t = ref.encoder.block[0].layer[0].layer_norm(torch.tensor(emb_t)[None]).numpy()[0]
    att_out_t = SA(hidden_states=torch.tensor(ln0_out_t)[None], mask=None,
                   key_value_states=None, position_bias=None, layer_head_mask=None,
                   query_length=None, cache=None)[0].numpy()[0]
    ln = torch.tensor(ln0_out_t)[None]
    qs = SA.q(ln).view(1,9,8,64).transpose(1,2)  # (1,8,9,64)
    ks = SA.k(ln).view(1,9,8,64).transpose(1,2)
    vs = SA.v(ln).view(1,9,8,64).transpose(1,2)
    pbh = SA.compute_bias(9,9)
    sc = torch.matmul(qs, ks.transpose(-1,-2)) * (SA.inner_dim**-0.5 if False else 1.0/math.sqrt(64)) + pbh
    # reproduce exactly like T5Attention
    masked_scores = sc
    probs = torch.nn.functional.softmax(masked_scores, dim=-1, dtype=torch.float32)
    attn = torch.einsum("bhqk,bhkd->bhqd", probs, vs)
    attn = attn.contiguous().view(1,9,512)
    out = SA.o(attn)[0].numpy()
print("torch-step attention out vs HF att_out_t:", np.abs(out-att_out_t).max(), "mean", np.abs(out-att_out_t).mean())
print("HF att mag mean:", np.abs(att_out_t).mean())

# now my mha
L = m.enc_layers[0]
_ln = mx.array(ln0_out_t)  # use EXACT hf ln
b = relative_position_bucket(position_delta(9,9), True, 32, 128)
pb = mx.array(bias_from_buckets(L.rel_bias, b))
from t5 import mha as _mha
my_att = np.asanyarray(_mha(L.q(_ln), L.k(_ln), L.v(_ln), L.o, 8, 64, attn_bias=pb, causal=False))
print("MY mha vs HF att_out_t:", np.abs(my_att-att_out_t).max(), "mean", np.abs(my_att-att_out_t).mean())
print("MY mha vs torch-step :", np.abs(my_att-out).max(), "mean", np.abs(my_att-out).mean())
