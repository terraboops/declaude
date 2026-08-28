import mlx.core as mx, numpy as np, torch
from transformers import T5ForConditionalGeneration
from t5 import T5ConditionalMLX, relative_position_bucket, position_delta, bias_from_buckets, mha as _mha

snap = "/Users/terra/.cache/huggingface/hub/models--Lynote--humanize-text-model/snapshots/3b5497bbee8eeab31192294337dc9359ce68cd46/en"
ref = T5ForConditionalGeneration.from_pretrained(snap).to("cpu").eval()
m = T5ConditionalMLX(snap, "t5", d_model=512, num_encoder_layers=6,
                     num_decoder_layers=6, num_heads=8, d_kv=64, vocab_size=32128)
ids = torch.tensor([[3, 14, 8, 7, 3, 1, 9, 22, 2]])
il = ids[0].tolist()
with torch.no_grad():
    emb_t = ref.shared(torch.tensor(ids)).numpy()[0]
    l0 = ref.encoder.block[0].layer[0]
    l1 = ref.encoder.block[0].layer[1]
    ln0_out_t = l0.layer_norm(torch.tensor(emb_t)[None]).numpy()[0]
    att_out_t = l0.SelfAttention(hidden_states=torch.tensor(ln0_out_t)[None], mask=None,
                                 key_value_states=None, position_bias=None, layer_head_mask=None,
                                 query_length=None, cache=None)[0].numpy()[0]
    post_att_t = emb_t + att_out_t
    ln1_out_t = l1.layer_norm(torch.tensor(post_att_t)[None]).numpy()[0]
    wi_t = l1.DenseReluDense.wi(torch.tensor(ln1_out_t)[None]).numpy()[0]
    relu_t = np.maximum(wi_t,0)
    wo_t = l1.DenseReluDense.wo(torch.tensor(relu_t)[None]).numpy()[0]

L = m.enc_layers[0]
my_ln0 = np.asanyarray(L.ln1(mx.array(emb_t)))
print("LN0 diff max/mean:", np.abs(ln0_out_t-my_ln0).max(), np.abs(ln0_out_t-my_ln0).mean())
_ln = mx.array(my_ln0)
b = relative_position_bucket(position_delta(9,9), True, 32, 128)
pb = mx.array(bias_from_buckets(L.rel_bias, b))
my_att = np.asanyarray(_mha(L.q(_ln), L.k(_ln), L.v(_ln), L.o, 8, 64, attn_bias=pb, causal=False))
print("ATT diff max/mean:", np.abs(att_out_t-my_att).max(), np.abs(att_out_t-my_att).mean(), "| mag ref/mine", np.abs(att_out_t).mean(), np.abs(my_att).mean())
my_post = emb_t + my_att
my_ln1 = np.asanyarray(L.ln2(mx.array(my_post)))
print("LN1 diff max/mean:", np.abs(ln1_out_t-my_ln1).max(), np.abs(ln1_out_t-my_ln1).mean())
my_wo = np.asanyarray(L.wo(mx.array(np.maximum( np.asanyarray(L.wi(mx.array(my_ln1))),0))))
print("WO diff max/mean:", np.abs(wo_t-my_wo).max(), np.abs(wo_t-my_wo).mean())
with torch.no_grad():
    out0 = ref.encoder.block[0](torch.tensor(emb_t)[None])[0].numpy()
my_out0 = np.asanyarray(L.forward(mx.array(emb_t), 8,64,32,128))
print("BLK0 full diff max/mean:", np.abs(out0-my_out0).max(), np.abs(out0-my_out0).mean(), "reldiff%", np.abs(out0-my_out0).mean()/np.abs(out0).mean()*100)
