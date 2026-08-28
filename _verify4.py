import mlx.core as mx, numpy as np, torch
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
    layer0 = ref.encoder.block[0].layer[0]
    SA = layer0.SelfAttention
    ln0_out_t = layer0.layer_norm(torch.tensor(emb_t)[None]).numpy()[0]
    # HF position_bias (what SA uses internally)
    pb_hf = SA.compute_bias(ln0_out_t.shape[0], ln0_out_t.shape[0]).numpy()[0]  # (H,9,9)
    # HF raw scores
    qh = SA.q(torch.tensor(ln0_out_t)[None]).numpy()[0].reshape(9,8,64).transpose(1,0,2)
    kh = SA.k(torch.tensor(ln0_out_t)[None]).numpy()[0].reshape(9,8,64).transpose(1,0,2)
    vh = SA.v(torch.tensor(ln0_out_t)[None]).numpy()[0].reshape(9,8,64).transpose(1,0,2)
    scores_hf = (qh @ kh.transpose(0,2,1) * (1/8.0)) + pb_hf
    soft_hf = torch.softmax(torch.tensor(scores_hf), dim=-1).numpy()
    att_hf = (soft_hf @ vh)  # (8,9,64)
    att_hf_t = att_hf.transpose(1,0,2).reshape(9,512)

# my pb
L = m.enc_layers[0]
ln_mine = np.asanyarray(L.ln1(mx.array(emb_t)))
ql=9
b = relative_position_bucket(position_delta(ql,ql), True, 32, 128)
my_pb = np.asanyarray(bias_from_buckets(L.rel_bias, b))
print("pb diff vs HF:", np.abs(my_pb-pb_hf).max(), "mean", np.abs(my_pb-pb_hf).mean())
print("pb_hf   mean|.|", np.abs(pb_hf).mean(), "max", np.abs(pb_hf).max())

# my scores
_ml = mx.array(ln_mine)
Q = mx.reshape(L.q(_ml),(9,8,64)).transpose(1,0,2)
K = mx.reshape(L.k(_ml),(9,8,64)).transpose(1,0,2)
V = mx.reshape(L.v(_ml),(9,8,64)).transpose(1,0,2)
my_scores = np.asanyarray(mx.matmul(Q,K.transpose(0,2,1))*(1/8.0) + mx.array(my_pb))
print("SCORES diff:", np.abs(my_scores-scores_hf).max(), "mean", np.abs(my_scores-scores_hf).mean())
