import mlx.core as mx, numpy as np, torch
from transformers import T5ForConditionalGeneration
from t5 import T5ConditionalMLX

snap = "/Users/terra/.cache/huggingface/hub/models--Lynote--humanize-text-model/snapshots/3b5497bbee8eeab31192294337dc9359ce68cd46/en"
ref = T5ForConditionalGeneration.from_pretrained(snap).to("cpu").eval()
m = T5ConditionalMLX(snap, "t5", d_model=512, num_encoder_layers=6,
                     num_decoder_layers=6, num_heads=8, d_kv=64, vocab_size=32128)
ids = torch.tensor([[3, 14, 8, 7, 3, 1, 9, 22, 2]])
il = ids[0].tolist()

hiddens_ref=[]
def hook(mod,inp,out): hiddens_ref.append(out[0].numpy()[0])
hs=[b.register_forward_hook(hook) for b in ref.encoder.block]
with torch.no_grad():
    emb_t = ref.shared(torch.tensor(ids)).numpy()[0]
    ref.encoder(input_ids=ids)
for h in hs: h.remove()

h = m._embed_inputs(il, m.pos_enc)
hiddens_mine=[]
for layer in m.enc_layers:
    h = layer.forward(h, m.num_heads, m.d_kv, m.rel_buckets, m.rel_max_dist)
    hiddens_mine.append(np.asanyarray(h))

for i,(r,my) in enumerate(zip(hiddens_ref,hiddens_mine)):
    rel = np.abs(r-my).mean() / (np.abs(r).mean()+1e-8)
    print(f"blk{i}: ref_mean|.|={np.abs(r).mean():7.3f} diff_mean={np.abs(r-my).mean():8.3f} rel={rel*100:5.1f}%")

with torch.no_grad():
    enc_t = ref.encoder(input_ids=ids).last_hidden_state[0].numpy()
my_enc=np.asanyarray(m.encode(il))
print("FINAL: diff_mean", np.abs(enc_t-my_enc).mean(), " rel%", np.abs(enc_t-my_enc).mean()/(np.abs(enc_t).mean()+1e-9)*100)
