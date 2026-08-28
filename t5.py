"""
Self-contained, native-MLX implementation of the T5-family encoder-decoder
architecture (T5 and Pegasus).

Loads HuggingFace safetensors weights directly into MLX arrays via
safetensors (framework='np') + mx.array.  No torch / transformers / mlx-lm at
inference time (the *driver* script may use transformers only to build a
tokenizer).

Both T5 and Pegasus share the T5 encoder-decoder skeleton, so one code path
serves both, parameterized by:

  * positional mode
      - 'relative' : T5. Relative attention bias per HF t5 bucketing
        (relative_attention_num_buckets / relative_attention_max_distance),
        bidirectional in encoder & decoder self-attention, absent from
        cross-attention.  No absolute position embedding.
      - 'absolute' : Pegasus. Learned absolute position embeddings
        (model.encoder/decoder.embed_positions) added to the (optionally
        scaled) word embedding.

  * HF weight-name layout  ('t5' or 'pegasus').

Everything runs in float32.
"""

import glob
import math
import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open


# --------------------------------------------------------------------------- #
# small affine layer
# --------------------------------------------------------------------------- #
class Linear:
    __slots__ = ("weight", "bias")

    def __init__(self, weight, bias):
        # HF stores linear weights as (out, in); transpose to (in, out)
        self.weight = weight.astype(mx.float32).T
        self.bias = bias.astype(mx.float32) if bias is not None else None

    def __call__(self, x):
        out = mx.matmul(x, self.weight)  # x (..., in); weight (in, out)
        if self.bias is not None:
            out = out + self.bias
        return out


# --------------------------------------------------------------------------- #
# Layer norms. T5 uses a NON-centering normalization (normalize by
# rsqrt(mean(x^2)+eps)); Pegasus uses a standard centered LayerNorm.
# --------------------------------------------------------------------------- #
class T5LayerNorm:
    __slots__ = ("weight", "bias", "eps")

    def __init__(self, dim, eps):
        self.weight = mx.ones(dim)
        self.bias = None  # T5 layernorm is weight-only
        self.eps = eps

    def __call__(self, x):
        var = mx.mean(x * x, axis=-1, keepdims=True)  # mean of squares, NOT centered
        return self.weight * (x * mx.rsqrt(var + self.eps))


# --------------------------------------------------------------------------- #
# relative-position bucketing (faithful port of transformers .t5)
# --------------------------------------------------------------------------- #
def relative_position_bucket(relative_position, bidirectional=True,
                             num_buckets=32, max_distance=128):
    # relative_position: (q_len, k_len) int, value = q_idx - k_idx
    ret = mx.zeros(relative_position.shape, dtype=mx.int32)
    n = relative_position
    if bidirectional:
        num_buckets //= 2
        ret = ret + (relative_position > 0).astype(mx.int32) * num_buckets
        n = mx.abs(relative_position)
    else:
        n = mx.maximum(-relative_position, mx.zeros_like(relative_position))
    max_exact = num_buckets // 2
    is_small = n < max_exact
    nf = mx.maximum(n.astype(mx.float32), 1.0)
    val_if_large = max_exact + (
        mx.log(nf / max_exact) / mx.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    )
    val_if_large = mx.minimum(
        val_if_large.astype(mx.int32),
        mx.full_like(val_if_large.astype(mx.int32), num_buckets - 1),
    )
    return ret + mx.where(is_small, n, val_if_large)


def position_delta(q_len, k_len):
    idx = mx.arange(q_len, dtype=mx.int32)[:, None]
    jdx = mx.arange(k_len, dtype=mx.int32)[None, :]
    return jdx - idx  # key_position - query_position (matches HF compute_bias)


def bias_from_buckets(table, bucket):
    # table (num_buckets, num_heads); bucket (q_len, k_len)
    flat = mx.take(table, mx.reshape(bucket, (-1,)), axis=0)
    q_len, k_len = bucket.shape
    return flat.reshape(q_len, k_len, -1).transpose(2, 0, 1)  # (H,q,k)


def causal_mask(q_len, k_len):
    m = mx.full((q_len, k_len), -float("inf"), dtype=mx.float32)
    return mx.triu(m, k=1)


# --------------------------------------------------------------------------- #
# multi-head attention.
# Head dim is contiguous (dim index 2 post-reshape) -- the convention used by
# BOTH HF T5 (transformers>=4.50) and HF Pegasus. q/k/v: (seq, d_model).
# --------------------------------------------------------------------------- #
def mha(q, k, v, o_proj, num_heads, d_kv, attn_bias=None, causal=False, scale=1.0):
    q_len = q.shape[0]
    k_len = k.shape[0]
    Q = mx.reshape(q, (q_len, num_heads, d_kv)).transpose(1, 0, 2)
    K = mx.reshape(k, (k_len, num_heads, d_kv)).transpose(1, 0, 2)
    V = mx.reshape(v, (k_len, num_heads, d_kv)).transpose(1, 0, 2)
    scores = mx.matmul(Q, K.transpose(0, 2, 1)) * scale
    if attn_bias is not None:
        scores = scores + attn_bias
    if causal:
        scores = scores + causal_mask(q_len, k_len)
    probs = mx.softmax(scores, axis=-1)
    out = mx.matmul(probs, V)
    out = out.transpose(1, 0, 2).reshape(q_len, num_heads * d_kv)
    return o_proj(out)


# --------------------------------------------------------------------------- #
# Encoder layer (pre-norm, residual, relu FFN)
# --------------------------------------------------------------------------- #
class EncoderLayer:
    def __init__(self, ln1, q, k, v, o, ln2, wi, wo, rel_bias):
        self.ln1, self.q, self.k, self.v, self.o = ln1, q, k, v, o
        self.ln2, self.wi, self.wo = ln2, wi, wo
        self.rel_bias = rel_bias

    def forward(self, hidden, heads, d_kv, buckets, max_dist, scale=1.0):
        residual = hidden
        h = self.ln1(hidden)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        bias = None
        if self.rel_bias is not None:
            ql = h.shape[0]
            bucket = relative_position_bucket(
                position_delta(ql, ql), True, buckets, max_dist)
            bias = bias_from_buckets(self.rel_bias, bucket)
        hidden = residual + mha(q, k, v, self.o, heads, d_kv,
                                attn_bias=bias, causal=False, scale=scale)
        residual = hidden
        h = self.ln2(hidden)
        h = self.wi(h)
        h = mx.maximum(h, mx.zeros_like(h))  # relu
        h = self.wo(h)
        return residual + h


# --------------------------------------------------------------------------- #
# Decoder layer: self-attn (causal + relative bias), cross-attn (to encoder),
# relu FFN.
# --------------------------------------------------------------------------- #
class DecoderLayer:
    def __init__(self, ln1, q, k, v, o, rel_bias,
                 ln2, cq, ck, cv, co,
                 ln3, wi, wo):
        self.ln1, self.q, self.k, self.v, self.o = ln1, q, k, v, o
        self.rel_bias = rel_bias
        self.ln2, self.cq, self.ck, self.cv, self.co = ln2, cq, ck, cv, co
        self.ln3, self.wi, self.wo = ln3, wi, wo

    def forward_self(self, hidden, heads, d_kv, buckets, max_dist, scale=1.0):
        residual = hidden
        h = self.ln1(hidden)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        bias = None
        if self.rel_bias is not None:
            ql = h.shape[0]
            bucket = relative_position_bucket(
                position_delta(ql, ql), False, buckets, max_dist)
            bias = bias_from_buckets(self.rel_bias, bucket)
        hidden = residual + mha(q, k, v, self.o, heads, d_kv,
                                attn_bias=bias, causal=True, scale=scale)
        return hidden

    def forward_cross(self, hidden, enc_out, heads, d_kv, scale=1.0):
        residual = hidden
        h = self.ln2(hidden)
        q = self.cq(h)
        k = self.ck(enc_out)
        v = self.cv(enc_out)
        hidden = residual + mha(q, k, v, self.co, heads, d_kv,
                                attn_bias=None, causal=False, scale=scale)
        return hidden

    def forward_ff(self, hidden):
        residual = hidden
        h = self.ln3(hidden)
        h = self.wi(h)
        h = mx.maximum(h, mx.zeros_like(h))
        h = self.wo(h)
        return residual + h

    def forward(self, hidden, enc_out, heads, d_kv, buckets, max_dist, scale=1.0):
        hidden = self.forward_self(hidden, heads, d_kv, buckets, max_dist, scale)
        hidden = self.forward_cross(hidden, enc_out, heads, d_kv, scale)
        return self.forward_ff(hidden)


# --------------------------------------------------------------------------- #
# T5-family conditional model
# --------------------------------------------------------------------------- #
class T5ConditionalMLX:
    def __init__(self, snapshot_dir, style, *, d_model, d_ff=None,
                 num_encoder_layers, num_decoder_layers, num_heads, d_kv,
                 vocab_size, positional="relative", rel_buckets=32,
                 rel_max_dist=128, scale_embedding=False,
                 layer_norm_eps=1e-6, add_final_layer_norm=True,
                 attn_scale=None):
        assert style in ("t5", "pegasus")
        self.style = style
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.d_kv = d_kv
        self.vocab_size = vocab_size
        self.positional = positional
        self.rel_buckets = rel_buckets
        self.rel_max_dist = rel_max_dist
        self.scale_embedding = scale_embedding
        self.eps = layer_norm_eps
        self.add_final_layer_norm = add_final_layer_norm
        # T5 folds relative bias in and does NOT scale q·k (HF self.scaling=1.0).
        # Pegasus scales by head_dim**-0.5 (set explicitly by caller).
        self.attn_scale = attn_scale if attn_scale is not None else 1.0
        self.w = self._load(snapshot_dir)
        self._build(num_encoder_layers, num_decoder_layers)

    # ---------- weight loading ---------- #
    def _load(self, snapshot_dir):
        hits = glob.glob(snapshot_dir + "/**/model.safetensors", recursive=True)
        if not hits:
            raise FileNotFoundError(f"no model.safetensors under {snapshot_dir}")
        # pick the deepest / most specific (e.g. the en/ subdir for Lynote)
        hits.sort(key=lambda p: p.count("/"))
        w = {}
        with safe_open(hits[0], framework="np") as f:
            for key in f.keys():
                w[key] = mx.array(f.get_tensor(key))
        return w

    # ---------- construction ---------- #
    def _build(self, n_enc, n_dec):
        s = self.style
        d = self.d_model
        eps = self.eps

        def LN():
            if s == "t5":
                return T5LayerNorm(d, eps)      # T5 non-centering layernorm
            return nn.LayerNorm(d, eps=eps)     # Pegasus standard centered

        def ml(weight_key, has_bias=False):
            w = self.w[weight_key]
            b = self.w[weight_key.replace(".weight", ".bias")] if has_bias else None
            return Linear(w, b)

        # shared embedding
        self.embed = self.w["shared.weight" if s == "t5" else "model.shared.weight"].astype(mx.float32)
        self.embed_scale = math.sqrt(d) if self.scale_embedding else 1.0
        self.pos_enc = self.pos_dec = None
        if s == "pegasus":
            self.pos_enc = self.w["model.encoder.embed_positions.weight"].astype(mx.float32)
            self.pos_dec = self.w["model.decoder.embed_positions.weight"].astype(mx.float32)
        self.out_bias = self.w.get("final_logits_bias", None)

        self.enc_layers = []
        self.dec_layers = []
        enc_rel = dec_rel = None

        for i in range(n_enc):
            if s == "t5":
                B = f"encoder.block.{i}.layer.0.SelfAttention"
                F = f"encoder.block.{i}.layer.1.DenseReluDense"
                if i == 0:
                    enc_rel = self.w[B + ".relative_attention_bias.weight"].astype(mx.float32)
                ln1 = LN(); ln1.weight = self.w[f"encoder.block.{i}.layer.0.layer_norm.weight"]; ln1.bias = None
                ln2 = LN(); ln2.weight = self.w[f"encoder.block.{i}.layer.1.layer_norm.weight"]; ln2.bias = None
                layer = EncoderLayer(ln1, ml(B + ".q.weight"), ml(B + ".k.weight"),
                                     ml(B + ".v.weight"), ml(B + ".o.weight"),
                                     ln2, ml(F + ".wi.weight"), ml(F + ".wo.weight"),
                                     enc_rel)
            else:
                B = f"model.encoder.layers.{i}.self_attn"
                F = f"model.encoder.layers.{i}"
                ln1 = LN(); ln1.weight = self.w[f"model.encoder.layers.{i}.self_attn_layer_norm.weight"]; ln1.bias = self.w[f"model.encoder.layers.{i}.self_attn_layer_norm.bias"]
                ln2 = LN(); ln2.weight = self.w[f"model.encoder.layers.{i}.final_layer_norm.weight"]; ln2.bias = self.w[f"model.encoder.layers.{i}.final_layer_norm.bias"]
                layer = EncoderLayer(ln1, ml(B + ".q_proj.weight", True), ml(B + ".k_proj.weight", True),
                                     ml(B + ".v_proj.weight", True), ml(B + ".out_proj.weight", True),
                                     ln2, ml(F + ".fc1.weight", True), ml(F + ".fc2.weight", True),
                                     None)
            self.enc_layers.append(layer)

        for i in range(n_dec):
            if s == "t5":
                B = f"decoder.block.{i}.layer.0.SelfAttention"
                C = f"decoder.block.{i}.layer.1.EncDecAttention"
                F = f"decoder.block.{i}.layer.2.DenseReluDense"
                if i == 0:
                    dec_rel = self.w[B + ".relative_attention_bias.weight"].astype(mx.float32)
                ln1 = LN(); ln1.weight = self.w[f"decoder.block.{i}.layer.0.layer_norm.weight"]; ln1.bias = None
                ln2 = LN(); ln2.weight = self.w[f"decoder.block.{i}.layer.1.layer_norm.weight"]; ln2.bias = None
                ln3 = LN(); ln3.weight = self.w[f"decoder.block.{i}.layer.2.layer_norm.weight"]; ln3.bias = None
                layer = DecoderLayer(
                    ln1, ml(B + ".q.weight"), ml(B + ".k.weight"), ml(B + ".v.weight"), ml(B + ".o.weight"), dec_rel,
                    ln2, ml(C + ".q.weight"), ml(C + ".k.weight"), ml(C + ".v.weight"), ml(C + ".o.weight"),
                    ln3, ml(F + ".wi.weight"), ml(F + ".wo.weight"))
            else:
                B = f"model.decoder.layers.{i}.self_attn"
                C = f"model.decoder.layers.{i}.encoder_attn"
                F = f"model.decoder.layers.{i}"
                ln1 = LN(); ln1.weight = self.w[f"model.decoder.layers.{i}.self_attn_layer_norm.weight"]; ln1.bias = self.w[f"model.decoder.layers.{i}.self_attn_layer_norm.bias"]
                ln2 = LN(); ln2.weight = self.w[f"model.decoder.layers.{i}.encoder_attn_layer_norm.weight"]; ln2.bias = self.w[f"model.decoder.layers.{i}.encoder_attn_layer_norm.bias"]
                ln3 = LN(); ln3.weight = self.w[f"model.decoder.layers.{i}.final_layer_norm.weight"]; ln3.bias = self.w[f"model.decoder.layers.{i}.final_layer_norm.bias"]
                layer = DecoderLayer(
                    ln1, ml(B + ".q_proj.weight", True), ml(B + ".k_proj.weight", True), ml(B + ".v_proj.weight", True), ml(B + ".out_proj.weight", True), None,
                    ln2, ml(C + ".q_proj.weight", True), ml(C + ".k_proj.weight", True), ml(C + ".v_proj.weight", True), ml(C + ".out_proj.weight", True),
                    ln3, ml(F + ".fc1.weight", True), ml(F + ".fc2.weight", True))
            self.dec_layers.append(layer)

        self.enc_final = LN()
        self.dec_final = LN()
        if s == "t5":
            self.enc_final.weight = self.w["encoder.final_layer_norm.weight"]; self.enc_final.bias = None
            self.dec_final.weight = self.w["decoder.final_layer_norm.weight"]; self.dec_final.bias = None
        else:
            self.enc_final.weight = self.w["model.encoder.layer_norm.weight"]; self.enc_final.bias = self.w["model.encoder.layer_norm.bias"]
            self.dec_final.weight = self.w["model.decoder.layer_norm.weight"]; self.dec_final.bias = self.w["model.decoder.layer_norm.bias"]

    # ---------- forward ---------- #
    def _embed_inputs(self, ids, pos):
        x = mx.take(self.embed, mx.array(ids, dtype=mx.int32), axis=0).astype(mx.float32)
        if self.embed_scale != 1.0:
            x = x * self.embed_scale
        if pos is not None:
            L = x.shape[0]
            p = mx.take(pos, mx.arange(L, dtype=mx.int32), axis=0)
            x = x + p
        return x

    def encode(self, input_ids):
        hidden = self._embed_inputs(input_ids, self.pos_enc)
        sc = self.attn_scale
        for layer in self.enc_layers:
            hidden = layer.forward(hidden, self.num_heads, self.d_kv,
                                   self.rel_buckets, self.rel_max_dist, sc)
        if self.add_final_layer_norm:
            hidden = self.enc_final(hidden)
        return hidden

    def _decoder_logits(self, decoder_ids, enc_out):
        hidden = self._embed_inputs(decoder_ids, self.pos_dec)
        sc = self.attn_scale
        for layer in self.dec_layers:
            hidden = layer.forward(hidden, enc_out, self.num_heads, self.d_kv,
                                   self.rel_buckets, self.rel_max_dist, sc)
        if self.add_final_layer_norm:
            hidden = self.dec_final(hidden)
        logits = mx.matmul(hidden, self.embed.T)  # (L, vocab)
        if self.out_bias is not None:
            logits = logits + self.out_bias.astype(mx.float32)
        return logits

    def generate(self, input_ids, max_new_tokens=200, eos_id=1, start_id=0):
        enc_out = self.encode(input_ids)
        gen = [start_id]
        for _ in range(max_new_tokens):
            logits = self._decoder_logits(gen, enc_out)
            nxt = int(mx.argmax(logits[-1]).item())
            gen.append(nxt)
            if nxt == eos_id:
                break
        return gen[1:]
