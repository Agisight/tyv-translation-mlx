"""
NLLB (архитектура M2M100) на MLX: энкодер + декодер + жадная генерация с KV-кэшем.

Повторяет transformers.M2M100ForConditionalGeneration:
- pre-LayerNorm в каждом слое, финальный LayerNorm у энкодера и декодера;
- эмбеддинги умножаются на sqrt(d_model), общие для энкодера, декодера и выходного слоя;
- синусоидальные позиции со сдвигом padding_idx (таблица не хранится в весах, считается);
- декодер стартует с [eos, тег целевого языка].

Используется в convert_nllb_mlx.py и evaluate_nllb_mlx.py.
"""
import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

NEG_INF = -1e9


def sinusoidal_table(num: int, dim: int, pad_idx: int) -> mx.array:
    half = dim // 2
    step = math.log(10000) / (half - 1)
    freqs = np.exp(np.arange(half, dtype=np.float32) * -step)
    ang = np.arange(num, dtype=np.float32)[:, None] * freqs[None, :]
    table = np.concatenate([np.sin(ang), np.cos(ang)], axis=1)
    if dim % 2 == 1:
        table = np.concatenate([table, np.zeros((num, 1), dtype=np.float32)], axis=1)
    table[pad_idx] = 0.0
    return mx.array(table)


class Attention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = d // heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def _split(self, x):
        B, L, _ = x.shape
        return x.reshape(B, L, self.heads, self.head_dim).transpose(0, 2, 1, 3)

    def __call__(self, x, memory=None, mask=None, cache=None, is_cross=False):
        B, L, D = x.shape
        q = self._split(self.q_proj(x))
        if is_cross:
            if cache is None:
                k, v = self._split(self.k_proj(memory)), self._split(self.v_proj(memory))
            else:
                k, v = cache
        else:
            k, v = self._split(self.k_proj(x)), self._split(self.v_proj(x))
            if cache is not None:
                k = mx.concatenate([cache[0], k], axis=2)
                v = mx.concatenate([cache[1], v], axis=2)
        if mask is not None:
            mask = mask.astype(q.dtype)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.out_proj(out), (k, v)


class EncoderLayer(nn.Module):
    def __init__(self, d, heads, ffn):
        super().__init__()
        self.self_attn = Attention(d, heads)
        self.self_attn_layer_norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, ffn)
        self.fc2 = nn.Linear(ffn, d)
        self.final_layer_norm = nn.LayerNorm(d)

    def __call__(self, h, mask):
        a, _ = self.self_attn(self.self_attn_layer_norm(h), mask=mask)
        h = h + a
        return h + self.fc2(nn.relu(self.fc1(self.final_layer_norm(h))))


class DecoderLayer(nn.Module):
    def __init__(self, d, heads, ffn):
        super().__init__()
        self.self_attn = Attention(d, heads)
        self.self_attn_layer_norm = nn.LayerNorm(d)
        self.encoder_attn = Attention(d, heads)
        self.encoder_attn_layer_norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, ffn)
        self.fc2 = nn.Linear(ffn, d)
        self.final_layer_norm = nn.LayerNorm(d)

    def __call__(self, h, memory, self_mask, cross_mask, cache):
        self_cache, cross_cache = cache if cache is not None else (None, None)
        a, self_cache = self.self_attn(self.self_attn_layer_norm(h), mask=self_mask, cache=self_cache)
        h = h + a
        a, cross_cache = self.encoder_attn(self.encoder_attn_layer_norm(h), memory=memory,
                                           mask=cross_mask, cache=cross_cache, is_cross=True)
        h = h + a
        h = h + self.fc2(nn.relu(self.fc1(self.final_layer_norm(h))))
        return h, (self_cache, cross_cache)


class Stack(nn.Module):
    def __init__(self, layer_cls, n, d, heads, ffn):
        super().__init__()
        self.layers = [layer_cls(d, heads, ffn) for _ in range(n)]
        self.layer_norm = nn.LayerNorm(d)


class NLLB(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["d_model"]
        self.cfg = cfg
        self.pad = cfg["pad_token_id"]
        self.shared = nn.Embedding(cfg["vocab_size"], d)
        self.encoder = Stack(EncoderLayer, cfg["encoder_layers"], d,
                             cfg["encoder_attention_heads"], cfg["encoder_ffn_dim"])
        self.decoder = Stack(DecoderLayer, cfg["decoder_layers"], d,
                             cfg["decoder_attention_heads"], cfg["decoder_ffn_dim"])
        self._scale = math.sqrt(d) if cfg.get("scale_embedding", True) else 1.0
        self._pos = sinusoidal_table(cfg["max_position_embeddings"] + 2, d, self.pad)

    def _embed(self, ids, positions):
        e = self.shared(ids) * self._scale
        return e + self._pos[positions].astype(e.dtype)

    def encode(self, ids, attn_mask):
        """ids, attn_mask: [B, L] (mask: 1 = токен, 0 = паддинг)."""
        m = attn_mask.astype(mx.int32)
        positions = mx.cumsum(m, axis=1) * m + self.pad
        h = self._embed(ids, positions)
        add_mask = ((1 - m) * NEG_INF)[:, None, None, :].astype(mx.float32)
        for layer in self.encoder.layers:
            h = layer(h, add_mask)
        return self.encoder.layer_norm(h)

    def decode(self, ids, memory, cross_mask, caches, past_len: int):
        """ids: [B, L] новые токены декодера; caches: список на слой или None."""
        B, L = ids.shape
        positions = mx.broadcast_to(mx.arange(1, L + 1)[None, :] + past_len + self.pad, (B, L))
        h = self._embed(ids, positions)
        self_mask = None
        if L > 1:
            causal = mx.triu(mx.full((L, L), NEG_INF), k=1)
            if past_len:
                causal = mx.concatenate([mx.zeros((L, past_len)), causal], axis=1)
            self_mask = causal[None, None]
        new_caches = []
        for i, layer in enumerate(self.decoder.layers):
            h, c = layer(h, memory, self_mask, cross_mask,
                         caches[i] if caches is not None else None)
            new_caches.append(c)
        h = self.decoder.layer_norm(h)
        return self.shared.as_linear(h), new_caches


def generate(model: NLLB, ids: np.ndarray, attn_mask: np.ndarray, tgt_lang_id: int,
             eos_id: int, max_new_tokens: int):
    """Жадная генерация батчем. Возвращает список списков id (без eos и тега языка)."""
    ids_mx, mask_mx = mx.array(ids), mx.array(attn_mask)
    memory = model.encode(ids_mx, mask_mx)
    cross_mask = ((1 - mask_mx.astype(mx.int32)) * NEG_INF)[:, None, None, :].astype(mx.float32)
    B = ids.shape[0]
    start = mx.array(np.tile(np.array([[eos_id, tgt_lang_id]], dtype=np.int32), (B, 1)))
    logits, caches = model.decode(start, memory, cross_mask, None, past_len=0)
    past = 2
    out = [[] for _ in range(B)]
    done = np.zeros(B, dtype=bool)
    for _ in range(max_new_tokens):
        nxt = mx.argmax(logits[:, -1, :], axis=-1)
        nxt_np = np.array(nxt)
        for b in range(B):
            if not done[b]:
                if nxt_np[b] == eos_id:
                    done[b] = True
                else:
                    out[b].append(int(nxt_np[b]))
        if done.all():
            break
        logits, caches = model.decode(nxt[:, None].astype(mx.int32), memory, cross_mask, caches, past_len=past)
        past += 1
    return out


def load_model(path: str) -> NLLB:
    """Загрузка сконвертированной модели (папка с config.json и weights.safetensors)."""
    p = Path(path)
    cfg = json.loads((p / "config.json").read_text())
    model = NLLB(cfg)
    q = cfg.get("mlx_quantization")
    if q:
        nn.quantize(model, group_size=q["group_size"], bits=q["bits"])
    model.load_weights(str(p / "weights.safetensors"))
    mx.eval(model.parameters())
    return model


def save_model(model: NLLB, cfg: dict, path: str):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(p / "weights.safetensors"), dict(tree_flatten(model.parameters())))
    (p / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
