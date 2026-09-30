"""
Конвертирует NLLB v3 из формата Hugging Face в MLX.

  python convert_nllb_mlx.py                    # f32 (точная копия, для проверки)
  python convert_nllb_mlx.py --dtype float16    # вдвое меньше
  python convert_nllb_mlx.py --bits 8           # int8 (как Core ML int8 в приложении)
  python convert_nllb_mlx.py --bits 4           # int4 — самая лёгкая

Результат: models/nllb-v3-mlx-<вариант>/ (config.json, weights.safetensors, файлы токенизатора).
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from nllb_mlx import NLLB, save_model

SKIP = ("encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight",
        "encoder.embed_positions", "decoder.embed_positions")


def dir_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024 / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="Agisight/nllb-rus-tyv-v3-dict_16.5k")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--bits", type=int, default=0, choices=[0, 4, 8], help="0 = без квантизации")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = Path(snapshot_download(args.hf))
    cfg = json.loads((src / "config.json").read_text())
    assert cfg.get("model_type") == "m2m_100", f"ожидал m2m_100, а это {cfg.get('model_type')}"

    weights = {}
    for f in sorted(src.glob("*.safetensors")):
        weights.update(mx.load(str(f)))
    mapped = {}
    for k, v in weights.items():
        k = k[len("model."):] if k.startswith("model.") else k
        if k.startswith(SKIP):
            continue
        mapped[k] = v.astype(getattr(mx, args.dtype))
    if "shared.weight" not in mapped:
        mapped["shared.weight"] = weights["lm_head.weight"].astype(getattr(mx, args.dtype))

    model = NLLB(cfg)
    model.load_weights(list(mapped.items()), strict=True)

    variant = f"q{args.bits}" if args.bits else {"float32": "f32", "float16": "f16", "bfloat16": "bf16"}[args.dtype]
    if args.bits:
        nn.quantize(model, group_size=args.group_size, bits=args.bits)
        cfg["mlx_quantization"] = {"bits": args.bits, "group_size": args.group_size}
    cfg["mlx_dtype"] = args.dtype
    mx.eval(model.parameters())

    out = Path(args.out or f"models/nllb-v3-mlx-{variant}")
    save_model(model, cfg, str(out))
    AutoTokenizer.from_pretrained(args.hf).save_pretrained(str(out))
    print(f"Исходная модель: {dir_mb(src):.0f} МБ")
    print(f"MLX ({variant}):  {dir_mb(out):.0f} МБ  → {out}")


if __name__ == "__main__":
    main()
