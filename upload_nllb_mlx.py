"""
Публикует MLX-версию NLLB v3 на Hugging Face: веса, токенизатор, nllb_mlx.py и карточку модели.

  hf auth login                                   # один раз, токен с правом записи
  python upload_nllb_mlx.py --variant q8          # приватно (проверить)
  python upload_nllb_mlx.py --variant q8 --public # открыто

Репозиторий: Agisight/nllb-rus-tyv-v3-mlx-<variant>
"""
import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi

BASE = "Agisight/nllb-rus-tyv-v3-dict_16.5k"
DATASET = "Agisight/tyv-rus-200k"

# Результаты на чистом тесте (1369 пар) и замеры на MacBook Air M4 — из EXPERIMENTS.md
METRICS = {
    "f32": dict(size="1413 MB", chrf=("49.2", "49.1"), bleu=("19.8", "23.5"), lat=("0.19", "0.12"), mem="1491 MB"),
    "q8":  dict(size="400 MB",  chrf=("49.3", "48.8"), bleu=("19.9", "23.4"), lat=("0.07", "0.06"), mem="596 MB"),
    "q4":  dict(size="224 MB",  chrf=("48.5", "48.1"), bleu=("19.3", "22.2"), lat=("0.05", "0.03"), mem="420 MB"),
}
LABEL = {"f32": "float32 (exact)", "q8": "8-bit", "q4": "4-bit"}


def card(variant: str, repo: str) -> str:
    rows = "\n".join(
        f"| {LABEL[v]} | {m['size']} | {m['chrf'][0]} | {m['chrf'][1]} | {m['bleu'][0]} | {m['bleu'][1]} "
        f"| {m['lat'][0]} s | {m['lat'][1]} s | {m['mem']} |"
        for v, m in METRICS.items())
    return f"""---
library_name: mlx
license: cc-by-nc-4.0
language:
- ru
- tyv
pipeline_tag: translation
base_model: {BASE}
datasets:
- {DATASET}
tags:
- mlx
- nllb
- m2m_100
- translation
- low-resource
- tuvan
- on-device
---

# NLLB Russian–Tuvan v3 — MLX ({LABEL[variant]})

MLX port of [{BASE}](https://huggingface.co/{BASE}), a Russian ↔ Tuvan translation model
fine-tuned from Meta's NLLB-200-distilled-600M with a vocabulary pruned to 16.5K tokens.
Runs fully offline on Apple Silicon (Mac, iPhone, iPad) via [MLX](https://github.com/ml-explore/mlx).

`mlx-lm` does not support encoder-decoder models, so the M2M100 architecture is
re-implemented in `nllb_mlx.py` (included), with a KV cache for fast greedy decoding.
The float32 port reproduces the PyTorch model exactly: 100/100 identical translations on a
spot check and identical chrF++/BLEU on the full clean test set.

## Results

Clean test set: held-out test split of `{DATASET}` used for NLLB v3
(`shuffle(seed=42)`, pairs 2000–4000), with pairs whose Tuvan or Russian side also appears
in training removed — **1,369 pairs**. Greedy decoding, same preprocessing as NLLB v3 training.
Latency: median over 30 random test sentences per direction, one sentence at a time,
MacBook Air M4 (16 GB).

| Variant | Size | chrF++ ru→tyv | chrF++ tyv→ru | BLEU ru→tyv | BLEU tyv→ru | Latency ru→tyv | Latency tyv→ru | Peak memory |
|---|---|---|---|---|---|---|---|---|
{rows}

**This repository: {LABEL[variant]}.** 8-bit is lossless versus float32 and ~3× faster; 4-bit loses ~1 chrF++.

## Usage

```bash
pip install mlx transformers sentencepiece huggingface_hub
```

```python
import sys
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

path = snapshot_download("{repo}")
sys.path.insert(0, path)
from nllb_mlx import generate, load_model

tok = AutoTokenizer.from_pretrained(path)
model = load_model(path)

def translate(text, src="rus_Cyrl", tgt="tyv_Cyrl"):
    tok.src_lang = src
    enc = tok([text], return_tensors="np")
    out = generate(model, enc["input_ids"].astype("int32"), enc["attention_mask"].astype("int32"),
                   tok.convert_tokens_to_ids(tgt), tok.eos_token_id, max_new_tokens=128)
    return tok.batch_decode(out, skip_special_tokens=True)[0]

print(translate("Завтра я поеду в Кызыл к родителям."))
print(translate("Экии!", src="tyv_Cyrl", tgt="rus_Cyrl"))
```

For best quality, apply the same text normalization as in training (Moses punctuation
normalization + NFKC); see `preproc()` in the evaluation script of the project.

## Credits

- Base model: Meta AI, [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M) (CC-BY-NC-4.0).
- Fine-tuning pipeline adapted from David Dale's NLLB fine-tuning code, with his consultation.
- Fine-tuning, vocabulary pruning, MLX port: Ali Kuzhuget.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="q8", choices=list(METRICS))
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--repo", default=None)
    args = ap.parse_args()

    src = Path(f"models/nllb-v3-mlx-{args.variant}")
    assert (src / "weights.safetensors").exists(), f"нет {src} — сначала convert_nllb_mlx.py"
    repo = args.repo or f"Agisight/nllb-rus-tyv-v3-mlx-{args.variant}"

    shutil.copy("nllb_mlx.py", src / "nllb_mlx.py")
    (src / "README.md").write_text(card(args.variant, repo), encoding="utf-8")

    api = HfApi()
    api.create_repo(repo, private=not args.public, exist_ok=True)
    # create_repo не меняет видимость уже существующего репозитория — задаём явно
    api.update_repo_settings(repo_id=repo, private=not args.public)
    api.upload_folder(repo_id=repo, folder_path=str(src),
                      commit_message=f"NLLB v3 MLX {args.variant}")
    print(f"Готово: https://huggingface.co/{repo}  ({'публично' if args.public else 'приватно'})")


if __name__ == "__main__":
    main()
