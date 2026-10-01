"""
Публикует сокращённую Gemma (словарь 22 755 токенов, только текст) в MLX на Hugging Face:
веса из локальной папки + карточка модели.

  python upload_gemma_pruned.py                 # 6 бит из models/pruned-6bit, приватно
  python upload_gemma_pruned.py --public        # открыто
  python upload_gemma_pruned.py --card-only     # обновить только карточку
"""
import argparse

from huggingface_hub import HfApi

REPO = "Agisight/tyv-gemma4-e4b-pruned-mlx-6bit"
BASE = "google/gemma-4-E4B-it"
ADAPTER = "Agisight/tyv-gemma4-e4b-lora"
NLLB = "Agisight/nllb-rus-tyv-v3-dict_16.5k"
DATASET = "Agisight/tyv-rus-200k"
GITHUB = "https://github.com/Agisight/tyv-translation-mlx"

CARD = f"""---
library_name: mlx
base_model: {BASE}
license: apache-2.0
language:
- ru
- tyv
pipeline_tag: translation
datasets:
- {DATASET}
tags:
- mlx
- gemma
- gemma4
- translation
- low-resource
- tuvan
- on-device
- vocabulary-pruning
- 6-bit
---

# Gemma 4 E4B — Russian ↔ Tuvan translation, pruned vocabulary, MLX 6-bit

A compact, text-only version of [{BASE}](https://huggingface.co/{BASE}) fine-tuned for
bidirectional Russian ↔ Tuvan translation (LoRA adapter: [{ADAPTER}](https://huggingface.co/{ADAPTER})).
Small enough for on-device use on Apple Silicon, with the quality of the full model.

| | Full fine-tuned model (bf16) | **This model** |
|---|---|---|
| Vocabulary | 262,144 tokens | **22,755 tokens** |
| Parameters | ~8B (incl. vision and audio) | **4.28B (text only)** |
| Size on disk | 16 GB | **3.2 GB** |
| Peak memory, MacBook Air M4 | — | **3.8 GB** |
| Time per sentence (median) | — | **0.79 s** |

## How it was made

1. **Merged** the LoRA adapter into the base weights.
2. **Kept only the language model** — the vision and audio towers are removed.
3. **Pruned the vocabulary.** Gemma 4 E4B stores its 262K-token vocabulary in two tables: token
   embeddings (262,144 × 2,560) and per-layer embeddings (262,144 × 42 layers × 256) — about 44% of all
   weights. We kept the 22,755 tokens used in the **training** data, plus special, byte-fallback and
   single-character tokens, and the BPE "ancestors" needed to reproduce the original tokenization.
   The new tokenizer splits text **exactly** like the original on the whole test set; rare unseen
   pieces fall back to byte tokens.
4. **Quantized to 6 bits** with `mlx-lm`.

## Results

Clean held-out test set, 1,369 pairs (NLLB v3 test split of `{DATASET}` with training duplicates
removed). chrF++, greedy decoding, one sentence at a time on a MacBook Air M4.

| Model | Size | ru→tyv | tyv→ru |
|---|---|---|---|
| NLLB v3 ([{NLLB}](https://huggingface.co/{NLLB})) | 1.4 GB | 49.2 | 49.1 |
| Fine-tuned Gemma 4 E4B, bf16 | 16 GB | 50.5 | 50.9 |
| Pruned, bf16 | 8.0 GB | 50.5 | 50.8 |
| **This model: pruned, MLX 6-bit** | **3.2 GB** | **50.2** | **50.8** |
| Pruned, MLX 5-bit | 2.7 GB | 49.7 | 49.9 |
| Pruned, MLX 4-bit | 2.3 GB | 47.9 * | 43.3 * |

\\* First 100 pairs only (bf16 on the same pairs: 51.1 / 49.6).

Paired bootstrap significance (chrF++):

- **vs. NLLB v3:** better in both directions — p = 0.029 (ru→tyv) and p = 0.002 (tyv→ru).
- **vs. the full bf16 model:** no significant difference — p = 0.12 and p = 0.29.

6 bits is the smallest setting that keeps the full model's quality: 5 bits already loses ~1 point
(significant), and below 5 bits quality drops sharply — translation into Russian is the most sensitive.

## Prompt format

Single-turn chat message with a short prefix, exactly as in training:

- Russian → Tuvan: `ru→tyv: <Russian text>`
- Tuvan → Russian: `tyv→ru: <Tuvan text>`

## Usage

```bash
pip install -U mlx-lm
mlx_lm.generate --model {REPO} --prompt "ru→tyv: Завтра я поеду в Кызыл к родителям." --max-tokens 64 --temp 0.0
```

```python
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

model, tokenizer = load("{REPO}")
messages = [{{"role": "user", "content": "ru→tyv: Завтра я поеду в Кызыл к родителям."}}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print(generate(model, tokenizer, prompt=prompt, max_tokens=64, sampler=make_sampler(temp=0.0)))
```

## Limitations

- Trained and evaluated on one test domain; other genres are untested.
- Tuvan → Russian sometimes picks a wrong word sense and has to guess gender when the Tuvan source
  does not mark it (Tuvan has no grammatical gender).
- Text that needs tokens outside the pruned vocabulary is handled through byte tokens and may be
  translated less well.
- On iPhone, 3.8 GB peak memory needs a device with 8 GB of RAM; on-device testing is planned.

## Code

Training, pruning and evaluation notebooks and scripts: [Agisight/tyv-translation-mlx]({GITHUB}).

## Credits

- Base model: Google, Gemma 4 E4B.
- Data, fine-tuning, vocabulary pruning, quantization and evaluation: Ali Kuzhuget.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="models/pruned-6bit")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--card-only", action="store_true")
    args = ap.parse_args()

    api = HfApi()
    api.create_repo(REPO, private=not args.public, exist_ok=True)
    api.update_repo_settings(repo_id=REPO, private=not args.public)
    if not args.card_only:
        api.upload_folder(repo_id=REPO, folder_path=args.folder, ignore_patterns=[".converted", "README.md"],
                          commit_message="Gemma 4 E4B tyv: pruned vocabulary, text-only, MLX 6-bit")
    api.upload_file(path_or_fileobj=CARD.encode("utf-8"), path_in_repo="README.md", repo_id=REPO,
                    commit_message="Model card")
    print(f"Готово: https://huggingface.co/{REPO}  ({'публично' if args.public else 'приватно'})")


if __name__ == "__main__":
    main()
