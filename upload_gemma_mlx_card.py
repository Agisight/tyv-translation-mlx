"""
Карточки для MLX-версий дообученной Gemma (заменяет пустую заготовку mlx-vlm).
Меняется только README.md — веса не трогаются.

  python upload_gemma_mlx_card.py            # обе: 4-bit и 8-bit
  python upload_gemma_mlx_card.py --bits 4

Когда посчитаем качество 4/8 бит на Маке — впиши цифры в QUANT_RESULTS и запусти снова.
"""
import argparse

from huggingface_hub import HfApi

BASE = "google/gemma-4-E4B-it"
ADAPTER = "Agisight/tyv-gemma4-e4b-lora"
NLLB = "Agisight/nllb-rus-tyv-v3-dict_16.5k"
DATASET = "Agisight/tyv-rus-200k"
GITHUB = "https://github.com/Agisight/tyv-translation-mlx"

SIZES = {4: ("4.9 GB", "5.2"), 8: ("8.4 GB", "9.0")}

# chrF++ ru→tyv / tyv→ru на чистом тесте (1369 пар) для квантизованных версий.
# None = ещё не посчитано на Маке.
QUANT_RESULTS = {4: None, 8: None}


def card(bits: int) -> str:
    size, bpw = SIZES[bits]
    other = 8 if bits == 4 else 4
    r = QUANT_RESULTS[bits]
    quant_line = (f"| This {bits}-bit MLX model | **{r[0]}** | **{r[1]}** |" if r
                  else f"| This {bits}-bit MLX model | evaluation in progress | evaluation in progress |")
    return f"""---
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
- {bits}-bit
---

# Gemma 4 E4B — Russian ↔ Tuvan translation, MLX {bits}-bit

[{BASE}](https://huggingface.co/{BASE}) fine-tuned for bidirectional Russian ↔ Tuvan
translation (LoRA adapter: [{ADAPTER}](https://huggingface.co/{ADAPTER})), merged into the base
weights and quantized to **{bits} bits** for Apple Silicon with
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm). Runs fully offline on a Mac.

- Size: **{size}** ({bpw} bits per weight on average; some layers are kept at higher precision).
- Other variant: [Agisight/tyv-gemma4-e4b-mlx-{other}bit](https://huggingface.co/Agisight/tyv-gemma4-e4b-mlx-{other}bit).

## Results

Clean held-out test set, 1,369 pairs (NLLB v3 test split of `{DATASET}` with training
duplicates removed). chrF++, greedy decoding.

| Model | ru→tyv | tyv→ru |
|---|---|---|
| NLLB v3 ([{NLLB}](https://huggingface.co/{NLLB})) | 49.2 | 49.1 |
| Fine-tuned Gemma 4 E4B, bf16 (adapter) | **50.5** | **50.9** |
{quant_line}

The bf16 adapter significantly outperforms NLLB v3 in both directions (paired bootstrap,
p < 0.05) and also on BLEU, normalized chrF++ and COMET — see the
[adapter card](https://huggingface.co/{ADAPTER}) for the full table.

## Prompt format

Single-turn chat message with a short prefix, exactly as in training:

- Russian → Tuvan: `ru→tyv: <Russian text>`
- Tuvan → Russian: `tyv→ru: <Tuvan text>`

## Usage

```bash
pip install -U mlx-vlm
mlx_vlm.generate --model Agisight/tyv-gemma4-e4b-mlx-{bits}bit \\
  --prompt "ru→tyv: Завтра я поеду в Кызыл к родителям." --max-tokens 64 --temperature 0.0
```

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("Agisight/tyv-gemma4-e4b-mlx-{bits}bit")
prompt = apply_chat_template(processor, model.config, "ru→tyv: Завтра я поеду в Кызыл к родителям.")
print(generate(model, processor, prompt, max_tokens=64, temperature=0.0))
```

## Notes

- The vision and audio towers of Gemma 4 are kept, but the model was trained for text
  translation only.
- On iPhone, {size} exceeds Apple's ~2 GB guidance for on-device models; a pruned-vocabulary
  version for iOS is planned.

## Code

Training notebook (Colab), data preparation, conversion and evaluation scripts:
[Agisight/tyv-translation-mlx]({GITHUB}).

## Credits

- Base model: Google, Gemma 4 E4B.
- Data, fine-tuning, evaluation and MLX conversion: Ali Kuzhuget.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, choices=[4, 8], nargs="*", default=[4, 8])
    args = ap.parse_args()
    api = HfApi()
    for bits in args.bits:
        repo = f"Agisight/tyv-gemma4-e4b-mlx-{bits}bit"
        api.upload_file(path_or_fileobj=card(bits).encode("utf-8"), path_in_repo="README.md",
                        repo_id=repo, commit_message="Model card")
        print(f"Карточка обновлена: https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
