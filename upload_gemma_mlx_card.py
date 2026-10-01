"""
Карточки для MLX-версий дообученной Gemma (заменяет пустую заготовку mlx-vlm).
Меняется только README.md — веса не трогаются.

  python upload_gemma_mlx_card.py            # обе: 4-bit и 8-bit
  python upload_gemma_mlx_card.py --bits 4

Цифры — в QUANT_RESULTS / SPEED; после изменения запусти снова.
"""
import argparse

from huggingface_hub import HfApi

BASE = "google/gemma-4-E4B-it"
ADAPTER = "Agisight/tyv-gemma4-e4b-lora"
NLLB = "Agisight/nllb-rus-tyv-v3-dict_16.5k"
DATASET = "Agisight/tyv-rus-200k"
GITHUB = "https://github.com/Agisight/tyv-translation-mlx"

SIZES = {4: ("4.9 GB", "5.2"), 8: ("8.4 GB", "9.0")}
SPEED = {4: "0.53 s per sentence (median), 5.1 GB peak memory",
         8: "0.89 s per sentence (median), 8.9 GB peak memory"}

# chrF++ ru→tyv / tyv→ru для квантизованных версий (MacBook Air M4, evaluate_gemma_mlx.py)
QUANT_RESULTS = {
    8: ("50.3", "50.9", "full clean test, 1,369 pairs"),
    4: ("48.2", "44.3", "first 100 pairs only; bf16 on the same pairs: 51.1 / 49.6"),
}


def card(bits: int) -> str:
    size, bpw = SIZES[bits]
    other = 8 if bits == 4 else 4
    r = QUANT_RESULTS[bits]
    quant_line = f"| This {bits}-bit MLX model ({r[2]}) | {r[0]} | {r[1]} |"
    verdict = ("**Recommended MLX version.** On the full clean test it matches the bf16 model "
               "(50.3 / 50.9 vs. 50.5 / 50.9) and stays above NLLB v3 (49.2 / 49.1)."
               if bits == 8 else
               "**Not recommended.** 4-bit quantization noticeably hurts this model: on the first 100 test "
               "pairs it loses 2.9 / 5.3 chrF++ against bf16, while the 8-bit version is lossless. "
               "Use [Agisight/tyv-gemma4-e4b-mlx-8bit](https://huggingface.co/Agisight/tyv-gemma4-e4b-mlx-8bit) "
               "unless memory is the hard limit. A more careful 4-bit quantization is planned.")
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
- Speed on a MacBook Air M4 (16 GB): {SPEED[bits]}.
- Other variant: [Agisight/tyv-gemma4-e4b-mlx-{other}bit](https://huggingface.co/Agisight/tyv-gemma4-e4b-mlx-{other}bit).

## Results

Clean held-out test set, 1,369 pairs (NLLB v3 test split of `{DATASET}` with training
duplicates removed). chrF++, greedy decoding.

| Model | ru→tyv | tyv→ru |
|---|---|---|
| NLLB v3 ([{NLLB}](https://huggingface.co/{NLLB})) | 49.2 | 49.1 |
| Fine-tuned Gemma 4 E4B, bf16 (adapter) | **50.5** | **50.9** |
{quant_line}

{verdict}

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
