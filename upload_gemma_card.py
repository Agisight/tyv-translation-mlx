"""
Заменяет автоматическую карточку репозитория Agisight/tyv-gemma4-e4b-lora на нормальную.
Меняется только README.md — веса адаптера не трогаются.

  python upload_gemma_card.py
"""
from huggingface_hub import HfApi

REPO = "Agisight/tyv-gemma4-e4b-lora"
BASE = "google/gemma-4-E4B-it"
DATASET = "Agisight/tyv-rus-200k"
NLLB = "Agisight/nllb-rus-tyv-v3-dict_16.5k"

CARD = f"""---
library_name: peft
base_model: {BASE}
license: apache-2.0
language:
- ru
- tyv
pipeline_tag: translation
datasets:
- {DATASET}
tags:
- lora
- peft
- gemma
- gemma4
- translation
- low-resource
- tuvan
---

# Gemma 4 E4B — Russian ↔ Tuvan translation (LoRA)

LoRA adapter for [{BASE}](https://huggingface.co/{BASE}) fine-tuned for bidirectional
Russian ↔ Tuvan translation. Tuvan is a low-resource Turkic language with roughly 280,000 speakers.

On the clean held-out test set this adapter **significantly outperforms** the specialized
translation model [{NLLB}](https://huggingface.co/{NLLB}) (NLLB-200-distilled-600M fine-tuned
on the same data) in both directions and on every metric.

## Results

**Clean test set — 1,369 pairs.** The held-out test split used for NLLB v3
(`shuffle(seed=42)`, pairs 2000–4000 of `{DATASET}`), with pairs whose Tuvan or Russian side also
appears in training removed. Greedy decoding, same length limit as NLLB
(`32 + 3 × input length` new tokens).

| Metric | NLLB v3 | **This adapter** | Δ | p (paired bootstrap) |
|---|---|---|---|---|
| chrF++ ru→tyv | 49.2 | **50.5** | +1.3 | 0.009 |
| chrF++ tyv→ru | 49.1 | **50.9** | +1.8 | 0.001 |
| BLEU ru→tyv | 19.8 | **20.9** | +1.1 | 0.042 |
| BLEU tyv→ru | 23.5 | **26.2** | +2.7 | 0.002 |
| Normalized chrF++ ru→tyv ¹ | 49.0 | **49.9** | +0.9 | — |
| Normalized chrF++ tyv→ru ¹ | 53.1 | **54.6** | +1.5 | — |
| COMET tyv→ru ² | 0.806 | **0.827** | +0.021 | — |

**Full NLLB test set — 1,999 pairs** (comparable with the NLLB v3 model card):
chrF++ **49.7 / 49.9** (ru→tyv / tyv→ru) vs. NLLB v3 48.5 / 48.1.

¹ Normalization forgives acceptable variation: for Russian output, words are lemmatized and
3rd-person pronouns merged (Tuvan has no grammatical gender, so «он»/«она» is ambiguous);
for Tuvan output, subject pronouns are dropped (person is marked on the verb).
² `Unbabel/wmt22-comet-da`; reported for tyv→ru only, since COMET's encoder does not cover Tuvan.

## Prompt format

The adapter was trained on single-turn chat messages with a short prefix:

- Russian → Tuvan: `ru→tyv: <Russian text>`
- Tuvan → Russian: `tyv→ru: <Tuvan text>`

Use exactly this format; other instructions were not seen in training.

## Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_id, adapter_id = "{BASE}", "{REPO}"
tok = AutoTokenizer.from_pretrained(adapter_id)
model = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_id).eval()

def translate(text, direction="ru→tyv"):
    prompt = tok.apply_chat_template([{{"role": "user", "content": f"{{direction}}: {{text}}"}}],
                                     add_generation_prompt=True, tokenize=False)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(**enc, max_new_tokens=32 + 3 * enc.input_ids.shape[1], do_sample=False)
    return tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()

print(translate("Завтра я поеду в Кызыл к родителям."))
print(translate("Экии!", "tyv→ru"))
```

## Training

- **Data:** `{DATASET}`, NLLB v3 split; 272,704 training pairs after cleaning, used in both
  directions — 543,616 examples. Pairs longer than 300 characters were dropped.
- **Method:** LoRA (r = 32, α = 64, dropout 0.05) on all attention and MLP projections of the
  language model only — 258 layers, 69.8M trainable parameters (0.87%). Vision and audio towers
  are untouched. Loss on the translation (completion) tokens only.
- **Hyperparameters:** 1 epoch (8,494 steps), batch size 64, learning rate 2e-4 with cosine
  schedule and 3% warmup, max length 256 tokens, bf16.
- **Hardware:** 1× NVIDIA A100 80 GB (Google Colab), ~7.5 hours.
- **Final validation loss:** 0.851; validation token accuracy 78.8%. Loss plateaued after ~7,000 steps.

## Limitations

- Trained and evaluated on a single test domain; quality on other genres (spoken language,
  technical texts) is untested.
- Tuvan → Russian sometimes picks a wrong word sense (e.g. «хевис» → «шалаш» instead of «ковёр»)
  and has to guess gender when the Tuvan source does not mark it.
- Tuvan input that uses Н/О/У instead of Ң/Ө/Ү (common when typed without a Tuvan keyboard)
  can lead to mistranslations.
- The adapter inherits the limitations and license terms of the base model.

## Code

Training notebook (Colab), data preparation, and evaluation scripts:
[Agisight/tyv-translation-mlx](https://github.com/Agisight/tyv-translation-mlx).

## Credits

- Base model: Google, Gemma 4 E4B.
- Data, fine-tuning, and evaluation: Ali Kuzhuget.
"""

api = HfApi()
api.upload_file(path_or_fileobj=CARD.encode("utf-8"), path_in_repo="README.md",
                repo_id=REPO, commit_message="Model card: description, results, usage, training details")
print(f"Карточка обновлена: https://huggingface.co/{REPO}")
