# Russian ↔ Tuvan machine translation on Apple devices

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Agisight/tyv-translation-mlx/blob/main/colab/train_gemma4_tyv_colab.ipynb)

Code, data preparation and evaluation for offline Russian ↔ Tuvan translation. Tuvan is a
low-resource Turkic language with roughly 280,000 speakers.

- **NLLB v3 on MLX** — a from-scratch MLX implementation of the NLLB (M2M100) encoder-decoder
  with a KV cache. float32 reproduces PyTorch exactly; the 8-bit version is lossless and ~3× faster.
- **Gemma 4 E4B + LoRA** — a fine-tuned LLM that significantly outperforms the specialized NLLB v3
  model in both directions, converted to MLX for Mac.
- **A clean test set and honest evaluation** — chrF++, BLEU, COMET, paired bootstrap significance,
  and a normalized chrF++ that does not penalize gender (Tuvan has none) or word forms.

Русская пошаговая инструкция для Mac: [README.ru.md](README.ru.md). Журнал экспериментов: [EXPERIMENTS.md](EXPERIMENTS.md).

## Results

Clean test set, 1,369 pairs (see [Test set](#test-set)). Latency: median per sentence, MacBook Air M4 (16 GB).

| Model | Size | chrF++ ru→tyv | chrF++ tyv→ru | COMET tyv→ru | Latency |
|---|---|---|---|---|---|
| NLLB v3, PyTorch / MLX float32 | 1.4 GB | 49.2 | 49.1 | 0.806 | 0.19 / 0.12 s |
| NLLB v3, MLX 8-bit | 400 MB | 49.3 | 48.8 | 0.806 | 0.07 / 0.06 s |
| NLLB v3, MLX 4-bit | 224 MB | 48.5 | 48.1 | 0.800 | 0.05 / 0.03 s |
| **Gemma 4 E4B + LoRA, bf16** | 16 GB | **50.5** | **50.9** | **0.827** | — |
| Gemma 4 E4B + LoRA, MLX 4-bit | 4.9 GB | in progress | in progress | — | — |

Gemma vs. NLLB v3: p = 0.009 (ru→tyv) and p = 0.001 (tyv→ru), paired bootstrap on chrF++.
On the full 1,999-pair NLLB test set: Gemma 49.7 / 49.9 vs. NLLB 48.5 / 48.1.

## Models

| Model | Format | Use on |
|---|---|---|
| [Agisight/nllb-rus-tyv-v3-dict_16.5k](https://huggingface.co/Agisight/nllb-rus-tyv-v3-dict_16.5k) | PyTorch, float32 | any (transformers) |
| [Agisight/nllb-rus-tyv-v3-mlx-q8](https://huggingface.co/Agisight/nllb-rus-tyv-v3-mlx-q8) | MLX 8-bit | Mac (recommended) |
| [Agisight/nllb-rus-tyv-v3-mlx-q4](https://huggingface.co/Agisight/nllb-rus-tyv-v3-mlx-q4) | MLX 4-bit | Mac, low memory |
| [Agisight/tyv-gemma4-e4b-lora](https://huggingface.co/Agisight/tyv-gemma4-e4b-lora) | LoRA adapter for `google/gemma-4-E4B-it` | GPU (transformers + peft) |
| Agisight/tyv-gemma4-e4b-mlx-4bit / -8bit | MLX | Mac — released after evaluation |

## Quick start: translate

All commands run from the repository root.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**NLLB v3 on a Mac (MLX, fastest):**

```bash
hf download Agisight/nllb-rus-tyv-v3-mlx-q8 --local-dir models/nllb-v3-mlx-q8
python evaluate_nllb_mlx.py --model models/nllb-v3-mlx-q8 --text "Завтра я поеду в Кызыл к родителям."
python evaluate_nllb_mlx.py --model models/nllb-v3-mlx-q8 --dir tyv-ru --text "Экии!"
```

**Gemma on a Mac (MLX):** the prompt format is `ru→tyv: <text>` or `tyv→ru: <text>`.

```bash
mlx_vlm.generate --model Agisight/tyv-gemma4-e4b-mlx-4bit \
  --prompt "ru→tyv: Завтра я поеду в Кызыл к родителям." --max-tokens 64 --temperature 0.0
```

**Gemma on a GPU (transformers + peft):** see the code example in the
[adapter card](https://huggingface.co/Agisight/tyv-gemma4-e4b-lora).

## Reproduce from scratch

### 1. Data and test set

```bash
python prepare_data.py
```

Downloads [Agisight/tyv-rus-200k](https://huggingface.co/datasets/Agisight/tyv-rus-200k) and builds
the NLLB v3 split, the clean test set, and chat-format training files in `data/`.

### 2. NLLB v3 baseline

```bash
python prepare_data.py --keep-dup-test && python evaluate_nllb.py   # reproduces the model card: 48.5 / 48.1
python prepare_data.py && python evaluate_nllb.py                   # clean test: 49.2 / 49.1
```

### 3. NLLB v3 → MLX

```bash
python convert_nllb_mlx.py --bits 8          # also: no flag (float32), --bits 4, --dtype float16
python evaluate_nllb_mlx.py --model models/nllb-v3-mlx-q8
python bench_nllb_mlx.py --model models/nllb-v3-mlx-q8
```

### 4. Train Gemma 4 E4B + LoRA (Google Colab, A100)

Click **Open in Colab** above, choose an A100 runtime with High-RAM, and run all cells. The notebook
downloads `prepare_data.py` from this repository, trains on the full training set (1 epoch,
~7.5 hours), evaluates on the clean and full test sets, and keeps checkpoints on Google Drive so an
interrupted run resumes. Publishing (cell 10) and MLX conversion (cell 11) are off by default.

### 5. Metrics and significance

```bash
python significance.py eval_results_nllb.jsonl eval_results_gemma4.json      # paired bootstrap
python metrics_extra.py eval_results_nllb.jsonl eval_results_gemma4.json     # normalized chrF++
```

COMET needs a separate environment (see `requirements-comet.txt`):

```bash
python3.12 -m venv .venv-comet && source .venv-comet/bin/activate
pip install -r requirements-comet.txt
python comet_eval.py eval_results_nllb.jsonl eval_results_gemma4.json
```

### 6. Small LLM on a MacBook (MLX LoRA)

An early experiment: Qwen3-1.7B fine-tuned on a MacBook Air with `mlx_lm.lora` (`legacy/qwen/`). It works end to end but reaches only ~12–15 chrF++ — the laptop GPU
saw ~4% of the data. Step-by-step instructions are in [README.ru.md](README.ru.md).

## Test set

The test set is the held-out split used to train and evaluate NLLB v3: `filter → shuffle(seed=42) →
pairs 2000–4000` of the dataset. 630 of its 1,999 pairs also appear in the training data (the dataset
contains duplicates), so we report results on the **clean subset of 1,369 pairs** and, for
comparison with earlier work, on the full 1,999 pairs. Conclusions are the same on both.

## Repository

| Path | What it does |
|---|---|
| `prepare_data.py` | Data split, clean test set, chat-format training data |
| `evaluate_nllb.py` | NLLB v3 baseline in PyTorch (same settings as its training) |
| `nllb_mlx.py`, `convert_nllb_mlx.py` | NLLB (M2M100) in MLX; weight conversion and quantization |
| `evaluate_nllb_mlx.py`, `bench_nllb_mlx.py` | MLX quality evaluation; single-sentence latency and memory |
| `colab/train_gemma4_tyv_colab.ipynb` | Gemma 4 E4B LoRA training, evaluation, MLX conversion |
| `significance.py`, `metrics_extra.py`, `comet_eval.py` | Significance test, normalized chrF++, COMET |
| `make_human_eval.py`, `count_human_eval.py` | Native-speaker review of quantization differences |
| `upload_*.py` | Publishing models and model cards to Hugging Face |
| `eval_results_*` | Saved translations used for all reported metrics |

## License

Code: MIT. Models keep their own licenses: NLLB-based models CC-BY-NC-4.0 (from Meta's NLLB-200),
Gemma-based models Apache-2.0.

## Author

Ali Kuzhuget. NLLB v3 fine-tuning builds on David Dale's NLLB fine-tuning code, with his consultation.
