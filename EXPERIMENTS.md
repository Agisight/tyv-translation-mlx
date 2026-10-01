# Experiment log

> Русская версия: [EXPERIMENTS.ru.md](EXPERIMENTS.ru.md). Both files are kept in sync.

## Status

- [x] Environment: MacBook Air M4, 16 GB; Python 3.12.14 (Homebrew) in `.venv`; mlx-lm 0.31.3; MLX sees the GPU; torch 2.14 (MPS) for NLLB evaluation
- [x] Git repository, `.gitignore`, `AGENTS.md`
- [x] `Qwen/Qwen3-1.7B` quantized to 4 bits → `models/qwen3-1.7b-4bit` (948 MB; 4.501 bits/weight; ~98 tok/s generation, ~1 GB peak memory)
- [x] Qwen baseline without fine-tuning
- [x] First Qwen training run (3000 steps: 500 + 2500 after resuming)
- [x] Qwen evaluation after training (old random test set)
- [x] Recovered the real NLLB v3 test set: `filter → shuffle(seed=42) → test [2000:4000]`; model-card metrics reproduced exactly
- [x] Clean test set without training duplicates (1,369 pairs) — the reference for all comparisons
- [x] Data rebuilt on the NLLB split: 272,704 training pairs after cleaning (60,000 examples for the Mac runs); validation = NLLB dev split
- [x] **Task 1: NLLB v3 → MLX.** Own M2M100 implementation with KV cache (`nllb_mlx.py`); float32 matches PyTorch (100/100 identical translations, 49.2 / 49.1 on the full clean test); int8 lossless (400 MB); int4 −0.7 / −1.0 (224 MB)
- [x] MLX int8 published: https://huggingface.co/Agisight/nllb-rus-tyv-v3-mlx-q8 (weights, `nllb_mlx.py`, model card with metrics; download and translation verified from a clean folder)
- [x] MLX int4 published: https://huggingface.co/Agisight/nllb-rus-tyv-v3-mlx-q4 (for low-memory devices, −1 chrF++)
- [ ] Native-speaker review of int4 (`human_eval_q4.csv`, 50 of the 1,429 sentences where int4 ≠ f32; delimiter `;`). Rated 11/50: 2 — 27%, 1 — 36%, 0 — 36%. **TODO:** add an "f32 rating" column (2/1/0) and rate f32 on the same rows → `python count_human_eval.py` shows how often int4 is worse than f32. Do not rerun `make_human_eval.py` — it overwrites the ratings
- [x] Single-sentence benchmarks (`bench_nllb_mlx.py`, 30 sentences per direction): f32 0.19 / 0.12 s, int8 0.07 / 0.06 s, int4 0.05 / 0.03 s (median)
- [ ] Mixed-precision int4 (embeddings and output layer at 8 bits) — an attempt to recover quality
- [ ] **Task 2: NLLB v3 → Core AI** (iOS/macOS 27): export from PyTorch, Xcode report of which ops run on the Neural Engine; comparison with Core ML int8 and MLX
- [x] Gemma 4 E4B trained in Colab (A100 80 GB) on the full training set: 1 epoch, 8,494 steps, 7.5 h; adapter on Google Drive and on HF: https://huggingface.co/Agisight/tyv-gemma4-e4b-lora
- [x] Gemma evaluation: clean and full test sets, significance (paired bootstrap), normalized chrF++, COMET — **significantly better than NLLB v3 in both directions**
- [x] Gemma 4 E4B LoRA published: https://huggingface.co/Agisight/tyv-gemma4-e4b-lora (model card with metrics, significance and code)
- [x] Gemma → MLX 4 and 8 bits (merge and `mlx-vlm` conversion in Colab, cell 11): Agisight/tyv-gemma4-e4b-mlx-4bit (4.9 GB), -8bit (8.4 GB) — private until evaluated
- [x] Gemma MLX runs on a MacBook Air M4: 8-bit — 0.90 s per sentence, 8.7 GB peak; 4-bit — 0.53 s, 5.1 GB
- [x] First 100 pairs: 8-bit ≈ bf16 (51.0 / 50.5 vs. 51.1 / 49.6); 4-bit 48.2 / 44.3 — losses come from quantization, not from mlx-vlm
- [x] Full clean-test run of Gemma MLX 8-bit: **50.3 / 50.9** (bf16: 50.5 / 50.9) — lossless; 0.89 s per sentence, 8.9 GB peak
- [x] Gemma MLX model cards updated with results; 8-bit published as the recommended MLX version, 4-bit marked not recommended
- [x] **Gemma vocabulary pruning + text-only** (`colab/prune_gemma4_vocab_colab.ipynb`): vocabulary 262,144 → 22,755 tokens (8.7%), vision/audio removed, 4.28B parameters, bf16 8.0 GB; clean test **50.5 / 50.8** (full model 50.5 / 50.9) — lossless. Private: Agisight/tyv-gemma4-e4b-pruned (bf16), -pruned-mlx-8bit (4.3 GB), -pruned-mlx-4bit (2.3 GB)
- [x] Pruned MLX 4-bit on the Mac: 2.5 GB peak memory (iPhone-sized), but first 100 pairs 47.9 / 43.3 — the same 4-bit loss as before
- [x] Pruned MLX 8-bit on the Mac (first 100 pairs): 50.3 / 50.0, 4.6 GB peak — half the memory of the unpruned 8-bit with the same quality
- [x] Quantization sweep of the pruned model (first 100 pairs): 4-bit (group 64 / 32), mixed 4/6 (group 64 / 32), 5-bit, 6-bit, 8-bit — quality holds from 5 bits up and drops below
- [x] Pruned **5-bit**, full clean test: **49.7 / 49.9**, 2.7 GB, 3.3 GB peak, 0.93 s per sentence. vs. NLLB: +0.5 (p = 0.14, n.s.) / +0.8 (p = 0.045). vs. full Gemma: −0.8 (p = 0.004) / −1.0 (p = 0.001) — a real loss
- [x] Pruned **6-bit**, full clean test: **50.2 / 50.8**, 3.2 GB, 3.8 GB peak, 0.79 s per sentence. vs. NLLB: significantly better (p = 0.029 / 0.002). vs. full Gemma: no significant difference (p = 0.12 / 0.29) — **the on-device version**
- [ ] Publish pruned 6-bit with a model card (`upload_gemma_pruned.py`)
- [ ] Run on iPhone (8 GB RAM devices; MLX Swift)
- [x] Project on GitHub: https://github.com/Agisight/tyv-translation-mlx (MIT; English/Russian README; the notebook downloads `prepare_data.py` itself; HF model cards link to it)
- [x] Reproducibility: dataset revision pinned (`DATASET_REVISION = 6d79b8ac…`), exact library versions in `requirements.lock`; the test set is unchanged (md5 of `eval_pairs.jsonl` = 3f4962de…)
- [x] Qwen experiment moved to `legacy/qwen/`
- [x] **Mac app** (`app/`, SwiftUI + mlx-swift-lm): the pruned 6-bit model runs offline in a native app, ~0.9 s per sentence after warm-up; translations correct (native speaker)
- [ ] Run on iPhone

## Results

All comparisons use the **clean test set (1,369 pairs)** unless noted otherwise.

| Date | Model | Settings | Test | It/sec | Val loss | chrF++ ru→tyv | chrF++ tyv→ru | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-09-30 | NLLB-200-distilled-600M v3 (16.5K vocabulary) | model card | NLLB test, 1,999 pairs | — | — | 48.5 | 48.1 | reproduced exactly (BLEU 19.8 / 23.2) |
| 2026-09-30 | NLLB-200-distilled-600M v3 (16.5K vocabulary) | model card | **clean test, 1,369 pairs** | — | — | **49.2** | **49.1** | **reference for all comparisons** (BLEU 19.8 / 23.5) |
| 2026-09-30 | NLLB v3 → MLX f32 | `nllb_mlx.py`, greedy, KV cache | clean test | — | — | 49.2 | 49.1 | matches PyTorch to the decimal (BLEU 19.8 / 23.5); 1,413 MB; sentence: median 0.19 / 0.12 s (ru→tyv / tyv→ru), 1,491 MB peak |
| 2026-09-30 | NLLB v3 → MLX int8 | group 64 | clean test | — | — | 49.3 | 48.8 | lossless (BLEU 19.9 / 23.4); 400 MB; sentence: median 0.07 / 0.06 s, 596 MB peak — **app variant** |
| 2026-09-30 | NLLB v3 → MLX int4 | group 64 | clean test | — | — | 48.5 | 48.1 | −0.7 / −1.0 (BLEU 19.3 / 22.2); 224 MB; sentence: median 0.05 / 0.03 s, 420 MB peak |
| 2026-09-30 | **Gemma 4 E4B + LoRA** (Colab A100) | r 32, α 64, lr 2e-4, bs 64, seq 256, 1 epoch (8,494 steps, 7.5 h), full train 543K | **clean test** | 0.31 | 0.851 | **50.5** | **50.9** | **significantly better than NLLB** (p = 0.009 / 0.001); BLEU 20.9 / 26.2; norm. chrF++ 49.9 / 54.6; COMET tyv→ru 0.827 |
| 2026-09-30 | Gemma 4 E4B + LoRA | same | full NLLB test, 1,999 pairs | — | — | 49.7 | 49.9 | vs. NLLB 48.5 / 48.1 on the same test; BLEU 21.1 / 26.0 |
| 2026-09-30 | **Gemma 4 E4B + LoRA → MLX 8-bit** | mlx-vlm, 9.0 bits/weight, 8.4 GB | **clean test, 1,369 pairs** | — | — | **50.3** | **50.9** | lossless vs. bf16 (50.5 / 50.9); BLEU 21.0 / 26.2; 0.89 s per sentence (median), 8.9 GB peak, MacBook Air M4 — **recommended MLX version** |
| 2026-10-01 | **Gemma 4 E4B + LoRA, pruned vocabulary, text-only** (bf16, Colab) | 22,755 tokens, 4.28B params, 8.0 GB | **clean test, 1,369 pairs** | — | — | **50.5** | **50.8** | lossless vs. full (50.5 / 50.9); BLEU 20.9 / 25.9; 85% / 88% of translations identical to the full model |
| 2026-10-01 | **Pruned → MLX 5-bit** (mlx-lm) | 5.5 bits/weight, 2.7 GB | **clean test, 1,369 pairs** | — | — | **49.7** | **49.9** | BLEU 20.2 / 24.6; vs. NLLB p = 0.14 / 0.045; vs. full Gemma p = 0.004 / 0.001; 0.93 s per sentence, 3.3 GB peak |
| 2026-10-01 | Pruned → MLX 8-bit | 8.5 bits/weight, 4.3 GB | clean test, first 100 pairs | — | — | 50.3 | 50.0 | ≈ bf16 (51.1 / 49.6); 4.6 GB peak |
| 2026-10-01 | **Pruned → MLX 6-bit** | 6.5 bits/weight, 3.2 GB | **clean test, 1,369 pairs** | — | — | **50.2** | **50.8** | BLEU 20.7 / 25.9; vs. NLLB p = 0.029 / 0.002; vs. full Gemma p = 0.12 / 0.29 (n.s.); 0.79 s per sentence, 3.8 GB peak — **on-device version** |
| 2026-10-01 | Pruned → MLX 6-bit | 6.5 bits/weight, 3.2 GB | clean test, first 100 pairs | — | — | 49.7 | 49.8 | 3.6 GB peak (full test above) |
| 2026-10-01 | Pruned → MLX 5-bit | 5.5 bits/weight, 2.7 GB | clean test, first 100 pairs | — | — | 50.6 | 49.2 | 3.1 GB peak (full test above) |
| 2026-10-01 | Pruned → MLX mixed 4/6, group 32 | 5.27 bits/weight, 2.6 GB | clean test, first 100 pairs | — | — | 49.8 | 48.1 | 2.9 GB peak |
| 2026-10-01 | Pruned → MLX mixed 4/6 | 4.77 bits/weight, 2.4 GB | clean test, first 100 pairs | — | — | 50.7 | 45.2 | 2.7 GB peak |
| 2026-10-01 | Pruned → MLX 4-bit, group 32 | 5.0 bits/weight, 2.5 GB | clean test, first 100 pairs | — | — | 48.9 | 47.3 | 2.8 GB peak |
| 2026-10-01 | Pruned → MLX 4-bit (mlx-lm, group 64) | 4.5 bits/weight, 2.3 GB | clean test, **first 100 pairs** | — | — | 47.9 | 43.3 | −3.2 / −6.3 vs. bf16 (51.1 / 49.6); 0.94 s per sentence, **2.5 GB peak** |
| 2026-09-30 | Gemma 4 E4B + LoRA → MLX 4-bit | mlx-vlm, 5.2 bits/weight, 4.9 GB | clean test, **first 100 pairs** | — | — | 48.2 | 44.3 | −2.9 / −5.3 vs. bf16; 0.53 s per sentence, 5.1 GB; output length same as bf16 (no looping) |
| 2026-09-29 | Qwen3-1.7B 4-bit, no fine-tuning | zero-shot, prompt `ru→tyv:` | old random test, n=100 | — | — | 8.4 | 4.5 | does not understand the task, answers like a chatbot |
| 2026-09-29 | Qwen3-1.7B 4-bit + LoRA | layers 8, bs 8, seq 256, lr 1e-4, 3000 it (500 + 2500 resumed) | old random test, n=200 | ~0.27 | 1.98 | 12.3 | 14.6 | pipeline works; the model understands the task but translates poorly |

## Notes

- **2026-09-29.** The system Python 3.9 is too old → Python 3.12 from Homebrew.
- **2026-09-29.** `mlx_lm.convert` failed while saving with `IncompleteSnapshotError` (the new `huggingface_hub` requires a full snapshot). Fix: `hf download Qwen/Qwen3-1.7B` before converting.
- **2026-09-29.** To free disk space: removed the Ollama model `qwen3:14b` (9.3 GB) and the original Qwen3-1.7B from the HF cache.
- **2026-09-29.** With `max_seq_length: 128`, some translations were truncated (Tuvan Cyrillic takes up to ~290 tokens per 300 characters) → 256. `grad_checkpoint` off: memory is sufficient (5.8 GB peak); speed is bound by the Air's GPU (~0.27–0.37 steps/s).
- **2026-09-29.** Qwen training was interrupted by a low battery and resumed with `--resume-adapter-file`. On battery the speed drops to ~0.2 steps/s. Checkpoint numbering restarts after resuming.
- **2026-09-29.** Qwen3-1.7B + LoRA on the Air saw only ~4% of the dataset; quality is low. The bottleneck is the Air's GPU. Next step: training in Colab.
- **2026-09-30.** `eval_for_paper.ipynb` split the data with `train_test_split` — a different split that overlaps NLLB's training data; its scores are inflated (52–54). **Do not use.**
- **2026-09-30.** The real NLLB v3 test set comes from `train_nllb_tyvan_v3_experimental.py`: `filter → shuffle(seed=42) → test [2000:4000]`, with the same preprocessing (Moses + NFKC), untruncated input and `max_new = 32 + 3 × input length`. Model-card metrics reproduced exactly on the Mac (48.5 / 48.1) — transformers 5, float32 and MPS give the same numbers as Colab.
- **2026-09-30.** 630 of the 1,999 test pairs duplicate training data. NLLB is slightly better on the clean test, so the duplicates did not inflate its score. The clean test (1,369 pairs) is the reference.
- Qwen results use the old random test, and the Qwen adapter was trained on the old split — for the paper, Qwen must be retrained and evaluated on the clean test. Code is in `legacy/qwen/`.
- **2026-09-30.** NLLB is an encoder-decoder, which `mlx_lm` does not support → own M2M100 implementation in MLX (`nllb_mlx.py`): pre-LayerNorm, shared embeddings × √d_model, sinusoidal positions offset by the padding index, decoder start `[eos, language tag]`, KV cache. float32 produces exactly the same translations as PyTorch.
- **2026-09-30.** Single-sentence benchmarks (`bench_nllb_mlx.py`): 30 random test sentences per direction, after warm-up, MacBook Air M4, only the terminal open, on power. Median time per sentence:

  | Variant | Size | ru→tyv | tyv→ru | Speed (median) | Peak memory |
  |---|---|---|---|---|---|
  | f32 | 1,413 MB | 0.19 s | 0.12 s | 79 / 74 tok/s | 1,491 MB |
  | int8 | 400 MB | 0.07 s | 0.06 s | 248 / 174 tok/s | 596 MB |
  | int4 | 224 MB | 0.05 s | 0.03 s | 343 / 304 tok/s | 420 MB |

  int8 is ~3× faster than f32, int4 ~4–5×. Batched throughput (300–600 tok/s) is unstable between runs — do not use it in the paper.
- **2026-09-30.** int4 translates differently from f32 in 1,429 of 2,738 sentences (52%: ru→tyv 743, tyv→ru 686), yet chrF++ drops only ~1 point — most differences are minor. Native-speaker review will show how many are acceptable.
- **2026-09-30.** Peak memory with batch 32 (2.8–3.8 GB) is mostly activations for long sentences, not weights.
- **2026-09-30.** On a test sentence, int4 produced «Даарта мен Кызылче ада-иемче чоруур мен.» instead of «Даарта Кызылче ада-иемче чоруур мен.» — a native speaker judges both correct (the explicit «мен» adds emphasis). Part of int4's chrF++ drop may be acceptable variation rather than errors — to be checked by the human review.
- **2026-09-30.** During a 100-example int8 evaluation the system ran out of memory (Chrome, Xcode and ChatGPT open) → speed numbers from that run are invalid. Keep only the terminal open for benchmarks.
- **2026-09-30.** Human review: some errors come from orthographic noise in the source (Ң typed as Н: «калчан» instead of «калчаң»). Paper candidate: normalizing Н/Ң, О/Ө, У/Ү on input; link to keyboard layouts. Mark such cases in the review comments.
- **2026-09-30. Gemma 4 E4B vs. NLLB v3 — summary.** Clean test (1,369 pairs):

  | Metric | NLLB v3 | Gemma 4 E4B | Δ | p (paired bootstrap) |
  |---|---|---|---|---|
  | chrF++ ru→tyv | 49.2 | **50.5** | +1.3 | 0.009 * |
  | chrF++ tyv→ru | 49.1 | **50.9** | +1.8 | 0.001 * |
  | BLEU ru→tyv | 19.8 | **20.9** | +1.1 | 0.042 * |
  | BLEU tyv→ru | 23.5 | **26.2** | +2.7 | 0.002 * |
  | norm. chrF++ ru→tyv | 49.0 | **49.9** | +0.9 | — |
  | norm. chrF++ tyv→ru | 53.1 | **54.6** | +1.5 | — |
  | COMET tyv→ru | 0.806 | **0.827** | +0.021 | — |

  Full test (1,999 pairs): Gemma 49.7 / 49.9 vs. NLLB 48.5 / 48.1. All metrics agree; the gap remains after normalizing word forms and gender. Native speaker: ru→tyv translations are correct, differences from the reference are wording. tyv→ru errors are sometimes lexical («хевис» → «шалаш» instead of «ковёр»); gender («она» instead of «он») is not an error — Tuvan has no grammatical gender.
- **2026-09-30.** Gemma training: loss plateaued after ~7,000 steps (0.861 → 0.852 → 0.851); a second epoch is unlikely to help much. Speed is GPU-bound (~21 examples/s at batch 32 and 64) — a larger batch does not speed it up. Colab Pro disconnects idle sessions even during computation — interact with the notebook; adapter and checkpoints are on Google Drive.
- **2026-09-30.** The Colab notebook detects its mode: adapter exists → evaluation only; none → training; interrupted → resume from checkpoint. Always "Run all". Fixes: remove the old torchao; LoRA only on the language model's Linear layers (Gemma 4's vision and audio towers use Gemma4ClippableLinear); warmup_steps instead of warmup_ratio; dataset_num_proc=8 (untested).
- **2026-09-30.** Gemma → MLX: adapter merge and `mlx_vlm.convert` run in Colab (needs ~50 GB RAM). The current mlx-vlm keeps Gemma 4's PLE layers at higher precision (hence 5.2 bits/weight for "4-bit") — the old bug that produced garbage when PLE layers were quantized did not appear. A test generation in Colab is useless: MLX on Linux is CPU-only and an 8B model hangs for tens of minutes; test on the Mac. Interrupting the conversion leaves an incomplete folder (8-bit: 1.5 GB instead of 8.4) — cell 11 now writes a `.converted` marker and rebuilds incomplete folders.
- **2026-09-30.** Gemma MLX, first 100 clean-test pairs: bf16 51.1 / 49.6, MLX 8-bit 51.0 / 50.5 (0.90 s per sentence, 8.7 GB), MLX 4-bit 48.2 / 44.3 (0.53 s, 5.1 GB). The mlx-vlm prompt is identical to training (`<bos><|turn>user…<turn|>\n<|turn>model\n`, no double `<bos>`), no looping. Conclusion: the mlx-vlm implementation is correct; the losses come from 4-bit quantization. Gemma is more sensitive to 4 bits than NLLB (−1). For the Mac — 8-bit.
- **2026-09-30.** Example of acceptable variation (native speaker): «ада-иемге баар мен» (-ге "to them", баар "I'll come/go") vs. «ада-иемче чоруур мен» (-че "towards", чоруур "to travel"). Both correctly render «поеду к родителям» ("I'll go to my parents"); chrF++ counts the difference as an error.
- **2026-09-30.** Gemma MLX 8-bit, full clean test (MacBook Air M4, one sentence at a time): chrF++ 50.3 / 50.9, BLEU 21.0 / 26.2 — the same as bf16 in Colab (50.5 / 50.9, BLEU 20.9 / 26.2). Median 0.89 s, mean 1.34 s per sentence (n = 2,538), 8.9 GB peak. 8-bit is the recommended MLX version; 4-bit is not.
- **2026-10-01.** Gemma 4 E4B stores its 262,144-token vocabulary in two tables: token embeddings (262,144 × 2,560 ≈ 0.67B parameters) and per-layer embeddings, PLE (262,144 × 42 layers × 256 ≈ 2.8B) — about 44% of all weights. Pruning to the 22,755 tokens used in the **training** data (plus special, byte-fallback and single-character tokens) and dropping the vision/audio towers halves the model: 4.28B parameters, 8.0 GB in bf16, with the same clean-test quality (50.5 / 50.8). Test tokens outside the new vocabulary are rare and fall back to byte tokens.
- **2026-10-01.** Pruning a BPE tokenizer: keep the "ancestors" of every kept token — the parts of **all** merges that produce it, not just the first one. With only the first merge, 1 of 10,476 texts ("Wi-Fi") tokenized differently; with all merges the new tokenizer matches the original exactly on the whole test set and 5,000 training texts. The notebook checks this and stops on any mismatch.
- **2026-10-01.** Pruned 4-bit MLX (mlx-lm quantizes everything uniformly, 4.5 bits/weight): 2.5 GB peak memory — iPhone-sized — but 47.9 / 43.3 on the first 100 pairs, the same loss as the unpruned 4-bit version (48.2 / 44.3). The loss comes from 4-bit quantization, not from pruning. Next: group size 32 and mixed 4/6-bit quantization.
- **2026-10-01.** Quantization sweep of the pruned model, first 100 clean-test pairs (bf16 on the same pairs: 51.1 / 49.6):

  | Variant | Size | Peak memory | ru→tyv | tyv→ru |
  |---|---|---|---|---|
  | 8-bit | 4.3 GB | 4.6 GB | 50.3 | 50.0 |
  | 6-bit | 3.2 GB | 3.6 GB | 49.7 | 49.8 |
  | 5-bit | 2.7 GB | 3.1 GB | 50.6 | 49.2 |
  | mixed 4/6, group 32 | 2.6 GB | 2.9 GB | 49.8 | 48.1 |
  | mixed 4/6 | 2.4 GB | 2.7 GB | 50.7 | 45.2 |
  | 4-bit, group 32 | 2.5 GB | 2.8 GB | 48.9 | 47.3 |
  | 4-bit, group 64 | 2.3 GB | 2.5 GB | 47.9 | 43.3 |

  Quality holds from ~5 bits up and drops below; translation **into Russian** is consistently more sensitive to quantization than into Tuvan. Mixed 4/6 fixes ru→tyv, a smaller group helps tyv→ru more. 100 pairs are noisy (±1–2 points) — the full test decides.
- **2026-10-01.** Pruned 5-bit on the full clean test: 49.7 / 49.9 — −0.8 / −1.0 vs. full Gemma, a significant loss (p = 0.004 / 0.001); vs. NLLB +0.5 (p = 0.14, not significant) / +0.8 (p = 0.045). On 100 pairs it looked lossless — only the full test shows the real gap. Summary so far: pruning is free, 8 bits are lossless, 5 bits cost ~1 point, below 5 bits quality collapses.
- **2026-10-01.** Native speaker on «оңгарже шымны берди»: «нырнул» / «погрузился» is exact, «упал» only approximate. 6- and 8-bit produced «нырнул», 5-bit and full Gemma «упала / упал». chrF++ barely separates these — a reason to add a native-speaker check when choosing between 5 and 6 bits.
- **2026-10-01. On-device version: pruned 6-bit.** Full clean test: 50.2 / 50.8 (BLEU 20.7 / 25.9) — no significant difference from the full bf16 model (p = 0.12 / 0.29) and significantly better than NLLB v3 in both directions (p = 0.029 / 0.002). 3.2 GB on disk, 3.8 GB peak, 0.79 s per sentence on a MacBook Air M4. From the 16 GB fine-tuned model to 3.2 GB — 5× smaller — at the same quality. 6 bits is the threshold: 5 bits loses ~1 point significantly.
- **2026-10-01.** SwiftUI app (`app/`, XcodeGen project, macOS + iOS): mlx-swift-lm supports `gemma4_text` and reads the vocabulary sizes from `config.json`, so the pruned model (22,755 tokens) loads unchanged; `num_experts: null` decodes as absent. The prompt is built by hand exactly as in training (`<bos><|turn>user\n…<turn|>\n<|turn>model\n`), greedy decoding, `32 + 3 × input length` tokens. Built on the first try; Xcode asks to download the Metal Toolchain (MLX compiles its shaders). First translation 4.2 s (warm-up, Debug build), then ~0.9 s per sentence. Native speaker: translations correct. «Эртен Кызылче ада-иемге баар мен» uses «эртен» instead of «даарта» — synonyms, both mean "tomorrow" ("in the morning" is «эртен / эртежик / эртенинде»); chrF++ would count it as an error.
