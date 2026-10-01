"""
Оценка MLX-версии дообученной Gemma на чистом тесте — те же пары и метрика,
что для NLLB и для Gemma bf16 в Colab. Генерация по одной фразе (как в приложении),
greedy, max_new = min(256, 32 + 3 × длина промпта).

Можно прерывать: уже переведённые пары сохраняются и при повторном запуске пропускаются.

  python evaluate_gemma_mlx.py --model models/tyv-gemma4-e4b-mlx-8bit --limit 100   # быстрая проверка
  python evaluate_gemma_mlx.py --model models/tyv-gemma4-e4b-mlx-8bit               # весь тест (~1 ч)
  python evaluate_gemma_mlx.py --model models/tyv-gemma4-e4b-pruned-mlx-4bit        # сокращённая (текстовая)

Бэкенд выбирается сам: мультимодальная модель (model_type gemma4) — mlx-vlm,
текстовая сокращённая (gemma4_text) — mlx-lm. Токенизатор берётся из папки модели.
"""
import argparse
import json
import re
import statistics
import time
from pathlib import Path

import mlx.core as mx
from sacrebleu.metrics import BLEU, CHRF

from prepare_data import PROMPTS


def peak_mem_mb() -> float:
    for fn in (getattr(mx, "get_peak_memory", None), getattr(getattr(mx, "metal", None), "get_peak_memory", None)):
        if fn:
            return fn() / 1024 / 1024
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/tyv-gemma4-e4b-mlx-4bit")
    ap.add_argument("--pairs", default="data/eval_pairs.jsonl")
    ap.add_argument("--limit", type=int, default=100000, help="примеров на направление")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    name = Path(args.model.rstrip("/")).name
    out_path = Path(args.out or f"eval_results_{name}.jsonl")
    done = {}
    if out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            r = json.loads(line)
            done[(r["direction"], r["src"])] = r
        print(f"Продолжаю: уже переведено {len(done)}")

    model_type = json.load(open(Path(args.model) / "config.json")).get("model_type", "")
    if model_type.endswith("_text"):
        from mlx_lm import generate as lm_generate, load as lm_load
        from mlx_lm.sample_utils import make_sampler
        model, tok = lm_load(args.model)
        sampler = make_sampler(temp=0.0)

        def make_prompt(text):
            return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)

        def run(prompt, max_new):
            return lm_generate(model, tok, prompt=prompt, max_tokens=max_new, sampler=sampler, verbose=False)
    else:
        from mlx_vlm import generate as vlm_generate, load as vlm_load
        from mlx_vlm.prompt_utils import apply_chat_template
        model, processor = vlm_load(args.model)
        tok = getattr(processor, "tokenizer", processor)

        def make_prompt(text):
            return apply_chat_template(processor, model.config, text, num_images=0)

        def run(prompt, max_new):
            return vlm_generate(model, processor, prompt, max_tokens=max_new, temperature=0.0, verbose=False)
    print(f"Бэкенд: {'mlx-lm' if model_type.endswith('_text') else 'mlx-vlm'} ({model_type})")

    rows = [json.loads(l) for l in open(args.pairs, encoding="utf-8")]
    chrf, bleu = CHRF(word_order=2), BLEU()
    times = []
    with out_path.open("a", encoding="utf-8") as f:
        for d in ("ru-tyv", "tyv-ru"):
            items = [r for r in rows if r["direction"] == d][: args.limit]
            for i, r in enumerate(items, 1):
                key = (d, r["src"])
                if key not in done:
                    prompt = make_prompt(PROMPTS[d].format(src=r["src"]))
                    max_new = min(256, 32 + 3 * len(tok.encode(prompt)))
                    t0 = time.perf_counter()
                    res = run(prompt, max_new)
                    times.append(time.perf_counter() - t0)
                    text = getattr(res, "text", res)
                    hyp = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
                    done[key] = {**r, "hyp": hyp}
                    f.write(json.dumps(done[key], ensure_ascii=False) + "\n")
                    f.flush()
                if i % 50 == 0 or i == len(items):
                    print(f"  {d}: {i}/{len(items)}")
            hyps = [done[(d, r["src"])]["hyp"] for r in items]
            refs = [r["ref"] for r in items]
            print(f"{d}: chrF++ = {chrf.corpus_score(hyps, [refs]).score:.1f}  "
                  f"BLEU = {bleu.corpus_score(hyps, [refs]).score:.1f}  (n={len(items)})")
            for r, h in list(zip(items, hyps))[:2]:
                print("   src:", r["src"], "\n   hyp:", h, "\n   ref:", r["ref"])
    if times:
        print(f"Время на фразу: медиана {statistics.median(times):.2f} с, среднее {statistics.mean(times):.2f} с "
              f"(n={len(times)}) | пик памяти {peak_mem_mb():.0f} МБ")
    print(f"Переводы: {out_path}")
    print("Сравнение: NLLB v3 49.2 / 49.1 | Gemma bf16 50.5 / 50.9 | MLX 8 бит 50.3 / 50.9 | сокращённая bf16 50.5 / 50.8")


if __name__ == "__main__":
    main()
