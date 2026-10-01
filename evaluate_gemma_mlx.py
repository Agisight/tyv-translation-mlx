"""
Оценка MLX-версии дообученной Gemma (mlx-vlm) на чистом тесте — те же пары и метрика,
что для NLLB и для Gemma bf16 в Colab. Генерация по одной фразе (как в приложении),
greedy, max_new = min(256, 32 + 3 × длина промпта).

Можно прерывать: уже переведённые пары сохраняются и при повторном запуске пропускаются.

  python evaluate_gemma_mlx.py --model models/tyv-gemma4-e4b-mlx-4bit --limit 100   # быстрая проверка
  python evaluate_gemma_mlx.py --model models/tyv-gemma4-e4b-mlx-4bit               # весь тест (~1–1.5 ч)
"""
import argparse
import json
import re
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
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

    model, processor = load(args.model)
    tok = getattr(processor, "tokenizer", processor)

    rows = [json.loads(l) for l in open(args.pairs, encoding="utf-8")]
    chrf, bleu = CHRF(word_order=2), BLEU()
    times = []
    with out_path.open("a", encoding="utf-8") as f:
        for d in ("ru-tyv", "tyv-ru"):
            items = [r for r in rows if r["direction"] == d][: args.limit]
            for i, r in enumerate(items, 1):
                key = (d, r["src"])
                if key not in done:
                    prompt = apply_chat_template(processor, model.config, PROMPTS[d].format(src=r["src"]), num_images=0)
                    max_new = min(256, 32 + 3 * len(tok.encode(prompt)))
                    t0 = time.perf_counter()
                    res = generate(model, processor, prompt, max_tokens=max_new, temperature=0.0, verbose=False)
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
    print("Сравнение: NLLB v3 49.2 / 49.1 | Gemma bf16 (Colab) 50.5 / 50.9")


if __name__ == "__main__":
    main()
