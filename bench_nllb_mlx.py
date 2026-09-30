"""
Замер скорости и памяти на ОДНУ фразу (как в приложении): среднее по N фразам теста.

  python bench_nllb_mlx.py --model models/nllb-v3-mlx-q8
  for v in f32 q8 q4; do python bench_nllb_mlx.py --model models/nllb-v3-mlx-$v; done

Перед замером: зарядка подключена, открыт только терминал.
"""
import argparse
import json
import random
import statistics
import time

import mlx.core as mx
from transformers import AutoTokenizer

from evaluate_nllb_mlx import peak_mem_mb, translate_batch
from nllb_mlx import load_model

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--pairs", default="data/eval_pairs.jsonl")
ap.add_argument("--n", type=int, default=30, help="фраз на направление")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
model = load_model(args.model)
rows = [json.loads(l) for l in open(args.pairs, encoding="utf-8")]
random.seed(args.seed)

translate_batch(model, tok, ["Привет, как дела?"], "ru-tyv")  # прогрев
translate_batch(model, tok, ["Экии!"], "tyv-ru")

name = args.model.rstrip("/").split("/")[-1]
print(f"{name}")
for d in ("ru-tyv", "tyv-ru"):
    items = random.sample([r for r in rows if r["direction"] == d], args.n)
    times, speeds = [], []
    for r in items:
        t0 = time.perf_counter()
        _, ntok = translate_batch(model, tok, [r["src"]], d)
        dt = time.perf_counter() - t0
        times.append(dt)
        speeds.append(ntok / dt)
    print(f"  {d}: время на фразу медиана {statistics.median(times):.2f} с, "
          f"среднее {statistics.mean(times):.2f} с | скорость медиана {statistics.median(speeds):.0f} ток/с (n={args.n})")
print(f"  пик памяти: {peak_mem_mb():.0f} МБ")
