"""
Таблица для ручной оценки: случаи, где квантизованная модель перевела иначе, чем f32.

  python make_human_eval.py                       # int4 против f32, 50 случаев
  python make_human_eval.py --variant q8 --n 30

Результат: human_eval_<variant>.csv — открыть в Numbers/Excel и заполнить колонку оценки:
2 = верно (другая правильная формулировка), 1 = мелкая ошибка, 0 = неверно.
"""
import argparse
import csv
import json
import random


def load(path):
    return {(r["direction"], r["src"]): r for r in map(json.loads, open(path, encoding="utf-8"))}


ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="q4")
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

base = load("eval_results_nllb-v3-mlx-f32.jsonl")
var = load(f"eval_results_nllb-v3-mlx-{args.variant}.jsonl")
diff = [(base[k], var[k]) for k in var if k in base and base[k]["hyp"] != var[k]["hyp"]]
by_dir = {d: sum(1 for a, _ in diff if a["direction"] == d) for d in ("ru-tyv", "tyv-ru")}
random.seed(args.seed)
sample = random.sample(diff, min(args.n, len(diff)))

out = f"human_eval_{args.variant}.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["direction", "source", "reference", "f32", args.variant,
                "оценка: 2=верно, 1=мелкая ошибка, 0=неверно", "комментарий"])
    for a, b in sample:
        w.writerow([a["direction"], a["src"], a["ref"], a["hyp"], b["hyp"], "", ""])

print(f"Отличий {args.variant} от f32: {len(diff)} из {len(var)} "
      f"(ru→tyv {by_dir['ru-tyv']}, tyv→ru {by_dir['tyv-ru']})")
print(f"В таблицу взято {len(sample)} → {out}")
