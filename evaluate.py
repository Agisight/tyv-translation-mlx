"""
Считает chrF++ для дообученной модели (и для базовой, чтобы видеть прирост).

  python evaluate.py --adapter adapters             # модель + адаптер
  python evaluate.py --no-adapter                   # базовая модель без дообучения
  python evaluate.py --model models/tyv-qwen3-1.7b --no-adapter   # слитая модель
"""
import argparse
import json
from collections import defaultdict

from mlx_lm import load
from sacrebleu.metrics import CHRF

from tyv_translate import translate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen3-1.7b-4bit")
    ap.add_argument("--adapter", default="adapters")
    ap.add_argument("--no-adapter", action="store_true")
    ap.add_argument("--pairs", default="data/eval_pairs.jsonl")
    ap.add_argument("--limit", type=int, default=500, help="примеров на направление")
    ap.add_argument("--out", default="eval_results.jsonl")
    args = ap.parse_args()

    adapter = None if args.no_adapter else args.adapter
    model, tok = load(args.model, adapter_path=adapter)

    rows = [json.loads(l) for l in open(args.pairs, encoding="utf-8")]
    by_dir = defaultdict(list)
    for r in rows:
        if len(by_dir[r["direction"]]) < args.limit:
            by_dir[r["direction"]].append(r)

    chrf = CHRF(word_order=2)  # chrF++
    with open(args.out, "w", encoding="utf-8") as f:
        for direction, items in by_dir.items():
            hyps, refs = [], []
            for i, r in enumerate(items, 1):
                hyp = translate(model, tok, r["src"], direction)
                hyps.append(hyp)
                refs.append(r["ref"])
                f.write(json.dumps({**r, "hyp": hyp}, ensure_ascii=False) + "\n")
                if i % 50 == 0:
                    print(f"  {direction}: {i}/{len(items)}")
            score = chrf.corpus_score(hyps, [refs]).score
            print(f"{direction}: chrF++ = {score:.1f}  (n={len(items)})")


if __name__ == "__main__":
    main()
