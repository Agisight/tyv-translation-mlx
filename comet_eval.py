"""
COMET по уже сохранённым переводам (модели перевода не нужны).

COMET обучен на человеческих оценках качества перевода и терпим к синонимам и перефразу.
Он основан на XLM-R, который знает русский, но не знает тувинский, поэтому
основной результат — для tyv→ru (ответ на русском). ru→tyv считается только с флагом
--both и в статью идёт с оговоркой.

Отдельное окружение, чтобы не трогать рабочее .venv:
  python3.12 -m venv .venv-comet
  source .venv-comet/bin/activate
  pip install unbabel-comet
  python comet_eval.py eval_results_nllb.jsonl eval_results_nllb-v3-mlx-q8.jsonl eval_results_nllb-v3-mlx-q4.jsonl
  deactivate && source .venv/bin/activate

Первый запуск скачает модель (~2.3 ГБ). Считается на CPU: ~5–15 минут на файл.
Оценки по каждому предложению сохраняются в comet_<файл>.jsonl — для разбора случаев.
"""
import argparse
import json
from pathlib import Path


def load(path: str):
    p = Path(path)
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    return [json.loads(l) for l in p.open(encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--model", default="Unbabel/wmt22-comet-da")
    ap.add_argument("--both", action="store_true", help="также ru→tyv (ненадёжно: COMET не знает тувинский)")
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()

    from comet import download_model, load_from_checkpoint
    model = load_from_checkpoint(download_model(args.model))  # загружается один раз

    directions = ["tyv-ru"] + (["ru-tyv"] if args.both else [])
    results = []
    for f in args.files:
        rows = load(f)
        seg_out = []
        scores = {}
        for d in directions:
            rs = [r for r in rows if r["direction"] == d]
            data = [{"src": r["src"], "mt": r["hyp"], "ref": r["ref"]} for r in rs]
            out = model.predict(data, batch_size=args.bs, gpus=0, progress_bar=True, num_workers=2)
            scores[d] = out.system_score
            seg_out += [{**r, "comet": s} for r, s in zip(rs, out.scores)]
            print(f"{Path(f).name} {d}: COMET = {out.system_score:.4f} (n={len(rs)})")
        name = f"comet_{Path(f).stem}.jsonl"
        with open(name, "w", encoding="utf-8") as fo:
            for r in seg_out:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        results.append((Path(f).name, scores))

    print("\n| Файл | COMET tyv→ru |" + (" COMET ru→tyv (с оговоркой) |" if args.both else ""))
    print("|---|---|" + ("---|" if args.both else ""))
    for name, s in results:
        line = f"| {name} | {s['tyv-ru']:.4f} |"
        if args.both:
            line += f" {s['ru-tyv']:.4f} |"
        print(line)


if __name__ == "__main__":
    main()
