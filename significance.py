"""
Статистическая значимость разницы между системами (paired bootstrap, sacrebleu).

  python significance.py eval_results_nllb.jsonl eval_results_gemma4.json
  python significance.py eval_results_nllb.jsonl eval_results_gemma4.json eval_results_nllb-v3-mlx-q4.jsonl

Первый файл — базовая система (обычно NLLB). Для каждой следующей sacrebleu скажет,
значима ли разница (p-value < 0.05 — разница не случайна). Метрики: chrF++ и BLEU.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def load(path):
    p = Path(path)
    rows = json.loads(p.read_text(encoding="utf-8")) if p.suffix == ".json" else \
        [json.loads(l) for l in p.open(encoding="utf-8")]
    return {(r["direction"], r["src"]): r for r in rows}


files = sys.argv[1:]
assert len(files) >= 2, "нужно минимум два файла: базовая система и сравниваемая"
systems = [load(f) for f in files]
keys_all = set.intersection(*(set(s) for s in systems))

for d in ("ru-tyv", "tyv-ru"):
    keys = sorted(k for k in keys_all if k[0] == d)
    with tempfile.TemporaryDirectory() as tmp:
        ref = Path(tmp) / "ref.txt"
        ref.write_text("\n".join(systems[0][k]["ref"].replace("\n", " ") for k in keys) + "\n", encoding="utf-8")
        hyp_paths = []
        for f, s in zip(files, systems):
            hp = Path(tmp) / (Path(f).stem + ".txt")
            hp.write_text("\n".join(s[k]["hyp"].replace("\n", " ") for k in keys) + "\n", encoding="utf-8")
            hyp_paths.append(str(hp))
        print(f"\n===== {d} (n={len(keys)}) =====")
        subprocess.run(["sacrebleu", str(ref), "-i", *hyp_paths, "-m", "chrf", "bleu",
                        "--chrf-word-order", "2", "--paired-bs", "-f", "text"], check=True)
