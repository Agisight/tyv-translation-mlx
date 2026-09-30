"""
Дополнительные метрики поверх уже сохранённых переводов (модели не нужны).

1. chrF++            — обычный, как в статье.
2. chrF++ norm       — «терпимый» к допустимым вариантам:
   - tyv→ru (ответ на русском): слова приводятся к начальной форме (pymorphy3),
     личные местоимения 3-го лица сводятся к одному — род, число и падеж не штрафуются
     (в тувинском нет грамматического рода: «ол» = он/она).
   - ru→tyv (ответ на тувинском): убираются местоимения-подлежащие
     (мен, сен, ол, бис, силер, олар) — лицо и так видно по окончанию глагола.
   Нормализация применяется одинаково к переводу и эталону.
3. COMET (--comet)   — нейросетевая метрика, только для tyv→ru (тувинского COMET не знает).

  pip install pymorphy3
  python metrics_extra.py eval_results_nllb.jsonl eval_results_nllb-v3-mlx-q8.jsonl eval_results_nllb-v3-mlx-q4.jsonl
  python metrics_extra.py eval_results_nllb.jsonl --comet      # + COMET (нужно: pip install unbabel-comet)
"""
import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

import pymorphy3
from sacrebleu.metrics import CHRF

WORD = re.compile(r"[\w-]+", re.UNICODE)
TYV_SUBJ_PRON = {"мен", "сен", "ол", "бис", "силер", "олар"}
RU_3SG = {"он", "она", "оно", "его", "ее", "её", "ему", "ей", "им", "ним", "ней",
          "нем", "нём", "него", "нее", "неё", "ею", "нею"}
_morph = pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=200_000)
def _lemma(w: str) -> str:
    if w in RU_3SG:
        return "он"
    return _morph.parse(w)[0].normal_form.replace("ё", "е")


def norm_ru(t: str) -> str:
    return " ".join(_lemma(w) for w in WORD.findall(t.lower().replace("ё", "е")))


def norm_tyv(t: str) -> str:
    return " ".join(w for w in WORD.findall(t.lower()) if w not in TYV_SUBJ_PRON)


def load(path: str):
    p = Path(path)
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    return [json.loads(l) for l in p.open(encoding="utf-8")]


def comet_scores(rows, model_name):
    from comet import download_model, load_from_checkpoint
    model = load_from_checkpoint(download_model(model_name))
    data = [{"src": r["src"], "mt": r["hyp"], "ref": r["ref"]} for r in rows]
    return model.predict(data, batch_size=32, gpus=0, progress_bar=True).system_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--comet", action="store_true")
    ap.add_argument("--comet-model", default="Unbabel/wmt22-comet-da")
    args = ap.parse_args()

    chrf = CHRF(word_order=2)
    header = "| Файл | n | chrF++ ru→tyv | norm ru→tyv | chrF++ tyv→ru | norm tyv→ru |" + (" COMET tyv→ru |" if args.comet else "")
    print(header)
    print("|" + "---|" * (header.count("|") - 1))
    for f in args.files:
        rows = load(f)
        cells = []
        n = None
        for d, norm in (("ru-tyv", norm_tyv), ("tyv-ru", norm_ru)):
            rs = [r for r in rows if r["direction"] == d]
            n = len(rs)
            h, ref = [r["hyp"] for r in rs], [r["ref"] for r in rs]
            plain = chrf.corpus_score(h, [ref]).score
            normed = chrf.corpus_score([norm(x) for x in h], [[norm(x) for x in ref]]).score
            cells += [f"{plain:.1f}", f"{normed:.1f}"]
        line = f"| {Path(f).name} | {n} | " + " | ".join(cells) + " |"
        if args.comet:
            rs = [r for r in rows if r["direction"] == "tyv-ru"]
            line += f" {comet_scores(rs, args.comet_model):.3f} |"
        print(line)


if __name__ == "__main__":
    main()
