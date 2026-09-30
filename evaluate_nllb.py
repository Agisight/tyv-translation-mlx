"""
Прогоняет NLLB v3 на том же тест-сете, что и Qwen, — на Маке (GPU через MPS).
Настройки как в финальной оценке при обучении v3 (train_nllb_tyvan_v3_experimental.py,
Ячейка 10): та же предобработка текста (Moses + NFKC), вход без обрезки,
max_new_tokens = 32 + 3 * длина входа, greedy.

Нужны доп. библиотеки (один раз):
  pip install torch sentencepiece sacremoses

Запуск:
  python evaluate_nllb.py --limit 100     # быстрая проверка
  python evaluate_nllb.py                 # все 2000 пар
"""
import argparse
import json
from collections import defaultdict

import re
import sys
import unicodedata

import torch
from sacrebleu.metrics import BLEU, CHRF
from sacremoses import MosesPunctNormalizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

LANG = {"ru-tyv": ("rus_Cyrl", "tyv_Cyrl"), "tyv-ru": ("tyv_Cyrl", "rus_Cyrl")}

# Предобработка как при обучении NLLB v3
_mpn = MosesPunctNormalizer(lang="en")
_mpn.substitutions = [(re.compile(r), s) for r, s in _mpn.substitutions]
_nonprint = {ord(c): " " for c in (chr(i) for i in range(sys.maxunicode + 1))
             if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}}


def preproc(t: str) -> str:
    return unicodedata.normalize("NFKC", _mpn.normalize(t).translate(_nonprint))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Agisight/nllb-rus-tyv-v3-dict_16.5k")
    ap.add_argument("--pairs", default="data/eval_pairs.jsonl")
    ap.add_argument("--limit", type=int, default=2000, help="примеров на направление")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--out", default="eval_results_nllb.jsonl")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device).eval()
    print(f"device: {device}")

    rows = [json.loads(l) for l in open(args.pairs, encoding="utf-8")]
    by_dir = defaultdict(list)
    for r in rows:
        if len(by_dir[r["direction"]]) < args.limit:
            by_dir[r["direction"]].append(r)

    chrf, bleu = CHRF(word_order=2), BLEU()
    with open(args.out, "w", encoding="utf-8") as f, torch.no_grad():
        for direction, items in by_dir.items():
            src_lang, tgt_lang = LANG[direction]
            tok.src_lang = src_lang
            order = sorted(range(len(items)), key=lambda i: len(items[i]["src"]), reverse=True)
            hyps = [None] * len(items)
            for s in range(0, len(order), args.bs):
                idx = order[s:s + args.bs]
                batch = [preproc(items[i]["src"]) for i in idx]
                enc = tok(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=1024).to(device)
                gen = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(tgt_lang),
                                     max_new_tokens=int(32 + 3 * enc.input_ids.shape[1]), num_beams=1)
                for i, h in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
                    hyps[i] = h
                print(f"  {direction}: {min(s + args.bs, len(items))}/{len(items)}")
            refs = [r["ref"] for r in items]
            for r, h in zip(items, hyps):
                f.write(json.dumps({**r, "hyp": h}, ensure_ascii=False) + "\n")
            print(f"{direction}: chrF++ = {chrf.corpus_score(hyps, [refs]).score:.1f}  "
                  f"BLEU = {bleu.corpus_score(hyps, [refs]).score:.1f}  (n={len(items)})")


if __name__ == "__main__":
    main()
