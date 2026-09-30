"""
Оценка NLLB на MLX — на том же тесте и с теми же настройками, что evaluate_nllb.py (PyTorch):
та же предобработка (Moses + NFKC), вход без обрезки, max_new = 32 + 3 * длина входа, greedy.

  python evaluate_nllb_mlx.py --model models/nllb-v3-mlx-f32 --limit 100
  python evaluate_nllb_mlx.py --model models/nllb-v3-mlx-q4
  python evaluate_nllb_mlx.py --model models/nllb-v3-mlx-q4 --text "Как ваше здоровье?"
"""
import argparse
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict

import mlx.core as mx
from sacrebleu.metrics import BLEU, CHRF
from sacremoses import MosesPunctNormalizer
from transformers import AutoTokenizer

from nllb_mlx import generate, load_model

LANG = {"ru-tyv": ("rus_Cyrl", "tyv_Cyrl"), "tyv-ru": ("tyv_Cyrl", "rus_Cyrl")}

_mpn = MosesPunctNormalizer(lang="en")
_mpn.substitutions = [(re.compile(r), s) for r, s in _mpn.substitutions]
_nonprint = {ord(c): " " for c in (chr(i) for i in range(sys.maxunicode + 1))
             if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}}


def preproc(t: str) -> str:
    return unicodedata.normalize("NFKC", _mpn.normalize(t).translate(_nonprint))


def peak_mem_mb() -> float:
    for fn in (getattr(mx, "get_peak_memory", None), getattr(getattr(mx, "metal", None), "get_peak_memory", None)):
        if fn:
            return fn() / 1024 / 1024
    return float("nan")


def translate_batch(model, tok, texts, direction):
    src_lang, tgt_lang = LANG[direction]
    tok.src_lang = src_lang
    enc = tok([preproc(t) for t in texts], return_tensors="np", padding=True,
              truncation=True, max_length=1024)
    ids = enc["input_ids"].astype("int32")
    mask = enc["attention_mask"].astype("int32")
    out = generate(model, ids, mask, tok.convert_tokens_to_ids(tgt_lang), tok.eos_token_id,
                   max_new_tokens=int(32 + 3 * ids.shape[1]))
    return tok.batch_decode(out, skip_special_tokens=True), sum(len(o) + 1 for o in out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/nllb-v3-mlx-f32")
    ap.add_argument("--pairs", default="data/eval_pairs.jsonl")
    ap.add_argument("--limit", type=int, default=100000, help="примеров на направление")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--text", default=None, help="перевести одну фразу и выйти")
    ap.add_argument("--dir", default="ru-tyv", choices=list(LANG))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model)

    if args.text:
        t0 = time.time()
        hyp, n = translate_batch(model, tok, [args.text], args.dir)
        dt = time.time() - t0
        print(hyp[0])
        print(f"[{n} токенов за {dt:.2f} с, {n / dt:.1f} ток/с, пик памяти {peak_mem_mb():.0f} МБ]")
        return

    rows = [json.loads(l) for l in open(args.pairs, encoding="utf-8")]
    by_dir = defaultdict(list)
    for r in rows:
        if len(by_dir[r["direction"]]) < args.limit:
            by_dir[r["direction"]].append(r)

    out_path = args.out or f"eval_results_{args.model.rstrip('/').split('/')[-1]}.jsonl"
    chrf, bleu = CHRF(word_order=2), BLEU()
    with open(out_path, "w", encoding="utf-8") as f:
        for direction, items in by_dir.items():
            order = sorted(range(len(items)), key=lambda i: len(items[i]["src"]), reverse=True)
            hyps = [None] * len(items)
            t0, ntok = time.time(), 0
            for s in range(0, len(order), args.bs):
                idx = order[s:s + args.bs]
                h, n = translate_batch(model, tok, [items[i]["src"] for i in idx], direction)
                ntok += n
                for i, x in zip(idx, h):
                    hyps[i] = x
                print(f"  {direction}: {min(s + args.bs, len(items))}/{len(items)}")
            dt = time.time() - t0
            refs = [r["ref"] for r in items]
            for r, x in zip(items, hyps):
                f.write(json.dumps({**r, "hyp": x}, ensure_ascii=False) + "\n")
            print(f"{direction}: chrF++ = {chrf.corpus_score(hyps, [refs]).score:.1f}  "
                  f"BLEU = {bleu.corpus_score(hyps, [refs]).score:.1f}  (n={len(items)}, "
                  f"{dt:.0f} с, {ntok / dt:.0f} ток/с)")
    print(f"Пик памяти: {peak_mem_mb():.0f} МБ | переводы: {out_path}")


if __name__ == "__main__":
    main()
