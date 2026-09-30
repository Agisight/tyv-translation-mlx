"""
Прогоняет NLLB v3 на том же тест-сете, что и Qwen, — на Маке (GPU через MPS).
Настройки как в eval_for_paper.ipynb: greedy, вход до 64 токенов, max_new 64.

Нужны доп. библиотеки (один раз):
  pip install torch "transformers<5" sentencepiece

Запуск:
  python evaluate_nllb.py --limit 100     # быстрая проверка
  python evaluate_nllb.py                 # все 2000 пар
"""
import argparse
import json
from collections import defaultdict

import torch
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

LANG = {"ru-tyv": ("rus_Cyrl", "tyv_Cyrl"), "tyv-ru": ("tyv_Cyrl", "rus_Cyrl")}


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
            hyps = []
            for i in range(0, len(items), args.bs):
                batch = [r["src"] for r in items[i:i + args.bs]]
                enc = tok(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=64).to(device)
                gen = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(tgt_lang),
                                     max_new_tokens=64, num_beams=1)
                hyps += tok.batch_decode(gen, skip_special_tokens=True)
                print(f"  {direction}: {min(i + args.bs, len(items))}/{len(items)}")
            refs = [r["ref"] for r in items]
            for r, h in zip(items, hyps):
                f.write(json.dumps({**r, "hyp": h}, ensure_ascii=False) + "\n")
            print(f"{direction}: chrF++ = {chrf.corpus_score(hyps, [refs]).score:.1f}  "
                  f"BLEU = {bleu.corpus_score(hyps, [refs]).score:.1f}  (n={len(items)})")


if __name__ == "__main__":
    main()
