"""
Быстрая ручная проверка перевода.

  python translate.py "Как ваше здоровье?"                    # ru→tyv
  python translate.py --dir tyv-ru "Кадыыңар кандыг-дыр?"     # tyv→ru
  python translate.py --model models/tyv-qwen3-1.7b --no-adapter "Спасибо"
"""
import argparse

from mlx_lm import load

from tyv_translate import translate

ap = argparse.ArgumentParser()
ap.add_argument("text")
ap.add_argument("--dir", default="ru-tyv", choices=["ru-tyv", "tyv-ru"])
ap.add_argument("--model", default="models/qwen3-1.7b-4bit")
ap.add_argument("--adapter", default="adapters")
ap.add_argument("--no-adapter", action="store_true")
args = ap.parse_args()

model, tok = load(args.model, adapter_path=None if args.no_adapter else args.adapter)
print(translate(model, tok, args.text, args.dir))
