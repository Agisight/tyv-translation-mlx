"""
Готовит датасет Agisight/tyv-rus-200k для LoRA-дообучения в mlx-lm.

Что делает:
  1. Скачивает датасет с Hugging Face (колонки tyv, ru).
  2. Чистит: пустые строки, дубли, слишком длинные тексты, строки где tyv == ru.
  3. Делит на train / valid / test без утечки: одно и то же тувинское
     предложение не может попасть и в train, и в test.
  4. Делает примеры в обе стороны (ru->tyv и tyv->ru) в chat-формате mlx-lm.

Запуск:
  python prepare_data.py                       # по умолчанию
  python prepare_data.py --train-limit 0       # взять весь train (долго)
  python prepare_data.py --test-file my_test.csv   # свой тест-сет (колонки tyv, ru)
"""
import argparse
import csv
import json
import random
import re
from pathlib import Path

# Короткий промпт без системной строки: меньше токенов на пример — быстрее обучение
SYSTEM = None
PROMPTS = {
    "ru-tyv": "ru→tyv: {src}",
    "tyv-ru": "tyv→ru: {src}",
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def make_example(src: str, tgt: str, direction: str) -> dict:
    msgs = [{"role": "system", "content": SYSTEM}] if SYSTEM else []
    msgs += [
        {"role": "user", "content": PROMPTS[direction].format(src=src)},
        {"role": "assistant", "content": tgt},
    ]
    return {"messages": msgs}


def load_pairs_from_hf(name: str):
    from datasets import load_dataset
    ds = load_dataset(name, split="train")
    return [(r["tyv"], r["ru"]) for r in ds]


def load_pairs_from_file(path: str):
    p = Path(path)
    pairs = []
    if p.suffix == ".jsonl":
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            pairs.append((r["tyv"], r["ru"]))
    else:
        with p.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                pairs.append((r["tyv"], r["ru"]))
    return pairs


def clean(pairs, max_chars):
    seen, out = set(), []
    for tyv, ru in pairs:
        if not tyv or not ru:
            continue
        tyv, ru = norm(tyv), norm(ru)
        if not tyv or not ru or tyv == ru:
            continue
        if len(tyv) > max_chars or len(ru) > max_chars:
            continue
        key = (tyv.lower(), ru.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append((tyv, ru))
    return out


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Agisight/tyv-rus-200k")
    ap.add_argument("--out", default="data")
    ap.add_argument("--max-chars", type=int, default=300,
                    help="отбросить пары длиннее N символов (длинные тексты раздувают память)")
    ap.add_argument("--valid-size", type=int, default=1000, help="пар в valid")
    ap.add_argument("--test-size", type=int, default=1000, help="пар в test")
    ap.add_argument("--train-limit", type=int, default=60000,
                    help="сколько train-примеров оставить (после удвоения на 2 направления); 0 = все")
    ap.add_argument("--test-file", default=None,
                    help="свой тест-сет csv/jsonl с колонками tyv, ru (например, тот же, что для NLLB)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pairs = clean(load_pairs_from_hf(args.dataset), args.max_chars)
    print(f"После очистки: {len(pairs):,} пар")

    # Группируем по тувинской стороне, чтобы не было утечки между сплитами
    groups = {}
    for tyv, ru in pairs:
        groups.setdefault(tyv.lower(), []).append((tyv, ru))
    keys = list(groups)
    rng.shuffle(keys)

    if args.test_file:
        test_pairs = clean(load_pairs_from_file(args.test_file), 10_000)
        test_keys = {t.lower() for t, _ in test_pairs}
        keys = [k for k in keys if k not in test_keys]  # убираем тест из train
        print(f"Свой тест-сет: {len(test_pairs):,} пар (исключены из train)")
    else:
        # В valid/test берём только нормальные предложения (3+ слова), не словарные статьи
        sent_keys = [k for k in keys if len(k.split()) >= 3]
        test_keys = set(sent_keys[: args.test_size])
        test_pairs = [groups[k][0] for k in test_keys]

    rest = [k for k in keys if k not in test_keys]
    valid_keys = set([k for k in rest if len(k.split()) >= 3][: args.valid_size])
    train_keys = [k for k in rest if k not in valid_keys]

    valid_pairs = [groups[k][0] for k in valid_keys]
    train_pairs = [p for k in train_keys for p in groups[k]]

    def both_ways(ps):
        ex = []
        for tyv, ru in ps:
            ex.append(make_example(ru, tyv, "ru-tyv"))
            ex.append(make_example(tyv, ru, "tyv-ru"))
        return ex

    train = both_ways(train_pairs)
    rng.shuffle(train)
    if args.train_limit:
        train = train[: args.train_limit]
    valid = both_ways(valid_pairs)
    test = both_ways(test_pairs)

    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "valid.jsonl", valid)
    write_jsonl(out / "test.jsonl", test)
    # Отдельный файл для оценки chrF++: исходник, эталон, направление
    eval_rows = []
    for tyv, ru in test_pairs:
        eval_rows.append({"direction": "ru-tyv", "src": ru, "ref": tyv})
        eval_rows.append({"direction": "tyv-ru", "src": tyv, "ref": ru})
    write_jsonl(out / "eval_pairs.jsonl", eval_rows)

    print(f"train: {len(train):,}  valid: {len(valid):,}  test: {len(test):,} примеров")
    print(f"Файлы в {out}/")


if __name__ == "__main__":
    main()
