"""
Готовит датасет Agisight/tyv-rus-200k для LoRA-дообучения в mlx-lm.

Что делает:
  1. Скачивает датасет с Hugging Face (колонки tyv, ru).
  2. Чистит: пустые строки, дубли, слишком длинные тексты, строки где tyv == ru.
  3. Делит на train / valid / test без утечки: одно и то же тувинское
     предложение не может попасть и в train, и в test.
  4. Делает примеры в обе стороны (ru->tyv и tyv->ru) в chat-формате mlx-lm.

Тест-сет по умолчанию — ТОТ ЖЕ held-out тест, на котором обучалась и оценивалась NLLB v3
(train_nllb_tyvan_v3_experimental.py, Ячейка 4):
  filter(непустые ru и tyv) -> shuffle(seed=42) -> dev = [0:2000], test = [2000:4000], train = остальное.
Цифры в карточке модели (48.50 / 48.11) посчитаны именно на этом тесте.
Дополнительно из теста убираются пары, чья тувинская ИЛИ русская сторона встречается в train
(в датасете есть дубли). Флаг --keep-dup-test оставляет все 2000.

ВНИМАНИЕ: eval_for_paper.ipynb делил данные иначе (train_test_split) — его тест
пересекается с обучающими данными NLLB, цифры оттуда завышены. Не использовать.

Запуск:
  python prepare_data.py                       # тест как в статье (по умолчанию)
  python prepare_data.py --train-limit 0       # взять весь train (долго)
  python prepare_data.py --split random        # старый случайный сплит
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
# Фиксированная версия датасета: от неё зависят сплит и тест. Новая версия датасета →
# другой тест и несравнимые цифры. Менять только осознанно (и записать в EXPERIMENTS.md).
DATASET_REVISION = "6d79b8acf7031384e8c371c35bd67485d9d19d4d"

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


def load_pairs_from_hf(name: str, revision: str = DATASET_REVISION):
    from datasets import load_dataset
    ds = load_dataset(name, split="train", revision=revision)
    return [(r["tyv"], r["ru"]) for r in ds]


def load_paper_split(name: str, revision: str = DATASET_REVISION):
    """Сплит ровно как при обучении NLLB v3 (train_nllb_tyvan_v3_experimental.py, Ячейка 4)."""
    from datasets import load_dataset
    ds = load_dataset(name, split="train", revision=revision)
    ds = ds.filter(lambda ex: bool(ex.get("ru")) and bool(ex.get("tyv")))
    ds = ds.shuffle(seed=42)
    n = len(ds)
    dev = ds.select(range(0, 2000))
    test = ds.select(range(2000, 4000))
    train = ds.select(range(4000, n))
    pairs = lambda d: [(r["tyv"], r["ru"]) for r in d]
    return pairs(train), pairs(test), pairs(dev)


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
    ap.add_argument("--revision", default=DATASET_REVISION,
                    help="коммит датасета на Hugging Face (по умолчанию — зафиксированный)")
    ap.add_argument("--out", default="data")
    ap.add_argument("--max-chars", type=int, default=300,
                    help="отбросить пары длиннее N символов (длинные тексты раздувают память)")
    ap.add_argument("--valid-size", type=int, default=1000, help="пар в valid")
    ap.add_argument("--test-size", type=int, default=1000, help="пар в test")
    ap.add_argument("--train-limit", type=int, default=60000,
                    help="сколько train-примеров оставить (после удвоения на 2 направления); 0 = все")
    ap.add_argument("--split", default="paper", choices=["paper", "random"],
                    help="paper — тест как в статье про NLLB; random — случайный сплит")
    ap.add_argument("--keep-dup-test", action="store_true",
                    help="не убирать из теста пары, которые дублируют train (как в исходном ноутбуке)")
    ap.add_argument("--test-file", default=None,
                    help="свой тест-сет csv/jsonl с колонками tyv, ru (например, тот же, что для NLLB)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.split == "paper" and not args.test_file:
        train_raw, test_raw, dev_raw = load_paper_split(args.dataset, args.revision)
        # Тест оставляем как в статье: без очистки и без ограничения длины, в исходном порядке
        test_pairs = [((t or "").strip(), (r or "").strip()) for t, r in test_raw]
        test_pairs = [(t, r) for t, r in test_pairs if t and r]
        if not args.keep_dup_test:
            tr_tyv = {(t or "").strip().lower() for t, _ in train_raw}
            tr_ru = {(r or "").strip().lower() for _, r in train_raw}
            before = len(test_pairs)
            test_pairs = [(t, r) for t, r in test_pairs
                          if t.lower() not in tr_tyv and r.lower() not in tr_ru]
            print(f"Из теста убраны дубли train: {before - len(test_pairs):,} пар")
        test_keys = {norm(t).lower() for t, _ in test_pairs}
        pairs = clean(train_raw, args.max_chars)
        dev_pairs = [p for p in clean(dev_raw, args.max_chars) if norm(p[0]).lower() not in test_keys]
        print(f"Тест как у NLLB v3: {len(test_pairs):,} пар; train после очистки: {len(pairs):,} пар")
    else:
        pairs = clean(load_pairs_from_hf(args.dataset, args.revision), args.max_chars)
        print(f"После очистки: {len(pairs):,} пар")
        test_pairs, test_keys, dev_pairs = None, set(), None

    # Группируем по тувинской стороне, чтобы не было утечки между сплитами
    groups = {}
    for tyv, ru in pairs:
        groups.setdefault(tyv.lower(), []).append((tyv, ru))
    keys = [k for k in groups if k not in test_keys]  # дубли тестовых фраз тоже убираем из train
    rng.shuffle(keys)

    if test_pairs is None and args.test_file:
        test_pairs = clean(load_pairs_from_file(args.test_file), 10_000)
        test_keys = {t.lower() for t, _ in test_pairs}
        keys = [k for k in keys if k not in test_keys]
        print(f"Свой тест-сет: {len(test_pairs):,} пар (исключены из train)")
    elif test_pairs is None:
        # В valid/test берём только нормальные предложения (3+ слова), не словарные статьи
        sent_keys = [k for k in keys if len(k.split()) >= 3]
        test_keys = set(sent_keys[: args.test_size])
        test_pairs = [groups[k][0] for k in sent_keys[: args.test_size]]

    rest = [k for k in keys if k not in test_keys]
    if dev_pairs is not None:
        # valid = dev-сплит NLLB (как при её обучении); его фразы убираем из train
        valid_pairs = dev_pairs[: args.valid_size]
        valid_keys = {t.lower() for t, _ in valid_pairs}
    else:
        valid_keys = set([k for k in rest if len(k.split()) >= 3][: args.valid_size])
        valid_pairs = [groups[k][0] for k in valid_keys]
    train_keys = [k for k in rest if k not in valid_keys]

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
