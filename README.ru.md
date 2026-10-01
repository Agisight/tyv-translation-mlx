# Тувинский переводчик на MLX — подробная инструкция для Mac (рус.)

> Краткий обзор проекта, результаты и ссылки на модели — в [README.md](README.md) (англ.).
> Здесь — пошаговая установка на MacBook и решения проблем, с которыми мы столкнулись.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Agisight/tyv-translation-mlx/blob/main/colab/train_gemma4_tyv_colab.ipynb)

LoRA-дообучение Qwen3-1.7B (4 бита) на корпусе
[Agisight/tyv-rus-200k](https://huggingface.co/datasets/Agisight/tyv-rus-200k)
прямо на MacBook Air M4. Цель первого прогона — проверить весь путь:
обучение, перевод в терминале, запуск на Mac и iPhone. Качество улучшаем потом.

Текущий статус и результаты — в [EXPERIMENTS.md](EXPERIMENTS.md).
Правила для ИИ-агентов — в [AGENTS.md](AGENTS.md).

> **Про терминал.** zsh на Маке не понимает комментарии `#` во вставленных командах.
> Все команды ниже можно копировать как есть.

---

## 0. Проверить Мак

```bash
uname -m
sysctl -n hw.memsize | awk '{print $1/1073741824 " GB"}'
df -h ~ | tail -1
```

Нужно: `arm64`, от 16 ГБ памяти, от 15 ГБ свободного места.
Если места мало — освободи заранее: при обучении macOS пишет своп на диск.

## 1. Python и окружение (один раз)

Системный `python3` на Маке — 3.9, он не подходит. Ставим 3.12 через Homebrew
и создаём окружение именно им:

```bash
brew install python@3.12
python3.12 -m venv .venv
source .venv/bin/activate
python --version
pip install -U pip
pip install "mlx-lm[train]" datasets sacrebleu
```

Проверка:

```bash
python -c "import mlx.core as mx; print(mx.default_device())"
python -c "import mlx_lm; print(mlx_lm.__version__)"
```

Должно быть `Device(gpu, 0)` и версия mlx-lm.

**В каждом новом терминале** сначала: `cd` в папку проекта и `source .venv/bin/activate`.

## 2. Модель: скачать и сжать в 4 бита (один раз)

Сначала полностью скачиваем репозиторий модели, потом конвертируем.
Без первой строки новый `huggingface_hub` падает при сохранении с
`IncompleteSnapshotError`.

```bash
hf download Qwen/Qwen3-1.7B
mlx_lm.convert --hf-path Qwen/Qwen3-1.7B -q --q-bits 4 --mlx-path models/qwen3-1.7b-4bit
du -sh models/qwen3-1.7b-4bit
mlx_lm.generate --model models/qwen3-1.7b-4bit --prompt "Hello" --max-tokens 30
```

Ожидаемо: около 950 МБ, ответ модели (с тегом `<think>` — это нормально).
Потом удали оригинал из кэша, он больше не нужен (около 4 ГБ):

```bash
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B
```

## 3. Данные

```bash
python prepare_data.py
wc -l data/*.jsonl
```

Тест-сет по умолчанию — **тот же held-out тест, на котором оценивалась NLLB v3**
(`filter → shuffle(seed=42) → dev [0:2000], test [2000:4000]`, как в
`train_nllb_tyvan_v3_experimental.py`). Из него дополнительно убраны пары,
дублирующие train. Valid — dev-сплит NLLB.

Не используй тест из `eval_for_paper.ipynb`: там другой сплит, он пересекается
с обучающими данными NLLB, и цифры завышены.

Проверка, что тест совпал — прогон NLLB (должно выйти около 48.5 / 48.1 на полном тесте):

```bash
pip install torch sentencepiece sacremoses
python evaluate_nllb.py --bs 32
```

## 4. Базовая линия до обучения

```bash
python legacy/qwen/translate.py --no-adapter "Как ваше здоровье?"
python legacy/qwen/evaluate.py --no-adapter --limit 100
```

Модель без обучения не понимает задачу и отвечает как чат-бот. Это нормально.
Запиши цифры в `EXPERIMENTS.md`.

## 5. Обучение (около часа)

Подключи зарядку, поставь Мак на подставку, закрой браузер, крышку не закрывай.

```bash
caffeinate -i mlx_lm.lora --config legacy/qwen/lora_config.yaml
```

- Через 20 шагов появится строка с `It/sec`. Время прогона ≈ 3000 / It/sec секунд.
- `Val loss` каждые 250 шагов должен падать.
- Нехватка памяти → в `legacy/qwen/lora_config.yaml` `batch_size: 4`, потом `max_seq_length: 96`.
- Адаптер сохраняется каждые 500 шагов. Продолжить прерванный прогон:

```bash
caffeinate -i mlx_lm.lora --config legacy/qwen/lora_config.yaml --resume-adapter-file adapters/adapters.safetensors
```

## 6. Проверить перевод

```bash
python legacy/qwen/translate.py "Как ваше здоровье?"
python legacy/qwen/translate.py --dir tyv-ru "Кадыыңар кандыг-дыр?"
python legacy/qwen/evaluate.py --limit 500
```

Переводы сохраняются в `eval_results.jsonl` — их полезно просмотреть глазами.
Запиши результат в `EXPERIMENTS.md`.

## 7. Собрать готовую модель

```bash
mlx_lm.fuse --model models/qwen3-1.7b-4bit --adapter-path adapters --save-path models/tyv-qwen3-1.7b
python legacy/qwen/translate.py --model models/tyv-qwen3-1.7b --no-adapter "Спасибо"
```

Папка около 1 ГБ — это модель для Mac и iPhone.

## 8. Выложить на Hugging Face

```bash
hf auth login
hf upload Agisight/tyv-qwen3-1.7b-4bit models/tyv-qwen3-1.7b --repo-type model --private
```

## 9. Запуск на Mac и iPhone

**MLX не работает в симуляторе iOS** — ему нужен настоящий GPU.
Проверяй на Маке (как Mac-приложение) или на реальном iPhone.

Самый быстрый путь — демо-приложение LLMEval:

1. `git clone https://github.com/ml-explore/mlx-swift-examples`
2. Открой проект, выбери схему **LLMEval**.
3. Замени модель на свою: `Agisight/tyv-qwen3-1.7b-4bit` или путь к `models/tyv-qwen3-1.7b`.
4. Отключи режим рассуждений: передай в шаблон `enable_thinking: false`
   (в `UserInput` это `additionalContext`).
5. Промпт — в формате обучения: `ru→tyv: Как ваше здоровье?`
6. Запусти на **My Mac**, потом на iPhone (нужен твой Team в Signing).

Модель около 1 ГБ — в пределах рекомендации Apple держать модели на iOS до ~2 ГБ.

---

## NLLB v3 на MLX (Задача 1)

NLLB — encoder-decoder, `mlx_lm` его не поддерживает, поэтому архитектура M2M100
реализована вручную в `nllb_mlx.py` (с KV-кэшем для быстрой генерации).

```bash
python convert_nllb_mlx.py
python evaluate_nllb_mlx.py --model models/nllb-v3-mlx-f32 --limit 100
python evaluate_nllb_mlx.py --model models/nllb-v3-mlx-f32
```

Точная копия (f32) должна дать то же, что PyTorch: **49.2 / 49.1** на чистом тесте.
Потом лёгкие варианты:

```bash
python convert_nllb_mlx.py --dtype float16
python convert_nllb_mlx.py --bits 8
python convert_nllb_mlx.py --bits 4
python evaluate_nllb_mlx.py --model models/nllb-v3-mlx-q4
python evaluate_nllb_mlx.py --model models/nllb-v3-mlx-q4 --text "Как ваше здоровье?"
```

Скорость и память на одну фразу (среднее по 30 фразам) и таблица для ручной оценки:

```bash
for v in f32 q8 q4; do python bench_nllb_mlx.py --model models/nllb-v3-mlx-$v; done
python make_human_eval.py
```

## Частые проблемы

| Симптом | Причина и решение |
|---|---|
| `command not found: python3.12` | Python не установлен: `brew install python@3.12` |
| `python --version` показывает 3.9 | Окружение создано не тем Python: удали `.venv`, создай через `python3.12 -m venv .venv` |
| Ошибки `awk`/`tail` с кусками русского текста | В команду попал комментарий `#` — zsh его не понимает |
| `IncompleteSnapshotError` при `mlx_lm.convert` | Сначала `hf download Qwen/Qwen3-1.7B`, потом конвертация заново |
| Out of memory при обучении | `batch_size: 4`, потом `max_seq_length: 96`, потом `num_layers: 4` |
| Модель пишет `<think>` | Нет `enable_thinking=False` (в Python уже есть в `tyv_translate.py`) |
| Перевод пустой или повторяется | Мало шагов обучения — смотри на `Val loss` |
| Подсказки `hf update`, `hf skills` | Игнорировать, обновлять не нужно |
