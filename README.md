# Тувинский переводчик на MLX: быстрый первый прогон

Цель первого прогона — проверить, что всё работает: обучение на Маке,
перевод в терминале, запуск на Mac и iPhone. Качество улучшаем потом.

Модель: Qwen3-1.7B в 4 битах (QLoRA). Ожидаемое время обучения на MacBook Air M4: около часа.

---

## 1. Окружение (один раз)

```bash
cd tyv-mlx
python3 -m venv .venv && source .venv/bin/activate
pip install "mlx-lm[train]" datasets sacrebleu
```

## 2. Скачать модель и сжать в 4 бита (один раз)

```bash
mlx_lm.convert --hf-path Qwen/Qwen3-1.7B -q --q-bits 4 --mlx-path models/qwen3-1.7b-4bit
```

## 3. Подготовить данные

```bash
python prepare_data.py
```

Для честного сравнения с NLLB (chrF++ 48.5 / 48.1) подставь тот же тест-сет:

```bash
python prepare_data.py --test-file path/to/nllb_test.csv   # колонки tyv, ru
```

## 4. Проверить базовую модель до обучения (2–3 минуты)

```bash
python translate.py --no-adapter "Как ваше здоровье?"
python evaluate.py --no-adapter --limit 100
```

Скорее всего, переводит плохо. Это нормально, так и должно быть до обучения.

## 5. Обучение

Подключи зарядку, поставь Мак на подставку, крышку не закрывай.

```bash
caffeinate -i mlx_lm.lora --config lora_config.yaml
```

- Через 20 шагов появится строка с `It/sec`. Время прогона ≈ 3000 / It/sec секунд.
- `Val loss` каждые 250 шагов должен падать.
- Если упало с нехваткой памяти, поставь в `lora_config.yaml` `batch_size: 4`.
- Адаптер сохраняется каждые 500 шагов в `adapters/`, прогон можно прервать и продолжить:
  `mlx_lm.lora --config lora_config.yaml --resume-adapter-file adapters/adapters.safetensors`

## 6. Проверить перевод

```bash
python translate.py "Как ваше здоровье?"
python translate.py --dir tyv-ru "Кадыыңар кандыг-дыр?"
python evaluate.py --limit 500
```

Результаты с переводами сохраняются в `eval_results.jsonl`. Их полезно просмотреть глазами.

## 7. Слить адаптер с моделью

```bash
mlx_lm.fuse --model models/qwen3-1.7b-4bit --adapter-path adapters --save-path models/tyv-qwen3-1.7b
python translate.py --model models/tyv-qwen3-1.7b --no-adapter "Спасибо"
```

Получится папка примерно на 1 ГБ. Это и есть модель для Mac и iPhone.

## 8. Выложить на Hugging Face (удобно для приложения)

```bash
pip install -U huggingface_hub
hf auth login
hf upload Agisight/tyv-qwen3-1.7b-4bit models/tyv-qwen3-1.7b --repo-type model --private
```

## 9. Запуск на Mac и iPhone

**Важно: MLX не работает в симуляторе iOS**, ему нужен настоящий GPU.
Проверять надо на Маке (как Mac-приложение) или на реальном iPhone.

Самый быстрый путь — готовое демо-приложение Apple LLMEval:

1. `git clone https://github.com/ml-explore/mlx-swift-examples`
2. Открой `mlx-swift-examples.xcodeproj`, выбери схему **LLMEval**.
3. В коде приложения замени модель на свою: идентификатор `Agisight/tyv-qwen3-1.7b-4bit`
   (или путь к локальной папке `models/tyv-qwen3-1.7b`).
4. Отключи режим рассуждений Qwen3: при подготовке ввода передай в шаблон
   `enable_thinking: false` (в `UserInput` это параметр `additionalContext`).
5. Промпт пиши в том же формате, что при обучении: `ru→tyv: Как ваше здоровье?`
6. Запусти на **My Mac**, потом на iPhone (нужен твой Team в Signing).

Модель 1.7B в 4 битах занимает около 1 ГБ, это укладывается в рекомендацию Apple
держать модели на iOS в пределах ~2 ГБ.

---

## Если что-то пошло не так

Пришли текст ошибки. Самые частые:
- **Out of memory** → `batch_size: 4`, потом `max_seq_length: 96`.
- **Модель «думает» и пишет `<think>`** → проверь `enable_thinking=False` (в Python это уже сделано в `tyv_translate.py`).
- **Перевод пустой или повторяется** → мало шагов обучения, смотри на `Val loss`.
