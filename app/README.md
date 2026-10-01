# Тыва Переводчик — приложение для Mac, iPad и iPhone

SwiftUI-приложение: перевод русский ↔ тувинский прямо на устройстве, без интернета.
Модель — [Agisight/tyv-gemma4-e4b-pruned-mlx-6bit](https://huggingface.co/Agisight/tyv-gemma4-e4b-pruned-mlx-6bit)
(Gemma 4 E4B, сокращённый словарь, 6 бит, 3.2 ГБ) через [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm).

## Сборка

```bash
brew install xcodegen
cd app
xcodegen
open TyvaTranslator.xcodeproj
```

В Xcode:
1. Дождаться загрузки пакетов (mlx-swift-lm, swift-huggingface, swift-transformers).
2. При первой сборке Xcode спросит про макросы `MLXHuggingFace` — «Trust & Enable».
3. Signing & Capabilities → выбрать свою команду (Team).
4. Цель — **My Mac** → Run (⌘R).

При первом запуске приложение скачает модель (~3.4 ГБ) в кэш, дальше работает офлайн.
Перевести — кнопка или ⌘↩. Несколько строк переводятся построчно.

## iPad и iPhone

- Только **реальное устройство**: в симуляторе нет видеокарты Apple Silicon для MLX.
- Нужна модель с 8 ГБ памяти и больше: iPad с чипом M, iPhone 15 Pro и новее.
- В `iOS.entitlements` включён повышенный лимит памяти (`increased-memory-limit`): модель
  занимает ~3.8 ГБ. Если подпись не проходит — включите capability «Increased Memory Limit»
  в Signing & Capabilities.

## Как устроено

- `Translator.swift` — загрузка модели и перевод. Промпт собирается вручную ровно в формате
  обучения (`<bos><|turn>user\nru→tyv: …<turn|>\n<|turn>model\n`), жадная генерация,
  максимум `32 + 3 × длина входа` токенов — как в оценке на чистом тесте.
- `ContentView.swift` — интерфейс: направление, ввод, перевод, копирование, время перевода.
