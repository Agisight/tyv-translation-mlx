# Tyva Translator — app for Mac, iPad and iPhone

> Русская версия: [README.ru.md](README.ru.md).

A SwiftUI app that translates Russian ↔ Tuvan on the device itself, without internet.
Model: [Agisight/tyv-gemma4-e4b-pruned-mlx-6bit](https://huggingface.co/Agisight/tyv-gemma4-e4b-pruned-mlx-6bit)
(Gemma 4 E4B, pruned vocabulary, 6-bit, 3.2 GB) via [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm).

## Features

- **Text** — type or paste, translate with ⌘↩, stop with ⌘. ; long text is split into sentences
  automatically, with progress ("12 of 40").
- **Documents** — open TXT, MD, CSV, TSV, RTF (and DOCX on macOS). Text files are translated sentence by
  sentence; for tables you pick a column, and the translation is added as a new column. Save the result
  as TXT or CSV. Excel files: save as CSV first. Line markers (✅, •, -, «1)») are cut off before translation
  and put back after, so the model cannot lose them.
- **Settings** — token-by-token output on or off, and the maximum fragment length.

**Why long text is split.** Gemma has a long context window, but this model was fine-tuned on single
sentences up to 300 characters, so quality drops on long paragraphs. The app splits text into sentences
and splits overly long sentences at punctuation into fragments of at most 300 characters (adjustable),
then translates each fragment separately — the same conditions as in the clean-test evaluation.

## Build

```bash
brew install xcodegen
cd app
xcodegen
open TyvaTranslator.xcodeproj
```

In Xcode:
1. Wait for the packages to resolve (mlx-swift-lm, swift-huggingface, swift-transformers).
2. Xcode asks to download the Metal Toolchain — MLX compiles its GPU shaders, so accept it.
3. On the first build Xcode asks about the `MLXHuggingFace` macros — choose "Trust & Enable".
4. Signing & Capabilities → pick your Team.
5. Target **My Mac** → Run (⌘R).

On first launch the app downloads the model (~3.4 GB) into a cache; after that it works offline.
Translate with the button or ⌘↩. Multiple lines are translated one by one.

Measured on a MacBook Air M4: 3.4 GB of memory for the weights, about 0.8–0.9 s per sentence after
warm-up (the very first translation takes a few seconds while MLX compiles its kernels).

## iPad and iPhone

- **Real device only:** the simulator has no Apple Silicon GPU for MLX.
- The device needs 8 GB of RAM or more: an iPad with an M chip, iPhone 15 Pro or newer.
- `iOS.entitlements` enables the increased memory limit (`increased-memory-limit`): the model needs
  ~3.8 GB. If signing fails, add the "Increased Memory Limit" capability in Signing & Capabilities.

## How it works

- `Translator.swift` — model loading, translation and token-by-token streaming. The prompt is built by hand exactly as in training
  (`<bos><|turn>user\nru→tyv: …<turn|>\n<|turn>model\n`), greedy decoding, at most
  `32 + 3 × input length` tokens — the same settings as the clean-test evaluation.
- `Segmenter.swift` — splitting into sentences (NaturalLanguage) and fragments of at most N characters.
- `DocumentIO.swift` — reading TXT/RTF/DOCX/CSV/TSV, writing CSV, export.
- `ContentView.swift`, `TextTranslateView.swift`, `DocumentsView.swift`, `SettingsView.swift` — the UI tabs.
