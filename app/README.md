# Tyva Translator — app for Mac, iPad and iPhone

> Русская версия: [README.ru.md](README.ru.md).

A SwiftUI app that translates Russian ↔ Tuvan on the device itself, without internet.
Model: [Agisight/tyv-gemma4-e4b-pruned-mlx-6bit](https://huggingface.co/Agisight/tyv-gemma4-e4b-pruned-mlx-6bit)
(Gemma 4 E4B, pruned vocabulary, 6-bit, 3.2 GB) via [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm).

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

- `Translator.swift` — model loading and translation. The prompt is built by hand exactly as in training
  (`<bos><|turn>user\nru→tyv: …<turn|>\n<|turn>model\n`), greedy decoding, at most
  `32 + 3 × input length` tokens — the same settings as the clean-test evaluation.
- `ContentView.swift` — the UI: direction, input, translation, copy, translation time.
