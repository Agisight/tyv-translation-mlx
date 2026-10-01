import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

enum Direction: String, CaseIterable, Identifiable {
    case ruToTyv, tyvToRu
    var id: Self { self }
    /// Префикс ровно как при обучении модели
    var prefix: String { self == .ruToTyv ? "ru→tyv" : "tyv→ru" }
    var source: String { self == .ruToTyv ? "Русский" : "Тувинский" }
    var target: String { self == .ruToTyv ? "Тувинский" : "Русский" }
    var targetCode: String { self == .ruToTyv ? "tyv" : "ru" }
    var swapped: Direction { self == .ruToTyv ? .tyvToRu : .ruToTyv }
}

enum TranslatorError: LocalizedError {
    case notReady
    var errorDescription: String? { "Модель ещё не загружена" }
}

/// Настройки, общие для всех экранов (хранятся в UserDefaults через @AppStorage)
enum Settings {
    static let streamingKey = "streaming"
    static let maxCharsKey = "maxChars"
    /// Модель обучена на предложениях до 300 символов — длиннее режем на части
    static let defaultMaxChars = 300
}

@MainActor
final class Translator: ObservableObject {
    /// Сокращённая Gemma 4 E4B (словарь 22 755 токенов, только текст), MLX 6 бит, 3.2 ГБ
    static let modelID = "Agisight/tyv-gemma4-e4b-pruned-mlx-6bit"

    enum State: Equatable {
        case idle
        case loading(Double)
        case ready
        case failed(String)
    }

    @Published var state: State = .idle
    private var container: ModelContainer?

    func load() async {
        guard container == nil else { return }
        state = .loading(0)
        do {
            let loaded = try await LLMModelFactory.shared.loadContainer(
                from: #hubDownloader(),
                using: #huggingFaceTokenizerLoader(),
                configuration: ModelConfiguration(id: Self.modelID)
            ) { progress in
                let fraction = progress.fractionCompleted
                Task { @MainActor in self.state = .loading(min(fraction, 0.99)) }
            }
            container = loaded
            state = .ready
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    /// Переводит любой текст: построчно, длинные строки — по предложениям и частям до `maxChars`.
    /// `onPartial` получает весь текст перевода по мере готовности (если включён потоковый вывод — по токенам).
    /// `onProgress` — сколько частей готово из скольких.
    func translateText(
        _ text: String,
        direction: Direction,
        maxChars: Int,
        streaming: Bool,
        onPartial: @escaping @MainActor (String) -> Void,
        onProgress: @escaping @MainActor (Int, Int) -> Void = { _, _ in }
    ) async throws -> String {
        // Маркеры в начале строки (✅, •, -, «1)») модель не переводит и может потерять — переносим их сами
        let split = text.components(separatedBy: .newlines).map(Segmenter.leadingMarker)
        let markers = split.map(\.marker)
        let lines = split.map { Segmenter.segments(of: $0.body, maxChars: maxChars) }
        let total = lines.reduce(0) { $0 + $1.count }
        var doneLines: [String] = []
        var done = 0
        onProgress(0, total)
        for (lineIndex, segments) in lines.enumerated() {
            let marker = markers[lineIndex]
            var doneSegments: [String] = []
            for segment in segments {
                try Task.checkCancellation()
                let linesSoFar = doneLines
                let segmentsSoFar = doneSegments
                let translated = try await translate(segment, direction: direction, streaming: streaming) { partial in
                    let current = marker + (segmentsSoFar + [partial]).joined(separator: " ")
                    onPartial((linesSoFar + [current]).joined(separator: "\n"))
                }
                doneSegments.append(translated)
                done += 1
                onProgress(done, total)
                onPartial((doneLines + [marker + doneSegments.joined(separator: " ")]).joined(separator: "\n"))
            }
            doneLines.append(segments.isEmpty ? marker.trimmingCharacters(in: .whitespaces) : marker + doneSegments.joined(separator: " "))
        }
        let result = doneLines.joined(separator: "\n")
        onPartial(result)
        return result
    }

    /// Один фрагмент. Промпт собирается вручную — точно как при обучении модели
    /// (Gemma 4: `<bos><|turn>user\n…<turn|>\n<|turn>model\n`), жадная генерация.
    func translate(
        _ source: String,
        direction: Direction,
        streaming: Bool,
        onPartial: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        guard let container else { throw TranslatorError.notReady }
        let prompt = "<bos><|turn>user\n\(direction.prefix): \(source)<turn|>\n<|turn>model\n"
        let (partials, continuation) = AsyncStream.makeStream(of: String.self, bufferingPolicy: .bufferingNewest(1))

        let job = Task.detached { () async throws -> String in
            defer { continuation.finish() }
            return try await container.perform { context in
                let ids = context.tokenizer.encode(text: prompt, addSpecialTokens: false)
                let endOfTurn = context.tokenizer.encode(text: "<turn|>", addSpecialTokens: false).first
                let maxTokens = min(256, 32 + 3 * ids.count)  // как в оценке: 32 + 3 × длина входа
                let parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)
                var generated: [Int] = []
                _ = try MLXLMCommon.generate(
                    input: LMInput(tokens: MLXArray(ids)), parameters: parameters, context: context
                ) { (token: Int) -> GenerateDisposition in
                    if token == endOfTurn || Task.isCancelled { return .stop }
                    generated.append(token)
                    if streaming {
                        continuation.yield(context.tokenizer.decode(tokenIds: generated, skipSpecialTokens: true))
                    }
                    return generated.count >= maxTokens ? .stop : .more
                }
                return context.tokenizer.decode(tokenIds: generated, skipSpecialTokens: true)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }

        return try await withTaskCancellationHandler {
            for await partial in partials { onPartial(partial) }
            return try await job.value
        } onCancel: {
            job.cancel()
        }
    }
}
