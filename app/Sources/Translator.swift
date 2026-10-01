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
    var swapped: Direction { self == .ruToTyv ? .tyvToRu : .ruToTyv }
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
    @Published var output = ""
    @Published var isTranslating = false
    @Published var lastSeconds: Double?

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
                Task { @MainActor in self.state = .loading(fraction) }
            }
            container = loaded
            state = .ready
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    /// Переводит построчно: модель обучена на отдельных предложениях.
    func translate(_ text: String, direction: Direction) async {
        guard let container, !isTranslating else { return }
        isTranslating = true
        output = ""
        let start = Date()
        var done: [String] = []
        for line in text.components(separatedBy: .newlines) {
            let source = line.trimmingCharacters(in: .whitespaces)
            if source.isEmpty {
                done.append("")
            } else {
                do {
                    done.append(try await Self.translateLine(source, direction: direction, container: container))
                } catch {
                    done.append("⚠️ \(error.localizedDescription)")
                }
            }
            output = done.joined(separator: "\n")
        }
        lastSeconds = Date().timeIntervalSince(start)
        isTranslating = false
    }

    /// Промпт собирается вручную — точно в том виде, в каком модель видела его при обучении
    /// (Gemma 4: `<bos><|turn>user\n…<turn|>\n<|turn>model\n`), жадная генерация.
    nonisolated private static func translateLine(
        _ source: String, direction: Direction, container: ModelContainer
    ) async throws -> String {
        let prompt = "<bos><|turn>user\n\(direction.prefix): \(source)<turn|>\n<|turn>model\n"
        return try await container.perform { context in
            let ids = context.tokenizer.encode(text: prompt, addSpecialTokens: false)
            let endOfTurn = context.tokenizer.encode(text: "<turn|>", addSpecialTokens: false).first
            let maxTokens = min(256, 32 + 3 * ids.count)  // как в оценке: 32 + 3 × длина входа
            let parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)
            var generated: [Int] = []
            _ = try MLXLMCommon.generate(
                input: LMInput(tokens: MLXArray(ids)), parameters: parameters, context: context
            ) { (token: Int) -> GenerateDisposition in
                if token == endOfTurn { return .stop }
                generated.append(token)
                return generated.count >= maxTokens ? .stop : .more
            }
            return context.tokenizer.decode(tokenIds: generated, skipSpecialTokens: true)
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }
}
