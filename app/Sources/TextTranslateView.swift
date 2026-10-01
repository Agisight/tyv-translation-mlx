import SwiftUI

/// Вкладка «Текст»: ввод, перевод, копирование. Длинный текст режется на предложения.
struct TextTranslateView: View {
    @EnvironmentObject private var translator: Translator
    @AppStorage(Settings.streamingKey) private var streaming = true
    @AppStorage(Settings.maxCharsKey) private var maxChars = Settings.defaultMaxChars

    @State private var direction: Direction = .ruToTyv
    @State private var input = "Завтра я поеду в Кызыл к родителям."
    @State private var output = ""
    @State private var progress: (done: Int, total: Int)?
    @State private var seconds: Double?
    @State private var job: Task<Void, Never>?

    private var isTranslating: Bool { job != nil }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            ModelStatusView()
            DirectionBar(direction: $direction) {
                if !output.isEmpty { input = output; output = "" }
            }

            TextEditor(text: $input)
                .font(.title3)
                .frame(minHeight: 120)
                .padding(6)
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(.secondary.opacity(0.3)))

            HStack(spacing: 12) {
                if isTranslating {
                    Button(role: .cancel) { job?.cancel() } label: { Label("Стоп", systemImage: "stop.fill") }
                        .keyboardShortcut(".", modifiers: .command)
                } else {
                    Button(action: start) { Label("Перевести", systemImage: "character.book.closed") }
                        .buttonStyle(.borderedProminent)
                        .keyboardShortcut(.return, modifiers: .command)
                        .disabled(translator.state != .ready || input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
                if isTranslating { ProgressView().controlSize(.small) }
                if let p = progress, p.total > 1 {
                    Text("\(p.done) из \(p.total)").foregroundStyle(.secondary).monospacedDigit()
                }
                Spacer()
                if let s = seconds, !isTranslating {
                    Text(String(format: "%.1f с", s)).foregroundStyle(.secondary)
                }
            }

            ScrollView {
                Text(output.isEmpty ? " " : output)
                    .font(.title3)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(minHeight: 120)
            .padding(10)
            .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 8))

            HStack {
                Button { Clipboard.copy(output) } label: { Label("Копировать", systemImage: "doc.on.doc") }
                    .disabled(output.isEmpty)
                Spacer()
                Text("Офлайн · Gemma 4 E4B, 6 бит · 3.2 ГБ").font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding()
    }

    private func start() {
        output = ""
        seconds = nil
        let started = Date()
        job = Task {
            do {
                _ = try await translator.translateText(
                    input, direction: direction, maxChars: maxChars, streaming: streaming,
                    onPartial: { output = $0 },
                    onProgress: { progress = ($0, $1) })
            } catch is CancellationError {
            } catch {
                output += (output.isEmpty ? "" : "\n") + "⚠️ \(error.localizedDescription)"
            }
            seconds = Date().timeIntervalSince(started)
            job = nil
        }
    }
}
