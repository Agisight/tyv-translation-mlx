import SwiftUI
#if os(macOS)
import AppKit
#else
import UIKit
#endif

struct ContentView: View {
    @StateObject private var translator = Translator()
    @State private var direction: Direction = .ruToTyv
    @State private var input = "Завтра я поеду в Кызыл к родителям."

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            status

            HStack {
                Text(direction.source).font(.headline)
                Spacer()
                Button(action: swap) { Image(systemName: "arrow.left.arrow.right") }
                    .help("Поменять направление")
                Spacer()
                Text(direction.target).font(.headline)
            }

            TextEditor(text: $input)
                .font(.title3)
                .frame(minHeight: 120)
                .padding(6)
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(.secondary.opacity(0.3)))

            HStack(spacing: 12) {
                Button {
                    Task { await translator.translate(input, direction: direction) }
                } label: {
                    Label("Перевести", systemImage: "character.book.closed")
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.return, modifiers: .command)
                .disabled(translator.state != .ready || translator.isTranslating || input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                if translator.isTranslating { ProgressView().controlSize(.small) }
                Spacer()
                if let s = translator.lastSeconds, !translator.isTranslating {
                    Text(String(format: "%.1f с", s)).foregroundStyle(.secondary)
                }
            }

            ScrollView {
                Text(translator.output.isEmpty ? " " : translator.output)
                    .font(.title3)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(minHeight: 120)
            .padding(10)
            .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 8))

            HStack {
                Button(action: copyOutput) { Label("Копировать", systemImage: "doc.on.doc") }
                    .disabled(translator.output.isEmpty)
                Spacer()
                Text("Офлайн · Gemma 4 E4B, 6 бит · 3.2 ГБ")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
        .frame(minWidth: 520, minHeight: 520)
        .task { await translator.load() }
    }

    @ViewBuilder private var status: some View {
        switch translator.state {
        case .idle:
            EmptyView()
        case .loading(let fraction):
            VStack(alignment: .leading, spacing: 4) {
                Text("Загрузка модели — 3.4 ГБ, только при первом запуске…").font(.caption)
                ProgressView(value: fraction)
            }
        case .ready:
            Label("Модель готова — работает без интернета", systemImage: "checkmark.circle")
                .font(.caption).foregroundStyle(.green)
        case .failed(let message):
            HStack {
                Label(message, systemImage: "exclamationmark.triangle").font(.caption).foregroundStyle(.red)
                Button("Повторить") { Task { await translator.load() } }
            }
        }
    }

    private func swap() {
        direction = direction.swapped
        if !translator.output.isEmpty {
            input = translator.output
            translator.output = ""
        }
    }

    private func copyOutput() {
        #if os(macOS)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(translator.output, forType: .string)
        #else
        UIPasteboard.general.string = translator.output
        #endif
    }
}
