import SwiftUI

/// Корневой экран: вкладки «Текст», «Документы», «Настройки»
struct ContentView: View {
    @StateObject private var translator = Translator()

    var body: some View {
        TabView {
            TextTranslateView()
                .tabItem { Label("Текст", systemImage: "character.book.closed") }
            DocumentsView()
                .tabItem { Label("Документы", systemImage: "doc.text") }
            SettingsView()
                .tabItem { Label("Настройки", systemImage: "gearshape") }
        }
        .environmentObject(translator)
        .frame(minWidth: 560, minHeight: 560)
        .task { await translator.load() }
    }
}

/// Строка состояния модели — одна на все вкладки
struct ModelStatusView: View {
    @EnvironmentObject private var translator: Translator

    var body: some View {
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
}

/// Направление перевода с кнопкой ⇄
struct DirectionBar: View {
    @Binding var direction: Direction
    var onSwap: () -> Void = {}

    var body: some View {
        HStack {
            Text(direction.source).font(.headline)
            Spacer()
            Button {
                direction = direction.swapped
                onSwap()
            } label: { Image(systemName: "arrow.left.arrow.right") }
                .help("Поменять направление")
            Spacer()
            Text(direction.target).font(.headline)
        }
    }
}
