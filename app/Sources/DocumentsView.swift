import SwiftUI
import UniformTypeIdentifiers

/// Вкладка «Документы»: перевод текстовых файлов и колонок таблиц (CSV/TSV) целиком.
struct DocumentsView: View {
    @EnvironmentObject private var translator: Translator
    @AppStorage(Settings.maxCharsKey) private var maxChars = Settings.defaultMaxChars

    @State private var direction: Direction = .ruToTyv
    @State private var document: LoadedDocument?
    @State private var hasHeader = true
    @State private var column = 0
    @State private var resultText = ""
    @State private var resultRows: [[String]] = []
    @State private var progress: (done: Int, total: Int)?
    @State private var started: Date?
    @State private var job: Task<Void, Never>?
    @State private var importing = false
    @State private var exporting = false
    @State private var errorMessage: String?

    private var isTranslating: Bool { job != nil }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            ModelStatusView()
            DirectionBar(direction: $direction)

            HStack {
                Button { importing = true } label: { Label("Открыть файл…", systemImage: "folder") }
                if let document { Text(document.name).font(.headline).lineLimit(1) }
                Spacer()
                Text("TXT, MD, CSV, TSV, RTF" + (isMac ? ", DOCX" : "")).font(.caption).foregroundStyle(.secondary)
            }

            if let document {
                switch document {
                case .text(_, let content):
                    textPanel(content)
                case .table(_, let rows, _):
                    tablePanel(rows)
                }
            } else {
                placeholder
            }

            if let errorMessage {
                Label(errorMessage, systemImage: "exclamationmark.triangle").font(.caption).foregroundStyle(.red)
            }
        }
        .padding()
        .fileImporter(isPresented: $importing, allowedContentTypes: DocumentIO.importTypes) { result in
            open(result)
        }
        .fileExporter(
            isPresented: $exporting, document: ExportDocument(text: exportText),
            contentType: exportType, defaultFilename: exportName
        ) { _ in }
    }

    // MARK: - Панели

    private var placeholder: some View {
        VStack(spacing: 10) {
            Image(systemName: "doc.badge.plus").font(.system(size: 40)).foregroundStyle(.secondary)
            Text("Откройте текстовый файл или таблицу").font(.headline)
            Text("""
            Текст переводится по предложениям, таблица — выбранная колонка, по ячейкам. \
            Длинные предложения режутся на части до \(maxChars) символов: модель обучена на коротких фразах. \
            Excel (.xlsx) сохраните как CSV.
            """)
            .font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func textPanel(_ content: String) -> some View {
        let segments = content.components(separatedBy: .newlines).reduce(0) {
            $0 + Segmenter.segments(of: $1, maxChars: maxChars).count
        }
        return VStack(alignment: .leading, spacing: 10) {
            Text("\(segments) фрагментов · примерно \(estimate(segments))").font(.caption).foregroundStyle(.secondary)
            HStack(alignment: .top, spacing: 10) {
                preview(content)
                preview(resultText.isEmpty ? " " : resultText)
            }
            controls(canExport: !resultText.isEmpty) { translateText(content) }
        }
    }

    private func tablePanel(_ rows: [[String]]) -> some View {
        let columns = rows.map(\.count).max() ?? 0
        let header = hasHeader ? (rows.first ?? []) : []
        let dataRows = hasHeader ? Array(rows.dropFirst()) : rows
        let shown = resultRows.isEmpty ? rows : resultRows
        return VStack(alignment: .leading, spacing: 10) {
            HStack {
                Toggle("Первая строка — заголовки", isOn: $hasHeader)
                Spacer()
                Picker("Колонка", selection: $column) {
                    ForEach(0..<max(columns, 1), id: \.self) { i in
                        Text(i < header.count && !header[i].isEmpty ? header[i] : "Колонка \(i + 1)").tag(i)
                    }
                }
                .frame(maxWidth: 260)
            }
            Text("\(dataRows.count) строк · примерно \(estimate(dataRows.count))").font(.caption).foregroundStyle(.secondary)
            tablePreview(shown)
            controls(canExport: !resultRows.isEmpty) { translateTable(rows) }
        }
    }

    private func controls(canExport: Bool, start: @escaping () -> Void) -> some View {
        HStack(spacing: 12) {
            if isTranslating {
                Button(role: .cancel) { job?.cancel() } label: { Label("Стоп", systemImage: "stop.fill") }
                if let p = progress {
                    ProgressView(value: Double(p.done), total: Double(max(p.total, 1))).frame(maxWidth: 220)
                    Text("\(p.done) из \(p.total)").monospacedDigit().foregroundStyle(.secondary)
                    if let left = remaining(p) { Text(left).foregroundStyle(.secondary) }
                }
            } else {
                Button(action: start) { Label("Перевести всё", systemImage: "character.book.closed") }
                    .buttonStyle(.borderedProminent)
                    .disabled(translator.state != .ready)
                Button { exporting = true } label: { Label("Сохранить…", systemImage: "square.and.arrow.down") }
                    .disabled(!canExport)
            }
            Spacer()
        }
    }

    private func preview(_ text: String) -> some View {
        ScrollView {
            Text(text).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(minHeight: 220)
        .padding(8)
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 8))
    }

    private func tablePreview(_ rows: [[String]]) -> some View {
        ScrollView([.horizontal, .vertical]) {
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 6) {
                ForEach(Array(rows.prefix(200).enumerated()), id: \.offset) { index, row in
                    GridRow {
                        ForEach(Array(row.enumerated()), id: \.offset) { _, cell in
                            Text(cell).lineLimit(2).frame(maxWidth: 260, alignment: .leading)
                                .fontWeight(hasHeader && index == 0 ? .semibold : .regular)
                        }
                    }
                }
            }
            .padding(8)
        }
        .frame(minHeight: 220)
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 8))
    }

    // MARK: - Действия

    private func open(_ result: Result<URL, Error>) {
        errorMessage = nil
        resultText = ""
        resultRows = []
        column = 0
        do {
            document = try DocumentIO.load(try result.get())
        } catch {
            errorMessage = "Не удалось открыть файл: \(error.localizedDescription)"
        }
    }

    private func translateText(_ content: String) {
        resultText = ""
        run {
            _ = try await translator.translateText(
                content, direction: direction, maxChars: maxChars, streaming: false,
                onPartial: { resultText = $0 },
                onProgress: { progress = ($0, $1) })
        }
    }

    private func translateTable(_ rows: [[String]]) {
        let col = column
        let header = hasHeader
        var output = rows.map { $0 }
        let first = header ? 1 : 0
        if header, !output.isEmpty {
            let title = col < output[0].count ? output[0][col] : "Колонка \(col + 1)"
            output[0].append("\(title) (\(direction.targetCode))")
        }
        resultRows = output
        run {
            let total = rows.count - first
            progress = (0, total)
            for r in first..<rows.count {
                try Task.checkCancellation()
                let cell = col < rows[r].count ? rows[r][col] : ""
                var translated = ""
                if !cell.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    translated = try await translator.translateText(
                        cell, direction: direction, maxChars: maxChars, streaming: false, onPartial: { _ in })
                }
                output[r].append(translated)
                resultRows = output
                progress = (r - first + 1, total)
            }
        }
    }

    private func run(_ work: @escaping @MainActor () async throws -> Void) {
        errorMessage = nil
        started = Date()
        job = Task {
            do {
                try await work()
            } catch is CancellationError {
            } catch {
                errorMessage = error.localizedDescription
            }
            job = nil
        }
    }

    // MARK: - Экспорт и оценки времени

    private var exportText: String {
        if case .table(_, _, let delimiter) = document, !resultRows.isEmpty {
            return DocumentIO.makeCSV(resultRows, delimiter: delimiter)
        }
        return resultText
    }

    private var exportType: UTType {
        if case .table(_, _, let delimiter) = document {
            return delimiter == "\t" ? .tabSeparatedText : .commaSeparatedText
        }
        return .plainText
    }

    private var exportName: String {
        "\(document?.name ?? "translation")-\(direction.targetCode)"
    }

    /// ~1 с на фрагмент на MacBook Air M4
    private func estimate(_ segments: Int) -> String {
        segments < 60 ? "\(max(segments, 1)) с" : "\(Int((Double(segments) / 60).rounded(.up))) мин"
    }

    private func remaining(_ p: (done: Int, total: Int)) -> String? {
        guard let started, p.done > 0, p.done < p.total else { return nil }
        let left = Date().timeIntervalSince(started) / Double(p.done) * Double(p.total - p.done)
        return left < 60 ? "≈ \(Int(left)) с" : "≈ \(Int((left / 60).rounded(.up))) мин"
    }

    private var isMac: Bool {
        #if os(macOS)
        true
        #else
        false
        #endif
    }
}
