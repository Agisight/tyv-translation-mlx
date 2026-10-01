import SwiftUI
import UniformTypeIdentifiers
#if os(macOS)
import AppKit
#else
import UIKit
#endif

/// Загруженный документ: обычный текст или таблица
enum LoadedDocument {
    case text(name: String, content: String)
    case table(name: String, rows: [[String]], delimiter: Character)

    var name: String {
        switch self {
        case .text(let name, _), .table(let name, _, _): return name
        }
    }
}

enum DocumentIO {
    static let docx = UTType("org.openxmlformats.wordprocessingml.document") ?? .data

    /// Что можно открыть. Excel (.xlsx) — сохраните как CSV в Excel или Numbers.
    static var importTypes: [UTType] {
        var types: [UTType] = [.plainText, .utf8PlainText, .commaSeparatedText, .tabSeparatedText, .rtf]
        if let md = UTType(filenameExtension: "md") { types.append(md) }
        #if os(macOS)
        types.append(docx)
        #endif
        return types
    }

    static func load(_ url: URL) throws -> LoadedDocument {
        let accessing = url.startAccessingSecurityScopedResource()
        defer { if accessing { url.stopAccessingSecurityScopedResource() } }
        let name = url.deletingPathExtension().lastPathComponent
        switch url.pathExtension.lowercased() {
        case "csv":
            return .table(name: name, rows: parseCSV(try readText(url), delimiter: ","), delimiter: ",")
        case "tsv", "tab":
            return .table(name: name, rows: parseCSV(try readText(url), delimiter: "\t"), delimiter: "\t")
        case "rtf":
            let attributed = try NSAttributedString(
                url: url, options: [.documentType: NSAttributedString.DocumentType.rtf], documentAttributes: nil)
            return .text(name: name, content: attributed.string)
        #if os(macOS)
        case "docx":
            let attributed = try NSAttributedString(
                url: url, options: [.documentType: NSAttributedString.DocumentType.officeOpenXML],
                documentAttributes: nil)
            return .text(name: name, content: attributed.string)
        #endif
        default:
            return .text(name: name, content: try readText(url))
        }
    }

    private static func readText(_ url: URL) throws -> String {
        let data = try Data(contentsOf: url)
        if let s = String(data: data, encoding: .utf8) { return s.replacingOccurrences(of: "\r\n", with: "\n") }
        if let s = String(data: data, encoding: .windowsCP1251) { return s.replacingOccurrences(of: "\r\n", with: "\n") }
        throw CocoaError(.fileReadInapplicableStringEncoding)
    }

    /// CSV по RFC 4180: кавычки, запятые и переводы строк внутри ячеек
    static func parseCSV(_ text: String, delimiter: Character) -> [[String]] {
        var rows: [[String]] = []
        var row: [String] = []
        var field = ""
        var inQuotes = false
        var chars = Array(text)
        if chars.first == "\u{FEFF}" { chars.removeFirst() }
        var i = 0
        while i < chars.count {
            let c = chars[i]
            if inQuotes {
                if c == "\"" {
                    if i + 1 < chars.count && chars[i + 1] == "\"" { field.append("\""); i += 1 } else { inQuotes = false }
                } else {
                    field.append(c)
                }
            } else if c == "\"" {
                inQuotes = true
            } else if c == delimiter {
                row.append(field); field = ""
            } else if c == "\n" || c == "\r" {
                if c == "\r" && i + 1 < chars.count && chars[i + 1] == "\n" { i += 1 }
                row.append(field); field = ""
                rows.append(row); row = []
            } else {
                field.append(c)
            }
            i += 1
        }
        if !field.isEmpty || !row.isEmpty { row.append(field); rows.append(row) }
        return rows.filter { !($0.count == 1 && $0[0].isEmpty) }
    }

    static func makeCSV(_ rows: [[String]], delimiter: Character) -> String {
        rows.map { row in
            row.map { cell in
                let needsQuotes = cell.contains(delimiter) || cell.contains("\"") || cell.contains("\n")
                return needsQuotes ? "\"" + cell.replacingOccurrences(of: "\"", with: "\"\"") + "\"" : cell
            }.joined(separator: String(delimiter))
        }.joined(separator: "\n") + "\n"
    }
}

/// Документ для сохранения результата (текст или CSV)
struct ExportDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.plainText, .commaSeparatedText, .tabSeparatedText] }
    var text: String

    init(text: String) { self.text = text }
    init(configuration: ReadConfiguration) throws {
        text = String(decoding: configuration.file.regularFileContents ?? Data(), as: UTF8.self)
    }
    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: Data(text.utf8))
    }
}

enum Clipboard {
    static func copy(_ text: String) {
        #if os(macOS)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        #else
        UIPasteboard.general.string = text
        #endif
    }
}
