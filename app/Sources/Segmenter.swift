import Foundation
import NaturalLanguage

/// Режет текст на части, которые модель видела при обучении: отдельные предложения до `maxChars` символов.
/// Контекст у Gemma большой, но модель дообучена на предложениях до 300 символов —
/// на длинных абзацах качество падает, поэтому длинное режем.
enum Segmenter {
    /// Отделяет маркер в начале строки: эмодзи, буллеты, тире, нумерацию «1)», «2.».
    /// Кавычки и скобки остаются в тексте — у них есть пара в конце фразы.
    static func leadingMarker(_ line: String) -> (marker: String, body: String) {
        let keep: Set<Character> = ["«", "\"", "„", "“", "'", "(", "["]
        var end = line.startIndex
        while end < line.endIndex, !line[end].isLetter, !keep.contains(line[end]) {
            end = line.index(after: end)
        }
        guard end < line.endIndex else { return (line, "") }  // строка без букв — оставить как есть
        return (String(line[..<end]), String(line[end...]))
    }

    static func segments(of line: String, maxChars: Int) -> [String] {
        let text = line.trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty else { return [] }
        var sentences: [String] = []
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let s = text[range].trimmingCharacters(in: .whitespaces)
            if !s.isEmpty { sentences.append(s) }
            return true
        }
        if sentences.isEmpty { sentences = [text] }
        return sentences.flatMap { split($0, maxChars: maxChars) }
    }

    /// Слишком длинное предложение: сначала по «;», «:», «—», «,», потом по словам
    private static func split(_ sentence: String, maxChars: Int) -> [String] {
        guard sentence.count > maxChars else { return [sentence] }
        for separator in ["; ", ": ", " — ", " – ", ", "] {
            let parts = sentence.components(separatedBy: separator)
            guard parts.count > 1 else { continue }
            var chunks: [String] = []
            var current = ""
            for (i, part) in parts.enumerated() {
                let piece = part + (i < parts.count - 1 ? separator.trimmingCharacters(in: .whitespaces) : "")
                if current.isEmpty {
                    current = piece
                } else if current.count + 1 + piece.count <= maxChars {
                    current += " " + piece
                } else {
                    chunks.append(current)
                    current = piece
                }
            }
            if !current.isEmpty { chunks.append(current) }
            if chunks.count > 1 { return chunks.flatMap { split($0, maxChars: maxChars) } }
        }
        var chunks: [String] = []
        var current = ""
        for word in sentence.split(separator: " ") {
            if current.isEmpty {
                current = String(word)
            } else if current.count + 1 + word.count <= maxChars {
                current += " " + word
            } else {
                chunks.append(current)
                current = String(word)
            }
        }
        if !current.isEmpty { chunks.append(current) }
        return chunks
    }
}
