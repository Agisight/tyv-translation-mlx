import SwiftUI

/// Вкладка «Настройки»
struct SettingsView: View {
    @AppStorage(Settings.streamingKey) private var streaming = true
    @AppStorage(Settings.maxCharsKey) private var maxChars = Settings.defaultMaxChars

    var body: some View {
        Form {
            Section("Вывод") {
                Toggle("Показывать перевод по мере генерации (токен за токеном)", isOn: $streaming)
                Text("Выключено — перевод появляется целиком, когда фраза готова.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Длинные тексты") {
                Stepper(value: $maxChars, in: 100...300, step: 25) {
                    Text("Максимальная длина фрагмента: \(maxChars) символов")
                }
                Text("""
                Модель дообучена на отдельных предложениях до 300 символов. Длинный текст режется \
                на предложения, а слишком длинные предложения — на части по знакам препинания, и каждая \
                часть переводится отдельно. Так качество остаётся таким же, как в оценке на тесте.
                """)
                .font(.caption).foregroundStyle(.secondary)
            }
            Section("Модель") {
                LabeledContent("Модель", value: "Gemma 4 E4B, сокращённый словарь")
                LabeledContent("Размер", value: "3.2 ГБ, 6 бит")
                LabeledContent("Hugging Face", value: Translator.modelID)
            }
        }
        .formStyle(.grouped)
        .padding()
    }
}
