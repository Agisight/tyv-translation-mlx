"""Общая функция перевода: одинаковый промпт при оценке, тестах и в приложении."""
import re

from mlx_lm import generate

from prepare_data import PROMPTS, SYSTEM


def build_prompt(tok, src: str, direction: str) -> str:
    messages = [{"role": "system", "content": SYSTEM}] if SYSTEM else []
    messages.append({"role": "user", "content": PROMPTS[direction].format(src=src)})
    # Qwen3 — модель с режимом рассуждений; для перевода он не нужен
    return tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )


def translate(model, tok, src: str, direction: str, max_tokens: int = 256) -> str:
    out = generate(model, tok, prompt=build_prompt(tok, src, direction),
                   max_tokens=max_tokens, verbose=False)
    return re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
