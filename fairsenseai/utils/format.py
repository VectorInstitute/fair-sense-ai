from typing import List


def highlight_bias(text: str, bias_words: List[str]) -> str:
    """
    Highlights bias words in the text with inline HTML styling.
    """
    if not bias_words:
        return f"<div>{text}</div>"
    for word in bias_words:
        text = text.replace(
            word,
            f"<span style='color: red; font-weight: bold;'>{word}</span>"
        )
    return f"<div>{text}</div>"