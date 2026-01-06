def clean_text(text: str) -> str:
    text = text.strip()
    text = text.replace("\n", " ")
    return text
