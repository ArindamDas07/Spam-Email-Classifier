import re

def clean_text(text: str) -> str:
    if not text:
        return ""
    
    # 1. Basic Cleaning
    text = text.strip()
    
    # 2. Remove HTML Tags (Very common in emails)
    text = re.sub(r'<[^>]+>', '', text)
    
    # 3. Replace URLs with a single word (Optional but helps BERT)
    text = re.sub(r'http\S+|www\S+|https\S+', 'url', text, flags=re.MULTILINE)

    # 4. Remove tabs/newlines and collapse multiple spaces
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)

    # 5. Consistency
    text = text.lower()
    
    return text