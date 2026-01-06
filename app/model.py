import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/bert-spam"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SpamModel:
    _model = None
    _tokenizer = None

    @classmethod
    def load(cls):
        if cls._model is None:
            cls._tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            cls._model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
            cls._model.to(DEVICE)
            cls._model.eval()

        return cls._model, cls._tokenizer
