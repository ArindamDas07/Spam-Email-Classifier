from celery import Celery
import torch

from app.model import SpamModel
from app.preprocess import clean_text
from app.redis_conn import redis_client

celery = Celery(
    "worker",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/0"
)

@celery.task(bind=True)
def classify_email(self, text: str):
    model, tokenizer = SpamModel.load()

    text = clean_text(text)

    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        prob = torch.softmax(outputs.logits, dim=-1)[0, 1].item()

    label = "Spam" if prob >= 0.7 else "Not Spam"

    redis_client.set(self.request.id, label)

    return label
