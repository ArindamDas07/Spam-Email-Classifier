from loguru import logger
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ---------------- Model Configuration ----------------
MODEL_PATH = "models/bert-spam"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SpamModel:
    _model = None
    _tokenizer = None #_ indicates these are meant to be private/internal.

     #means this method can be called on the class itself (SpamModel.load()) without creating an instance.
    @classmethod 
     #cls refers to the class SpamModel.
    def load(cls):
        if cls._model is None:
            try:
                logger.info(f"Loading model from {MODEL_PATH} on {DEVICE}")
                cls._tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
                cls._model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
                cls._model.to(DEVICE)
                #Sets the model to evaluation mode.
                #This disables dropout and other training-specific behavior, making predictions deterministic.
                cls._model.eval()

                # Warm-up to avoid cold start latency
                cls._warmup()
                logger.success("Model loaded successfully")
            except Exception as e:
                logger.exception(f"Error loading model: {e}")
                raise RuntimeError(f"Failed to load model from {MODEL_PATH}") from e    
        return cls._model,cls._tokenizer  

    @classmethod 
    def _warmup(cls):
        """Run a dummy inference to reduce first-request latency"""
        try:
            dummy_input = cls._tokenizer(
                "This is a test email",
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            dummy_input = {k: v.to(DEVICE) for k, v in dummy_input.items()}

            with torch.no_grad():
                _ = cls._model(**dummy_input)

            logger.info("Model warm-up completed")

        except Exception as e:
            logger.warning(f"Warm-up failed: {e}")    