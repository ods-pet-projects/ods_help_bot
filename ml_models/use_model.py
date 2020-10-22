from sentence_transformers import SentenceTransformer
from functools import wraps
import numpy as np

FEATURE_SIZE = 768

def singleton(cls):
    instance = None

    @wraps(cls)
    def inner(*args, **kwargs):
        nonlocal instance
        if instance is None:
            instance = cls(*args, **kwargs)
        return instance

    return inner

class STEmbedder:
    """
    Embedding Wrapper on SentenceTransformer Multilingual
    """

    def __init__(self):
        self.model_file = 'distilbert-multilingual-nli-stsb-quora-ranking'
        self.model = self.bert_model()
        self.error_count = 0

    @singleton
    def bert_model(self):
        model = SentenceTransformer(self.model_file).eval()
        return model

    def sentence_embedding(self, text):
        try:
            sent_embedding = self.model.encode(text)
            return sent_embedding
        except:
            self.error_count += 1
        return np.zeros(FEATURE_SIZE)