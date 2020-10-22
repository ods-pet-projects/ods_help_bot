from transformers import BertModel, BertTokenizer
from functools import wraps
import numpy as np

MAX_TEXT_LEN = 512
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


class BertEmbedder:
    """
    Embedding Wrapper on Bert Multilingual Uncased
    """

    def __init__(self):
        self.model_file = 'bert-base-multilingual-uncased'
        self.vocab_file = 'bert-base-multilingual-uncased'
        self.model = self.bert_model()
        self.tokenizer = self.bert_tokenizer()
        self.success_count = 0
        self.error_count = 0

    @singleton
    def bert_model(self):
        model = BertModel.from_pretrained(self.model_file).eval()
        return model

    @singleton
    def bert_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained(self.vocab_file, do_lower_case=True)
        return tokenizer

    def sentence_embedding(self, text):
        try:
            inputs_ids, token_type_ids, attention_mask = self.tokenizer.encode_plus(text,add_special_tokens = True,
                        max_length = MAX_TEXT_LEN,
                        padding = False,
                        truncation=True,
                        return_tensors = 'pt').values()
            
            with torch.no_grad():
                encoded_layers, _ = self.model(inputs_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            sent_embedding = encoded_layers.mean(dim=1)
            vect = sent_embedding[0].numpy()
            self.success_count += 1
            return vect
        except:
            self.error_count += 1
        return np.zeros(FEATURE_SIZE)