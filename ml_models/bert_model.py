from transformers import BertModel, BertTokenizer
from functools import wraps
import numpy as np
from utils.base_classes import BaseEmbedder
from utils.indexer_utils import get_text_by_ind, get_new_ind_by_ind, prepare_indexer, test_queries
from text_utils.utils import create_logger
from config import logger_path
import torch

logger = create_logger(__name__, logger_path['bert'])

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


class BertEmbedder(BaseEmbedder):
    """
    Embedding Wrapper on Bert Multilingual Uncased
    """

    def __init__(self, model_spec = 'bert-base-multilingual-uncased'):
        self.model_spec = model_spec
        self.model = self.bert_model()
        self.tokenizer = self.bert_tokenizer()
        self.success_count = 0
        self.error_count = 0

    @singleton
    def bert_model(self):
        model = BertModel.from_pretrained(self.model_spec).eval()
        return model

    @singleton
    def bert_tokenizer(self):
        do_lower_case = False
        if 'uncased' in self.model_spec:
            do_lower_case = True
        tokenizer = BertTokenizer.from_pretrained(self.model_spec, do_lower_case=do_lower_case)
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
            logger.exception('exception msg %s', text)
            self.error_count += 1
        return np.zeros(FEATURE_SIZE)


def check_indexer():
    for q in test_queries:
        print('____', q)
        ans_list = get_answer(q)
        for ans in ans_list:
            print('\t\t', ans.replace('\n', ''))
        print()
        print()


def get_answer(query):
    ans_list = [get_text_by_ind(ind) for k, ind in indexer.return_closest(query, k=4)]
    return ans_list

def get_answer_ind(query):
    ind_list = [get_new_ind_by_ind(ind) for k, ind in indexer.return_closest(query, k=4)]
    return ind_list

logger.info('bert indexer started')
indexer, df = prepare_indexer('bert', logger)
logger.info('bert indexer ready')
check_indexer()
