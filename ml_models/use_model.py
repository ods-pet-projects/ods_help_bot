from sentence_transformers import SentenceTransformer
from functools import wraps
import numpy as np
from utils.base_classes import BaseEmbedder
from utils.indexer_utils import get_text_by_ind, get_new_ind_by_ind, prepare_indexer, test_queries
from text_utils.utils import create_logger
from config import logger_path, ModelNames

logger = create_logger(__name__, logger_path['use'])

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


class STEmbedder(BaseEmbedder):
    """
    Embedding Wrapper on SentenceTransformer Multilingual
    """

    def __init__(self):
        self.model_file = 'distilbert-multilingual-nli-stsb-quora-ranking'
        self.model = self.bert_model()
        self.success_count = 0
        self.error_count = 0

    @singleton
    def bert_model(self):
        model = SentenceTransformer(self.model_file).eval()
        return model

    def sentence_embedding(self, text):
        try:
            sent_embedding = self.model.encode(text)
            self.success_count += 1
            return sent_embedding
        except:
            logger.exception('exception msg %s', text)
            self.error_count += 1
        return np.zeros(FEATURE_SIZE)


def check_indexer():
    for q in test_queries:
        print('____', q)
        ans_list = get_answer(q)
        for ans in ans_list:
            print('\t\t', ans['text'].replace('\n', ''))
        print()
        print()


def get_answer(query):
    ans_list = [get_text_by_ind(ind) for k, ind in indexer.return_closest(query, k=4)]
    return ans_list


def get_answer_ind(query):
    ind_list = [get_new_ind_by_ind(ind) for k, ind in indexer.return_closest(query, k=4)]
    return ind_list


from config import used_models

if ModelNames.USE in used_models:
    logger.info('use indexer started')
    indexer, df = prepare_indexer('use', logger)
    logger.info('use indexer ready')
    check_indexer()
