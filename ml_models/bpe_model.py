from functools import wraps
import numpy as np
from utils.base_classes import BaseEmbedder
from utils.indexer_utils import get_text_by_ind, get_new_ind_by_ind, prepare_indexer, test_queries
from text_utils.utils import create_logger
from config import logger_path
from bpemb import BPEmb

logger = create_logger(__name__, logger_path['bpe'])

FEATURE_SIZE = 300

def singleton(cls):
    instance = None

    @wraps(cls)
    def inner(*args, **kwargs):
        nonlocal instance
        if instance is None:
            instance = cls(*args, **kwargs)
        return instance

    return inner

class BPEEmbedder(BaseEmbedder):
    def __init__(self):
        self.model = self.build_model()
        self.success_count = 0
        self.error_count = 0

    @singleton
    def build_model(self):
        model = BPEmb(lang="multi", vs=1000000, dim=FEATURE_SIZE)
        return model

    def sentence_embedding(self, text):
        try:
            mean_emb = self.model.embed(text).mean(axis=0)
            self.success_count += 1
            return mean_emb / np.linalg.norm(mean_emb)
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

logger.info('bpe indexer started')
indexer, df = prepare_indexer('bpe', logger)
logger.info('bpe indexer ready')
check_indexer()
