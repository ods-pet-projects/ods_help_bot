from functools import wraps
import numpy as np
from text_utils.indexer import get_text_by_ind, prepare_indexer
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

class BPEEmbedder:
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
            self.error_count += 1
        return np.zeros(FEATURE_SIZE)

def check_indexer():
    test_queries = ['Есть ли аналоги pandas (ну или не аналоги а тоже либы для работы с данными) для работы с данными',
                    'Как стать kaggle grandmaster?',
                    'Что такое BERT?',
                    'что такое random_b?',
                    '''Привет! Хочу найти синонимы для слова в контексте (2-3 слова). 
                    я не верю что для такой задачи нужен трансформер, как BERT или RoBERTa. 
                    Что думаете? Каким было бы ваше решение в лоб?''',
                    'Подскажите, пожалуйста, с чего начать изучение NLP? Можете посоветовать какие-нибудь курсы?'
                    ]

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

logger.info('bpe indexer started')
indexer, df = prepare_indexer('bpe', logger)
logger.info('bpe indexer ready')
check_indexer()