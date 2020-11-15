from sentence_transformers import SentenceTransformer
from functools import wraps
import numpy as np
from utils.base_classes import BaseEmbedder
from indexers.nmslib_indexer  import get_text_by_ind, prepare_indexer
from text_utils.utils import create_logger
from config import logger_path

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

logger.info('use indexer started')
indexer, df = prepare_indexer('use', logger)
logger.info('use indexer ready')
check_indexer()
