from functools import wraps
import os
import pandas as pd
import nmslib
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from config import DATA
from text_utils.utils import prepare_ans

FEATURE_SIZE = 512
MAX_TEXT_LEN = 300


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
    Embedding Wrapper on SentenceTransformer Multilingual Cased
    """

    def __init__(self):
        self.model_file = 'distilbert-multilingual-nli-stsb-quora-ranking'
        self.model = self.bert_model()
        self.FEATURE_SIZE = FEATURE_SIZE

    @singleton
    def bert_model(self):
        model = SentenceTransformer(self.model_file).eval()
        return model

    def sentence_embedding(self, text):
        sent_embedding = self.model.encode(text)
        return sent_embedding

    def encode(self, data):
        names_sparse_matrix = []
        for i in tqdm(range(len(data))):
            try:
                names_sparse_matrix.append(self.sentence_embedding(data[i]))
            except:
                names_sparse_matrix.append(np.zeros(self.FEATURE_SIZE))
        return names_sparse_matrix


class STIndexer:
    def __init__(self, bert_model=None):
        self.model = bert_model or STEmbedder()
        self.space_type = 'cosinesimil'
        self.method_name = 'hnsw'
        self.index = nmslib.init(space=self.space_type,
                                 method=self.method_name,
                                 data_type=nmslib.DataType.DENSE_VECTOR,
                                 dtype=nmslib.DistType.FLOAT)
        self.index_params = {'NN': 15}
        self.index_is_loaded = False
        self.data = []

    def load_index(self, index_path, data):
        self.index.loadIndex(index_path, load_data=True)
        self.index_is_loaded = True
        self.data = data

    def create_index(self, index_path, data):
        start = pd.Timestamp.now()
        if not os.path.exists(index_path):
            names_sparse_matrix = self.make_data_embeddings(data)
            self.index.addDataPointBatch(data=names_sparse_matrix)
            self.index.createIndex(print_progress=True)
            self.index.saveIndex(index_path, save_data=True)
        else:
            self.load_index(index_path, data)
            print('loaded from', index_path)
        self.index_is_loaded = True
        self.data = data
        print('elapsed:', pd.Timestamp.now() - start)

    def create_embedding(self, text):
        return self.model.sentence_embedding(text)

    def make_data_embeddings(self, data):
        names_sparse_matrix = []
        for i in tqdm(range(len(data))):
            try:
                names_sparse_matrix.append(self.model.sentence_embedding(data[i]))
            except:
                names_sparse_matrix.append(np.zeros(FEATURE_SIZE))
        return names_sparse_matrix

    def return_closest(self, text, k=2, num_threads=2):
        if self.index_is_loaded:
            r = self.model.sentence_embedding(text)
            near_neighbors = self.index.knnQueryBatch(queries=[r], k=k, num_threads=num_threads)
            return [(self.data[i], i) for i in near_neighbors[0][0]]
        else:
            raise IndexError("Index is not yet created or loaded")


def prepare_indexer():
    indexer = STIndexer()
    df = pd.read_csv(f'{DATA}/ods_answers.csv')
    df = df.sort_values('pos_score', ascending=False)
    data = df['text']
    indexer.create_index(f'{DATA}/st_bert_index', data.values)
    return indexer, df


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
            print('\t\t', ans)
        print()
        print()


def get_text_by_ind(ind):
    ans_row = df.iloc[ind]
    channel = ans_row['channel']
    text = ans_row['text']
    ans_text = ans_row['answer_text']
    return prepare_ans(channel, text, ans_text, MAX_TEXT_LEN)


def get_answer(query):
    ans_list = [get_text_by_ind(ind) for k, ind in indexer.return_closest(query, k=4)]
    return ans_list


print('load/train st_bert indexer started')
indexer, df = prepare_indexer()
print('st_bert indexer ready')
check_indexer()
