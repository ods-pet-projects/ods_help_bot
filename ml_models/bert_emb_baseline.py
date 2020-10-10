from functools import wraps
import os
import pandas as pd
import nmslib
import numpy as np
import torch
from tqdm import tqdm
from pytorch_pretrained_bert import BertModel, BertTokenizer
from config import DATA
from text_utils.utils import prepare_ans

FEATURE_SIZE = 768
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


class BertEmbedder:
    """
    Embedding Wrapper on Bert Multilingual Cased
    """

    def __init__(self):
        # use self.model_file with a path instead of 'bert-base-uncased' if you have a custom pretrained model
        self.model_file = 'bert-base-uncased'  # os.path.join(path, "bert-base-multilingual-cased.tar.gz")
        self.vocab_file = 'bert-base-uncased'  # os.path.join(path, "data_bert-base-multilingual-cased-vocab.txt")
        self.model = self.bert_model()
        self.tokenizer = self.bert_tokenizer()
        self.embedding_matrix = self.get_bert_embed_matrix()
        self.FEATURE_SIZE = FEATURE_SIZE

    @singleton
    def bert_model(self):
        model = BertModel.from_pretrained(self.model_file).eval()
        return model

    @singleton
    def bert_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained(self.vocab_file, do_lower_case=True)
        return tokenizer

    @singleton
    def get_bert_embed_matrix(self):
        bert_embeddings = list(self.model.children())[0]
        bert_word_embeddings = list(bert_embeddings.children())[0]
        matrix = bert_word_embeddings.weight.data.numpy()
        return matrix

    def sentence_embedding(self, text):
        token_list = self.tokenizer.tokenize("[CLS] " + text + " [SEP]")
        segments_ids, indexed_tokens = [1] * len(token_list), self.tokenizer.convert_tokens_to_ids(token_list)
        segments_tensors, tokens_tensor = torch.tensor([segments_ids]), torch.tensor([indexed_tokens])
        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor, segments_tensors)
        sent_embedding = torch.mean(encoded_layers[11], 1)
        return sent_embedding

    def encode(self, data):
        names_sparse_matrix = []
        for i in tqdm(range(len(data))):
            try:
                names_sparse_matrix.append(self.sentence_embedding(data[i])[0].numpy())
            except:
                names_sparse_matrix.append(np.zeros(self.FEATURE_SIZE))
        return names_sparse_matrix


class BertIndexer:
    def __init__(self, bert_model=None):
        self.model = bert_model or BertEmbedder()
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
        return self.model.sentence_embedding(text).numpy()

    def make_data_embeddings(self, data):
        names_sparse_matrix = []
        for i in tqdm(range(len(data))):
            try:
                names_sparse_matrix.append(self.model.sentence_embedding(data[i])[0].numpy())
            except:
                names_sparse_matrix.append(np.zeros(FEATURE_SIZE))
        return names_sparse_matrix

    def return_closest(self, text, k=2, num_threads=2):
        if self.index_is_loaded:
            r = self.model.sentence_embedding(text).numpy()
            near_neighbors = self.index.knnQueryBatch(queries=[r], k=k, num_threads=num_threads)
            return [(self.data[i], i) for i in near_neighbors[0][0]]
        else:
            raise IndexError("Index is not yet created or loaded")


def prepare_indexer():
    indexer = BertIndexer()
    df = pd.read_csv(f'{DATA}/ods_answers.csv')
    df = df.sort_values('pos_score', ascending=False)
    data = df['text']
    indexer.create_index(f'{DATA}/bert_index', data.values)
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


print('load/train bert indexer started')
indexer, df = prepare_indexer()
print('bert indexer ready')
check_indexer()
