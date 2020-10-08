from functools import wraps
import os
import pandas as pd
import nmslib
import numpy as np
import torch
from tqdm import tqdm
from pytorch_pretrained_bert import BertModel, BertTokenizer
from config import DATA


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

    def __init__(self, path=''):
        # use self.model_file with a path instead of 'bert-base-uncased' if you have a custom pretrained model
        self.model_file = 'bert-base-uncased'  # os.path.join(path, "bert-base-multilingual-cased.tar.gz")
        self.vocab_file = 'bert-base-uncased'  # os.path.join(path, "data_bert-base-multilingual-cased-vocab.txt")
        self.model = self.bert_model()
        self.tokenizer = self.bert_tokenizer()
        self.embedding_matrix = self.get_bert_embed_matrix()
        self.FEATURE_SIZE = 768

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

    def sentences_embedding(self, text_list):
        embeddings = []
        for text in tqdm(text_list):
            token_list = self.tokenizer.tokenize("[CLS] " + text + " [SEP]")
            segments_ids, indexed_tokens = [1] * len(token_list), self.tokenizer.convert_tokens_to_ids(token_list)
            segments_tensors, tokens_tensor = torch.tensor([segments_ids]), torch.tensor([indexed_tokens])
            with torch.no_grad():
                encoded_layers, _ = self.model(tokens_tensor, segments_tensors)
            sent_embedding = torch.mean(encoded_layers[11], 1)
            embeddings.append(sent_embedding)
        return embeddings

    def token_embedding(self, token_list):
        token_embedding = []
        for token in token_list:
            ontoken = self.tokenizer.tokenize(token)
            segments_ids, indexed_tokens = [1] * len(ontoken), self.tokenizer.convert_tokens_to_ids(ontoken)
            segments_tensors, tokens_tensor = torch.tensor([segments_ids]), torch.tensor([indexed_tokens])
            with torch.no_grad():
                encoded_layers, _ = self.model(tokens_tensor, segments_tensors)
            ontoken_embeddings = []
            for subtoken_i in range(len(ontoken)):
                hidden_layers = []
                for layer_i in range(len(encoded_layers)):
                    vector = encoded_layers[layer_i][0][subtoken_i]
                    hidden_layers.append(vector)
                ontoken_embeddings.append(hidden_layers)
            cat_last_4_layers = [torch.cat((layer[-4:]), 0) for layer in ontoken_embeddings]
            token_embedding.append(cat_last_4_layers)
        token_embedding = torch.stack(token_embedding[0], 0) if len(token_embedding) > 1 else token_embedding[0][0]
        return token_embedding

    def encode(self, data):
        names_sparse_matrix = []
        for i in tqdm(range(len(data))):
            try:
                names_sparse_matrix.append(self.sentence_embedding(data[i])[0].numpy())
            except:
                names_sparse_matrix.append(np.zeros(768))
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

        class IndexError(Exception):
            """Base class for other exceptions"""
            pass

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
                names_sparse_matrix.append(np.zeros(768))
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
    df = pd.read_csv(f'{DATA}/help_title_v2.csv')
    df = df.query('section_4 not in ["Overview", "Important Information", "More Info", "Learn More", "More Resources"]')
    df = df.query('section_2 not in ["Gantt Chart Overview"]')
    df.reset_index(inplace=True)
    data = df['section_2'] + ' ' + df['section_3'] + ' ' + df['section_4']
    # data = df['section_1'] + ' / ' + df['section_2'] + ' / ' + df['section_3'] + ' / ' + df['section_4'] + \
    #        ' / ' + df['keywords'] + ' / ' + df['section_text']
    indexer.create_index('bert_index', data.values)

    test_queries = ['gantt chart', 'export report', 'support MS', 'share data', 'Machine Learning',
                    'how to create a task',
                    'how to reorder a subtask',
                    'where is user settings',
                    'how to export to Excel',
                    'how to reset a user password',
                    'how to reset a password',
                    'how to move task to another column']

    for q in test_queries:
        print('____', q)
        for x, ind in indexer.return_closest(q, k=4):
            ans = ' / '.join(df.loc[ind, ['section_2', 'section_3', 'section_4']].values)
            print('\t\t\t', ans, df.loc[ind, 'url_4'])
        print()
        print()
    return indexer, df


print('load/train bert indexer started')
try:
    indexer, df = prepare_indexer()
except:
    pass
print('bert indexer ready')


def format_msg(ind):
    ans = ' / '.join(df.loc[ind, ['section_2', 'section_3', 'section_4']].values)
    line = f"{ans}, {df.loc[ind, 'url_4']}"
    return line


def get_answer(query):
    ans_list = [f'\n_____ {format_msg(ind)}' for k, ind in indexer.return_closest(query, k=4)]
    return ans_list