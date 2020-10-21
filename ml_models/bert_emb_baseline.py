from functools import wraps
import os
import pandas as pd
import nmslib
import numpy as np
import torch
from tqdm import tqdm
tqdm.pandas()
from pytorch_pretrained_bert import BertModel, BertTokenizer
from config import DATA, ifile_train_path, LOG_DIR
from text_utils.utils import prepare_ans, create_logger

FEATURE_SIZE = 768
MAX_TEXT_LEN = 300

logger = create_logger(__name__, f'{LOG_DIR}/bert_emb_baseline.log')

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
        # self.model_file = 'bert-base-uncased'  # os.path.join(path, "bert-base-multilingual-cased.tar.gz")
        # self.vocab_file = 'bert-base-uncased'  # os.path.join(path, "data_bert-base-multilingual-cased-vocab.txt")
        self.model_file = 'bert-base-multilingual-uncased'
        self.vocab_file = 'bert-base-multilingual-uncased'
        self.model = self.bert_model()
        self.tokenizer = self.bert_tokenizer()
        self.embedding_matrix = self.get_bert_embed_matrix()
        self.FEATURE_SIZE = FEATURE_SIZE
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

    @singleton
    def get_bert_embed_matrix(self):
        bert_embeddings = list(self.model.children())[0]
        bert_word_embeddings = list(bert_embeddings.children())[0]
        matrix = bert_word_embeddings.weight.data.numpy()
        return matrix

    def sentence_embedding(self, text):
        try:
            token_list = self.tokenizer.tokenize(text)
            segments_ids, indexed_tokens = [1] * len(token_list), self.tokenizer.convert_tokens_to_ids(token_list)
            segments_tensors, tokens_tensor = torch.tensor([segments_ids]), torch.tensor([indexed_tokens])
            with torch.no_grad():
                encoded_layers, _ = self.model(tokens_tensor, segments_tensors)
            sent_embedding = torch.mean(encoded_layers[11], 1)
            vect = sent_embedding[0].numpy()
            self.success_count += 1
            return vect
        except:
            self.error_count += 1
        return np.zeros(FEATURE_SIZE)


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
            logger.info('train bert indexer started')
            names_sparse_matrix = self.make_data_embeddings(data)
            self.index.addDataPointBatch(data=names_sparse_matrix)
            self.index.createIndex(print_progress=True)
            self.index.saveIndex(index_path, save_data=True)
        else:
            logger.info('found prepared bert index, loading...')
            self.load_index(index_path, data)
            logger.info('loaded from %s', index_path)
        self.index_is_loaded = True
        self.data = data
        logger.info('elapsed: %s', pd.Timestamp.now() - start)

    def make_data_embeddings(self, data):
        # names_sparse_matrix = pd.DataFrame({'text': data})['text'].progress_apply(get_emb_vector, args=(self.model, ))
        names_sparse_matrix = []
        for i, text_prepared in tqdm(enumerate(data)):
            names_sparse_matrix.append(self.model.sentence_embedding(text_prepared))
        logger.info('make embeddings finished')
        logger.info('pos count %s', self.model.success_count)
        logger.info('neg count %s', self.model.error_count)
        return names_sparse_matrix

    def return_closest(self, text, k=2, num_threads=2):
        if self.index_is_loaded:
            r = self.model.sentence_embedding(text)
            near_neighbors = self.index.knnQueryBatch(queries=[r], k=k, num_threads=num_threads)
            return [(self.data[i], i) for i in near_neighbors[0][0]]
        else:
            raise IndexError("Index is not yet created or loaded")


def prepare_indexer():
    # index_file_path = f'{DATA}/bert_index'
    index_file_path = f'{DATA}/bert_index_100K'
    indexer = BertIndexer()
    df = pd.read_csv(ifile_train_path)
    df['text_len'] = df['text'].str.len()
    logger.info('init shape: %s', df.shape)
    logger.info('text_len > 768 %s', sum(df['text_len'] > 768))
    df['text'] = "[CLS] " + df['text'] + " [SEP]"
    df['text'] = df['text'].str.slice(0, FEATURE_SIZE)
    indexer.create_index(index_file_path, df['text'].values)
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
            print('\t\t', ans.replace('\n', ''))
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


logger.info('bert indexer started')
indexer, df = prepare_indexer()
logger.info('bert indexer ready')
check_indexer()
