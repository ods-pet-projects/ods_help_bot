import os
import nmslib
import faiss
from sklearn.metrics import pairwise_distances
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
from utils.base_classes import BaseIndexer
from config import INDEX_DIR, indexer_map

class NMSlibIndexer(BaseIndexer):
    def __init__(self, model_name, logger):
        if model_name == 'bert':
            from ml_models.bert_model import BertEmbedder
            self.model = BertEmbedder()
        elif model_name == 'use':
            from ml_models.use_model import STEmbedder
            self.model = STEmbedder()
        elif model_name == 'bpe':
            from ml_models.bpe_model import BPEEmbedder
            self.model = BPEEmbedder()
        else:
            raise ValueError(f'Model {model_name} not found.')
        self.model_name = model_name
        self.space_type = 'cosinesimil'
        self.method_name = 'hnsw'
        self.index = nmslib.init(space=self.space_type,
                                 method=self.method_name,
                                 data_type=nmslib.DataType.DENSE_VECTOR,
                                 dtype=nmslib.DistType.FLOAT)
        self.index_is_loaded = False
        self.data = []
        self.logger = logger

    def load_index(self, index_path, data):
        self.index.loadIndex(index_path, load_data=True)
        self.index_is_loaded = True
        self.data = data
    
    def save_index(self, index_path, save_data=True):
        self.index.saveIndex(index_path, save_data)

    def create_index(self, index_path, data):
        start = pd.Timestamp.now()
        if not os.path.exists(f'{INDEX_DIR}/{indexer_map[self.model_name]}'):
            os.mkdir(f'{INDEX_DIR}/{indexer_map[self.model_name]}')
        if not os.path.exists(index_path):
            self.logger.info(f'train {self.model_name} indexer started')
            names_sparse_matrix = self.make_data_embeddings(data)
            self.index.addDataPointBatch(data=names_sparse_matrix)
            self.index.createIndex(print_progress=True)
            self.save_index(index_path, save_data=True)
        else:
            self.logger.info(f'found prepared {self.model_name} index, loading...')
            self.load_index(index_path, data)
            self.logger.info('loaded from %s', index_path)
        self.index_is_loaded = True
        self.data = data
        self.logger.info('elapsed: %s', pd.Timestamp.now() - start)

    def make_data_embeddings(self, data):
        names_sparse_matrix = []
        for _, text_prepared in tqdm(enumerate(data)):
            names_sparse_matrix.append(self.model.sentence_embedding(text_prepared))
        self.logger.info('make embeddings finished')
        self.logger.info('pos count %s', self.model.success_count)
        self.logger.info('neg count %s', self.model.error_count)
        return names_sparse_matrix

    def return_closest(self, text, k=2, num_threads=2):
        if self.index_is_loaded:
            r = self.model.sentence_embedding(text)
            near_neighbors = self.index.knnQueryBatch(queries=[r], k=k, num_threads=num_threads)
            return [(self.data[i], i) for i in near_neighbors[0][0]]
        else:
            raise IndexError("Index is not yet created or loaded")


class FaissIndexer(BaseIndexer):
    def __init__(self, model_name, logger):
        if model_name == 'bert':
            from ml_models.bert_model import BertEmbedder
            self.model = BertEmbedder()
        elif model_name == 'use':
            from ml_models.use_model import STEmbedder
            self.model = STEmbedder()
        elif model_name == 'bpe':
            from ml_models.bpe_model import BPEEmbedder
            self.model = BPEEmbedder()
        else:
            raise ValueError(f'Model {model_name} not found.')
        self.model_name = model_name
        self.space_type = 'cosinesimil'
        self.index_is_loaded = False
        self.data = []
        self.logger = logger

    def load_index(self, index_path, data):
        self.index = faiss.read_index(index_path)
        self.index_is_loaded = True
        self.data = data

    def save_index(self, index_path):
         faiss.write_index(self.index, index_path)

    def create_index(self, index_path, data):
        start = pd.Timestamp.now()
        if not os.path.exists(f'{INDEX_DIR}/{indexer_map[self.model_name]}'):
            os.mkdir(f'{INDEX_DIR}/{indexer_map[self.model_name]}')
        if not os.path.exists(index_path):
            self.logger.info(f'train {self.model_name} indexer started')
            names_matrix = self.make_data_embeddings(data)
            features_dim = names_matrix.shape[1]
            if self.space_type == 'cosinesimil':
                self.index = faiss.index_factory(features_dim, "Flat", faiss.METRIC_INNER_PRODUCT)
                faiss.normalize_L2(names_matrix)
            else:
                print('Metric not found')
            self.index.add(names_matrix)
            self.save_index(index_path)
        else:
            self.logger.info(f'found prepared {self.model_name} index, loading...')
            self.load_index(index_path, data)
            self.logger.info('loaded from %s', index_path)
        self.index_is_loaded = True
        self.data = data
        self.logger.info('elapsed: %s', pd.Timestamp.now() - start)

    def make_data_embeddings(self, data):
        names_matrix = []
        for _, text_prepared in tqdm(enumerate(data)):
            names_matrix.append(self.model.sentence_embedding(text_prepared))
        self.logger.info('make embeddings finished')
        self.logger.info('pos count %s', self.model.success_count)
        self.logger.info('neg count %s', self.model.error_count)
        return np.array(names_matrix).astype('float32')

    def return_closest(self, text, k=2):
        k = min(k, self.index.ntotal)
        if self.index_is_loaded:
            r = self.model.sentence_embedding(text)
            r = np.array(r).astype('float32')
            r = np.expand_dims(r, axis=0)
            if self.space_type == 'cosinesimil':
                faiss.normalize_L2(r)
            _, indexs = self.index.search(r, k)
            return [(self.data[i], i) for i in list(indexs[0])]
        else:
            raise IndexError("Index is not yet created or loaded")