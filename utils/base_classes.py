from abc import ABC, abstractmethod

class BaseEmbedder(ABC):
    
    @abstractmethod
    def sentence_embedding(self, text):
        pass


class BaseIndexer(ABC):

    @abstractmethod
    def load_index(self, index_path, data):
        pass

    @abstractmethod
    def create_index(self, index_path, data):
        pass

    @abstractmethod
    def save_index(self, index_path, data):
        pass
    
    @abstractmethod
    def make_data_embeddings(self, data):
        pass

    @abstractmethod
    def return_closest(self, text, k=2, num_threads=2):
        pass

