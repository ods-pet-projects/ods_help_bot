import unittest
questions = ['Существуют ли в открытом доступе корпуса текстов "компьютерной" тематики?',
        'Всем привет! Появилась идея обучить транслятор из комментариев в код. Есть ли параллельные корпуса для такого рода задач?',
        'Всем привет!\nПодскажите, пожалуйста, где можно набрать датасет для бинарной классификации, нужно по фото определить лицо/не лицо?'
        ]
class TestIndexer(unittest.TestCase):

    def test_bert_indexer(self):
        from ml_models.bert_model import indexer
        for q in questions:
            r = indexer.model.sentence_embedding(q)
            self.assertTrue(r.shape[0]==768) # embedding dimension

            res = indexer.index.knnQueryBatch(queries=[r], k=4, num_threads=2)
            self.assertTrue(len(res[0][0])==4) # return 4 neighbours
            self.assertTrue(res[0][1][0]<1e-5) # find itself, distance apx. 0


    def test_use_indexer(self):
        from ml_models.use_model import indexer
        for q in questions:
            r = indexer.model.sentence_embedding(q)
            self.assertTrue(r.shape[0]==768) # embedding dimension

            res = indexer.index.knnQueryBatch(queries=[r], k=4, num_threads=2)
            self.assertTrue(len(res[0][0])==4) # return 4 neighbours
            self.assertTrue(res[0][1][0]<1e-5) # find itself, distance apx. 0

    def test_bpe_indexer(self):
        from ml_models.bpe_model import indexer
        for q in questions:
            r = indexer.model.sentence_embedding(q)
            self.assertTrue(r.shape[0]==300) # embedding dimension

            res = indexer.index.knnQueryBatch(queries=[r], k=4, num_threads=2)
            self.assertTrue(len(res[0][0])==4) # return 4 neighbours
            self.assertTrue(res[0][1][0]<1e-5) # find itself, distance apx. 0
    
def main():
    tester = TestIndexer()
    tester.test_bert_indexer()
    tester.test_use_indexer()
    tester.test_bpe_indexer()