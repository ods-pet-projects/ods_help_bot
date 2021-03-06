import unittest

questions = ['Существуют ли в открытом доступе корпуса текстов "компьютерной" тематики?',
             'Всем привет! Появилась идея обучить транслятор из комментариев в код. Есть ли параллельные корпуса для такого рода задач?',
             'Всем привет! \nПодскажите, пожалуйста, где можно набрать датасет для бинарной классификации, нужно по фото определить лицо/не лицо?'
             ]


class TestIndexer(unittest.TestCase):

    def test_bert_indexer(self):
        from ml_models.bert_model import indexer
        for q in questions:
            r = indexer.model.sentence_embedding(q)
            self.assertEqual(r.shape[0], 768)  # embedding dimension
            dis, res = indexer.return_closest(q, k=4, distance=True)
            self.assertEqual(len(res), 4)  # return 4 neighbours
            self.assertLess(dis[0][0], 1e-5)  # find itself, distance apx. 0

    def test_use_indexer(self):
        from ml_models.use_model import indexer
        for q in questions:
            r = indexer.model.sentence_embedding(q)
            self.assertEqual(r.shape[0], 768)  # embedding dimension

            dis, res = indexer.return_closest(q, k=4, distance=True)
            self.assertEqual(len(res), 4)  # return 4 neighbours
            self.assertLess(dis[0][0], 1e-5)  # find itself, distance apx. 0

    def test_bpe_indexer(self):
        from ml_models.bpe_model import indexer
        for q in questions:
            r = indexer.model.sentence_embedding(q)
            self.assertEqual(r.shape[0], 300)  # embedding dimension

            dis, res = indexer.return_closest(q, k=4, distance=True)
            self.assertEqual(len(res), 4)  # return 4 neighbours
            self.assertLess(dis[0][0], 1e-5)  # find itself, distance apx. 0


def main():
    tester = TestIndexer()
    tester.test_bert_indexer()
    tester.test_use_indexer()
    tester.test_bpe_indexer()


if __name__ == '__main__':
    main()
