import unittest

from chatbot import get_answer_from_api


class TestApi(unittest.TestCase):
    def test_find(self):
        ans_list = get_answer_from_api('test', 'elastic')
        self.assertEqual(4, len(ans_list))
