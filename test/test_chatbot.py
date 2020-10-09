import unittest
from unittest.mock import Mock, patch
from chatbot import next_button


class TestHandler(unittest.TestCase):
    def test_next(self):
        user_data = {
            'last_answers': [1, 2, 3, 4],
            'curr_ans': 0,
            'assessment_has_set': False
        }
        update = Mock()
        context = Mock()
        context.user_data = user_data

        next_button(update, context)
        next_button(update, context)
        next_button(update, context)
        self.assertEqual(context.user_data['curr_ans'], 3)
        with self.assertRaises(Exception):
            next_button(update, context)

    def test_get_answer(self):
        from ml_models.elastic_search_baseline import get_answer
        with patch('support_model.elastic_search_baseline') as es_mock:
            es_mock.get_answer = get_answer
            from support_model import get_answer
            ans = get_answer('Big data examples')
            self.assertLessEqual(len(ans), 4)

            ans2 = get_answer('asfdsdffhjgjkgjhfghdfhgfgsdfasdfdfg')
            self.assertEqual(ans2, ['not found :(\nPlease paraphrase your query'])
