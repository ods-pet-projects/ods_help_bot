import unittest
from unittest.mock import Mock, patch
from chatbot import next_button
from text_utils.utils import prepare_ans


class TestHandler(unittest.TestCase):
    def setUp(self) -> None:
        self.questions = ['recommender systems', 'bert', 'bpe',
                          'что такое ODS?', 'как стать специлаистом в Data Science?',
                          'bimorf',
                          'Pet projects'
                          ]

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

    def test_prepare_ans(self):
        examples = [
            ('career', 'В общем я решился начать, а разбираться буду по ходу дела',
             'Желающие могут добавиться в wait-list на интервью: <https://forms.gle/HxGyijUtnxMCj6JbA>'),
            (
                'random_talk',
                'В каком-то из разговоров на работе выяснилось, что среди всех коллег никто никогда не участвовал в мородобое',
                '<@ods_help_bot|ods_help_bot> Это :scream: и это многое объясняет.')
        ]

        for channel, text, ans_text in examples:
            ans = prepare_ans(channel, text, ans_text)
            self.assertTrue(ans)

    def test_bpe_ans(self):
        from ml_models.bert_model import get_answer
        for q in self.questions:
            ans_list = get_answer(q)
            print(q)
            for ans in ans_list:
                print('\t\t', ans)
            self.assertTrue(len(ans_list) > 0)

    def test_bert_ans(self):
        from ml_models.bert_model import get_answer
        for q in self.questions:
            ans_list = get_answer(q)
            print(q)
            for ans in ans_list:
                print('\t\t', ans)
            self.assertTrue(len(ans_list) > 0)
