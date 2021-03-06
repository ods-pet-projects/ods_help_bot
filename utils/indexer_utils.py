import pandas as pd
from text_utils.utils import prepare_ans
from config import indexer_map, index_path, ifile_train_path
from indexers.nmslib_indexer import NMSlibIndexer

df = pd.read_csv(ifile_train_path)
MAX_TEXT_LEN = 300


def prepare_indexer(model_name, logger):
    if indexer_map[model_name] == 'nmslib':
        indexer = NMSlibIndexer(model_name, logger)
    else:
        raise ValueError(f'Wrong indexer for {model_name} model specified')
    indexer.create_index(index_path[model_name], df['text'].values)
    return indexer, df


def get_text_by_ind(ind):
    ans_row = df.iloc[ind]
    channel = ans_row['channel']
    text = ans_row['text']
    ans_text = ans_row['answer_text']
    channel_id, timestamp = ans_row['new_ind'].split('_')
    return prepare_ans(channel, text, ans_text, MAX_TEXT_LEN, channel_id, timestamp)


def get_new_ind_by_ind(ind):
    ans_row = df.iloc[ind]
    return ans_row['new_ind']


test_queries = ['Есть ли аналоги pandas (ну или не аналоги а тоже либы для работы с данными) для работы с данными',
                'Как стать kaggle grandmaster?',
                'Что такое BERT?',
                'что такое random_b?',
                '''Привет! Хочу найти синонимы для слова в контексте (2-3 слова). 
                    я не верю что для такой задачи нужен трансформер, как BERT или RoBERTa. 
                    Что думаете? Каким было бы ваше решение в лоб?''',
                'Подскажите, пожалуйста, с чего начать изучение NLP? Можете посоветовать какие-нибудь курсы?']
