#!/usr/bin/env python
# coding: utf-8

# In[5]:
from functools import lru_cache

import numpy as np
import pandas as pd
from bpemb import BPEmb
import os
import re
from config import DATA
from text_utils.utils import replace_name, prepare_ans

bpe = BPEmb(lang="multi", vs=1000000, dim=300)

MAX_TEXT_LEN = 600


def get_embedding(sentence):
    mean_emb = bpe.embed(sentence).mean(axis=0)
    return mean_emb / np.linalg.norm(mean_emb)


def load_data():
    # df = pd.read_csv(f'{DATA}/channels_posts.csv')
    ifile_path = f'{DATA}/ods_answers.csv'
    cache_file_path = f'{DATA}/ods_answers_bpe_cache.csv'

    if os.path.exists(cache_file_path):
        df = pd.read_csv(cache_file_path)
        df['emb'] = df['emb'].apply(lambda x: np.array(eval(re.sub(' +', ', ', x).replace('[,', '['))))
        df = df.sort_values('pos_score', ascending=False)
    else:
        df = pd.read_csv(ifile_path)
        df['emb'] = df['text'].apply(get_embedding)
        df.to_csv(cache_file_path, index=False)
    return df


def get_answer(request):
    request_embedding = get_embedding(request)
    data["distance"] = data["emb"].apply(lambda x: sum(x * request_embedding))
    data.sort_values("distance", ascending=False, inplace=True)
    data_ans = data.head(4)
    ans_values = data_ans.apply(
                lambda row: prepare_ans(row['channel'], row['text'], row['answer_text'], MAX_TEXT_LEN
    ), axis=1).values
    return ans_values


data = load_data()


def main():
    questions = ['Есть ли аналоги pandas (ну или не аналоги а тоже либы для работы с данными) для работы с данными',
                 'как стать kaggle grandmaster?',
                 'Что такое BERT?',
                 'что такое random_b?',
                 '''Привет! Хочу найти синонимы для слова в контексте (2-3 слова). 
                 я не верю что для такой задачи нужен трансформер, как BERT или RoBERTa. 
                 Что думаете? Каким было бы ваше решение в лоб?''',
                 'Подскажите, пожалуйста, с чего начать изучение NLP? Можете посоветовать какие-нибудь курсы?',
                 'рекомендательные системы',
                 'где почитать про метрики?',
                 'где найти информацию про многоруких бандитов?'
                 ]

    for question in questions:
        print('_____________________\n:>', question)
        ans_list = get_answer(question)
        for ans in ans_list:
            print('\t\t', ans)
        print()
        print()
        print()


if __name__ == '__main__':
    main()
