#!/usr/bin/env python
# coding: utf-8

# In[5]:
from functools import lru_cache

import numpy as np
import pandas as pd
from tqdm import tqdm
from bpemb import BPEmb
from config import DATA

# bpe = BPEmb(lang="en", dim=300)
bpe = BPEmb(lang="multi", vs=1000000, dim=300)


# def get_embedding(sentence):
#     emb_av = np.array([t / np.linalg.norm(t) for t in bpe.embed(sentence)]).mean(axis=0)
#     emb_av = emb_av / np.linalg.norm(emb_av)
#     return emb_av

def get_embedding(sentence):
    mean_emb = bpe.embed(sentence).mean(axis=0)
    return mean_emb / np.linalg.norm(mean_emb)


# def get_text_embedding(sentence):
#     word_embs = [get_embedding(w) for w in sentence]
#     word_embs_normed = np.array([t / np.linalg.norm(t) for t in word_embs])
#     emb_av = word_embs_normed.mean(axis=0)
#     emb_av = emb_av / np.linalg.norm(emb_av)
#     return emb_av


def load_data():
    df = pd.read_csv(f'{DATA}/channels_posts.csv')
    df['emb'] = df['text'].map(get_embedding)
    return df


@lru_cache
def get_answer(request):
    request_embedding = get_embedding(request)
    data["distance"] = data["emb"].apply(lambda x: sum(x * request_embedding))
    data.sort_values("distance", ascending=False, inplace=True)
    return data.head(4)['text'].str.slice(0, 300).values


data = load_data()


def main():
    questions = ['Есть ли аналоги pandas (ну или не аналоги а тоже либы для работы с данными) для работы с данными',
                 'как стать kaggle grandmaster?',
                 'Что такое BERT?',
                 'что такое random_b?',
                    '''Привет! Хочу найти синонимы для слова в контексте (2-3 слова). 
                    я не верю что для такой задачи нужен трансформер, как BERT или RoBERTa. 
                    Что думаете? Каким было бы ваше решение в лоб?''',
                 'Подскажите, пожалуйста, с чего начать изучение NLP? Можете посоветовать какие-нибудь курсы?'
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
