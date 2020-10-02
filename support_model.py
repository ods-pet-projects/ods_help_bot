# -*- coding: utf-8 -*-
from gensim.summarization import keywords
from gensim.parsing.preprocessing import remove_stopwords
import enum
from ml_models import elastic_search_baseline, bert_emb_baseline, bpe_baseline


class ModelNames(enum.Enum):
    ELASTIC = 'elastic'
    BERT = 'bert'
    BPE = 'bpe'


# default model
MODEL_NAME = ModelNames.ELASTIC

model_name_dict = {x.value: x for x in ModelNames}


def get_keywords(query):
    return keywords(query)


def remove_stop_words_func(query):
    return remove_stopwords(query)


def get_answer(query, use_lower=True, use_keywords=False, use_remove_stopwords=False, model_name=MODEL_NAME):
    if use_lower:
        query = query.lower()
    if use_keywords:
        query = get_keywords(query)
    if use_remove_stopwords:
        query = remove_stop_words_func(query)
    try:
        if model_name == ModelNames.ELASTIC:
            return elastic_search_baseline.get_answer(query)
        if model_name == ModelNames.BERT:
            return bert_emb_baseline.get_answer(query)
        if model_name == ModelNames.BPE:
            return bpe_baseline.get_answer(query)

    except Exception as ex:
        print(ex)
        return "not found :(\nPlease paraphrase your query"


def main():
    query = 'How plot gantt Chart?'
    answer = get_answer(query)
    print(answer)


if __name__ == '__main__':
    main()
