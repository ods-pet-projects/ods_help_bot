# -*- coding: utf-8 -*-
import gspread
from gensim.summarization import keywords
from gensim.parsing.preprocessing import remove_stopwords

from config import ModelNames, MODEL_NAME, used_models

if ModelNames.ELASTIC in used_models:
    from ml_models import elastic_search_baseline
else:
    elastic_search_baseline = 1

if MODEL_NAME.BERT in used_models:
    from ml_models import bert_model
else:
    bert_model = 1

if MODEL_NAME.BPE in used_models:
    from ml_models import bpe_model
else:
    bpe_model = 1

if MODEL_NAME.USE in used_models:
    from ml_models import use_model
else:
    use_model = 1


def get_keywords(query):
    return keywords(query)


def remove_stop_words_func(query):
    return remove_stopwords(query)


def remove_slack_commands(query):
    slack_commands = {
        "<!everyone>": "everyone",
        "<!channel>": "channel",
        "<!here>": "here"
    }
    for k, v in slack_commands.items():
        query = query.replace(k, v)
    return query


def get_answer_if_known(query):
    gc = gspread.service_account(filename='ods-qna-secret.json')
    questions = gc.open("faq fixed").sheet1

    list_of_dicts = questions.get_all_records()

    for dct in list_of_dicts:
        if query in dct['known_question']:
            return [dct['known_answer']]


def get_answer(query, use_lower=True, use_keywords=False, use_remove_stopwords=False, model_name=MODEL_NAME,
               use_remove_slack_commands=True
               ):
    if use_lower:
        query = query.lower()
    if use_keywords:
        query = get_keywords(query)
    if use_remove_stopwords:
        query = remove_stop_words_func(query)
    if use_remove_slack_commands:
        query = remove_slack_commands(query)

    try:
        known_answer = get_answer_if_known(query)
        if known_answer:
            return known_answer

        answer_list = []
        if model_name == ModelNames.ELASTIC:
            answer_list = elastic_search_baseline.get_answer(query)
        if model_name == ModelNames.BERT:
            answer_list = bert_model.get_answer(query)
        if model_name == ModelNames.BPE:
            answer_list = bpe_model.get_answer(query)
        if model_name == ModelNames.USE:
            answer_list = use_model.get_answer(query)
        return answer_list
    except Exception as ex:
        print('exception:', ex)
        return ["not found :(\nPlease paraphrase your query"]


def main():
    query = 'How plot gantt Chart?'
    answer = get_answer(query)
    print(answer)


if __name__ == '__main__':
    main()
