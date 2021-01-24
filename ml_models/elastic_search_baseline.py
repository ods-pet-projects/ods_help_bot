from time import sleep
import subprocess
from elasticsearch import Elasticsearch
import pandas as pd
import sys
import re

sys.path.append('..')
from config import DATA, ifile_train_path, MAX_ANSWER_COUNT
from text_utils.utils import prepare_ans

MAX_TEXT_LEN = 600
es = Elasticsearch()

INDEX_NAME = "ods-index"


def get_answer_first(query):
    res = es.search(index=INDEX_NAME,
                    body={'query': {'match': {
                        'text':
                            {'query': query,
                             'operator': 'OR'}
                    }}})
    answer, _ = print_res(res)
    return answer


def get_answer(query):
    try:
        return get_answer_first(query)
    except:
        subprocess.call('systemctl restart elasticsearch', shell=True)
        sleep(10)
        return get_answer_first(query)


def get_answer_ind(query):
    res = es.search(index=INDEX_NAME,
                    body={'query': {'match': {
                        'text':
                            {'query': query,
                             'operator': 'OR'}
                    }}})
    _, ind = print_res(res)
    return ind


def print_res(res):
    if len(res['hits']['hits']) > 0:
        ans_list = []
        ind_list = []
        for i, item in enumerate(res['hits']['hits']):
            if i > MAX_ANSWER_COUNT:
                break
            doc_preview_text = item['_source']['show_text'] + '...'
            doc_ind = item['_source']['doc_id']
            if doc_preview_text not in ans_list:
                ans_list.append(doc_preview_text)
                ind_list.append(doc_ind)
        return ans_list, ind_list
    else:
        return ["not found :(\nPlease paraphrase your query"], [0]


def get_doc_title(doc_text):
    lines = doc_text.split('\n')
    doc_title = f'{lines[0]}/{lines[1]}'
    return doc_title


def add_doc_to_index(doc):
    try:
        es.index(index=INDEX_NAME, doc_type="text", body=doc)
        return True
    except Exception as ex:
        print(ex)
    return False


def find_urls(string):
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex, string)
    return [x[0] for x in url]


def build_index():
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME, ignore=[400, 404])
    # doc_dir = DATA
    # data = pd.read_csv(f'{doc_dir}/channels_posts_all.csv')
    # data = pd.read_csv(f'{doc_dir}/channels_posts.csv')
    # data = pd.read_csv(f'{doc_dir}/ods_answers.csv')
    data = pd.read_csv(ifile_train_path)
    meetings_cols = [x for x in data['channel'].values if x.startswith('_meetings')]
    excluded_channels = ['random_b', 'bimorf', 'topkek', 'random_b'] + meetings_cols
    init_shape = data.shape[0]
    data = data.query('channel not in @excluded_channels')
    new_shape = data.shape[0]
    print('filtered channels', excluded_channels)
    print('pref shape:', init_shape)
    print('new shape:', new_shape)
    print('dropped:', init_shape - new_shape)
    success_count = 0
    doc_id = 0
    data = data.dropna(subset=['answer_text'])

    for doc_id, doc_row in data.iterrows():
        client_msg_id = doc_row['new_ind']
        channel = doc_row['channel']
        text = doc_row['text']
        answer_text = doc_row['answer_text']

        doc_links = find_urls(text)
        show_text = prepare_ans(channel, text, answer_text, MAX_TEXT_LEN)
        doc = {
            "doc_id": client_msg_id,
            "doc_title": channel,
            "text": text,
            "answer_text": answer_text,
            "show_text": show_text,
            "link": doc_links
        }

        if add_doc_to_index(doc):
            success_count += 1
            print(success_count)

    print(f'success rate {success_count / (doc_id + 1):0.2f}')
    check_elastic_ans()


def check_elastic_ans():
    test_q_list = [
        'Gantt chart',
        'Machine learning',
        'assign subtasks so that each team member (or team members)',
        'New tasks are created as backlogged by default and remain that way unless you add start and end dates to the task or convert it to a milestone',
        'what information is or is not copied when you duplicate tasks',
        'what information is or is not copied when you duplicate task',
        'what information about duplicated task',
        'How to move task to another column?'
    ]
    for test_q in test_q_list:
        answer = get_answer(test_q)
        print(test_q)
        print('\t', answer)


def run_elastic_circle():
    while True:
        query = input('input query:\n')
        if query == 'stop':
            break
        print('Your query:', query)
        answer = get_answer(query)
        print(answer)
        print()


index_ready = False
if not es.indices.exists(index=INDEX_NAME):
    build_index()
    index_ready = True


def main():
    build_index()
    # check_elastic_ans()
    # run_elastic_circle()


if __name__ == '__main__':
    main()

# curl -X DELETE "localhost:9200/ods-index?pretty"
