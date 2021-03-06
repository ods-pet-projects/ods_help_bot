from elasticsearch import Elasticsearch
import os
import pandas as pd
import re
import subprocess
import sys
from time import sleep


sys.path.append('..')

from utils.indexer_utils import get_text_by_doc_id

from config import INDEX_DIR, ifile_train_path, MAX_ANSWER_COUNT
from text_utils.utils import prepare_ans

MAX_TEXT_LEN = 600
es = Elasticsearch()

INDEX_NAME = "ods-index"
anchor_file_path = f'{INDEX_DIR}/{INDEX_NAME}.anchor'


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
    except Exception as ex:
        print(ex)
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
            ans = get_text_by_doc_id(item['_source']['doc_id'])
            ans_list.append(ans)
            ind_list.append(item['_source']['doc_id'])
        return ans_list, ind_list
    else:
        empty_ans = {'text': "not found :(\nPlease paraphrase your query",
                     'channel_id': "0",
                     "timestamp": "0"
                     }
        return [empty_ans], [0]


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


def last_doc_ind_anchor():
    if not os.path.exists(anchor_file_path):
        return 0

    with open(anchor_file_path) as anchor_file:
        last_ind = anchor_file.read()
    return int(last_ind)


def save_doc_id_anchor(ind):
    with open(anchor_file_path, 'w') as ofile:
        ofile.write(f"{ind}")


def build_index():
    if es.indices.exists(index=INDEX_NAME) and not last_doc_ind_anchor():
        es.indices.delete(index=INDEX_NAME, ignore=[400, 404])
        print('>>>>deleted index', INDEX_NAME)

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

    anchor_lat_ind = last_doc_ind_anchor() + 1  # fix to avoid strange doc

    for doc_id, doc_row in data.iterrows():
        if doc_id <= anchor_lat_ind and anchor_lat_ind != 0:
            continue
        client_msg_id = doc_row['new_ind']
        channel = doc_row['channel']
        text = doc_row['text']
        answer_text = doc_row['answer_text']
        channel_id, timestamp = doc_row['new_ind'].split('_')
        doc_links = find_urls(text)
        ans_dict = prepare_ans(channel, text, answer_text, MAX_TEXT_LEN, channel_id, timestamp)
        doc = {
            "doc_id": client_msg_id,
            "doc_title": channel,
            "text": text[:MAX_TEXT_LEN],
            "answer_text": answer_text,
            "show_text": ans_dict['text'],
            "link": doc_links,
            "channel_id": channel_id,
            "timestamp": timestamp
        }

        if add_doc_to_index(doc):
            success_count += 1
            print(doc_id)
            save_doc_id_anchor(doc_id)

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
