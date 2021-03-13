import enum
import os

if os.getcwd().endswith('help_bot'):
    ROOT_DIR = os.getcwd()
else:
    ROOT_DIR = os.getcwd() + '/..'
DATA = f'{ROOT_DIR}/data'
LOG_DIR = f'{ROOT_DIR}/logs'
INDEX_DIR = f'{ROOT_DIR}/pretrained_indexes'
EVAL_FILES_DIR = f'{DATA}/alt_eval/*.csv'
EVAL_DATA_DIR= f'{DATA}/labelled_all.csv'
# ifile_train_path = f'{DATA}/ods_answers.csv'
# ifile_train_path = f'{DATA}/ods_answers_eval.csv'
ifile_train_path = f'{DATA}/ods_new_base.csv'

index_path = {
    'bert': 'bert_index',
    'use': 'st_bert_index',
    'bpe': 'bpe_index'
}

logger_path = {
    'bert': f'{LOG_DIR}/bert_model.log',
    'use': f'{LOG_DIR}/use_model.log',
    'bpe': f'{LOG_DIR}/bpe_model.log',
    # 'elastic': f'{LOG_DIR}/elastic_model.log'
    'app': f'{LOG_DIR}/app.log',
}

indexer_map = {
    'bert': 'faiss',
    'use': 'faiss',
    'bpe': 'faiss'
}

API_URL = 'http://0.0.0.0:8080/api/v1'
MAX_ANSWER_COUNT = 4


class ModelNames(enum.Enum):
    ELASTIC = 'elastic'
    BERT = 'bert'
    BPE = 'bpe'
    USE = 'use'
    SLACK = 'slack'


# default model
MODEL_NAME = ModelNames.ELASTIC
model_name_dict = {x.value: x for x in ModelNames}
used_models = [ModelNames.ELASTIC, ModelNames.BPE, ModelNames.BERT, ModelNames.USE]
