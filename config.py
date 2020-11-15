import os

if os.getcwd().endswith('help_bot'):
    ROOT_DIR = os.getcwd()
else:
    ROOT_DIR = os.getcwd() + '/..'
DATA = f'{ROOT_DIR}/data'
LOG_DIR = f'{ROOT_DIR}/logs'
INDEX_DIR = f'{ROOT_DIR}/pretrained_indexes'
# ifile_train_path = f'{DATA}/ods_answers.csv'
# ifile_train_path = f'{DATA}/ods_answers_eval.csv'
ifile_train_path = f'{DATA}/ods_new_base.csv'

index_path = {  
    'bert': f'{INDEX_DIR}/bert_index',
    'use': f'{INDEX_DIR}/st_bert_index',
    'bpe': f'{INDEX_DIR}/bpe_index'
    }

logger_path = {
    'bert': f'{LOG_DIR}/bert_model.log',
    'use': f'{LOG_DIR}/use_model.log',
    'bpe': f'{LOG_DIR}/bpe_model.log',
    # 'elastic': f'{LOG_DIR}/elastic_model.log'
}

indexer_map = {
    'bert': 'nmslib',
    'use': 'nmslib',
    'bpe': 'nmslib'
}