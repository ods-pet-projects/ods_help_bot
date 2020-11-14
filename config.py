import os

if os.getcwd().endswith('help_bot'):
    ROOT_DIR = os.getcwd()
else:
    ROOT_DIR = os.getcwd() + '/..'
DATA = f'{ROOT_DIR}/data'
LOG_DIR = f'{ROOT_DIR}/logs'
# ifile_train_path = f'{DATA}/ods_answers.csv'
# ifile_train_path = f'{DATA}/ods_answers_eval.csv'
ifile_train_path = f'{DATA}/ods_new_base.csv'

index_path = {  
    'bert': f'{DATA}/bert_index_100K',
    'use': f'{DATA}/st_bert_index',
    'bpe': f'{DATA}/bpe_index'
    }

logger_path = {
    'bert': f'{LOG_DIR}/bert_model.log',
    'use': f'{LOG_DIR}/use_model.log',
    'bpe': f'{LOG_DIR}/bpe_model.log',
    # 'elastic': f'{LOG_DIR}/elastic_model.log'
}