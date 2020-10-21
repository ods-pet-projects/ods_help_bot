import os

if os.getcwd().endswith('help_bot'):
    ROOT_DIR = os.getcwd()
else:
    ROOT_DIR = os.getcwd() + '/..'
DATA = f'{ROOT_DIR}/data'
LOG_DIR = f'{ROOT_DIR}/logs'
# ifile_train_path = f'{DATA}/ods_answers.csv'
ifile_train_path = f'{DATA}/ods_answers_eval.csv'
