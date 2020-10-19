import os

if os.getcwd().endswith('help_bot'):
    ROOT_DIR = os.getcwd()
else:
    ROOT_DIR = os.getcwd() + '/..'
DATA = f'{ROOT_DIR}/data'
# ifile_train_path = f'{DATA}/ods_answers.csv'
ifile_train_path = f'{DATA}/ods_answers_eval.csv'
