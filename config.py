import os

if os.getcwd().endswith('help_bot'):
    ROOT_DIR = os.getcwd()
else:
    ROOT_DIR = os.getcwd() + '/..'
DATA = f'{ROOT_DIR}/data'
