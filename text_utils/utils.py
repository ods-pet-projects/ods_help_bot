import logging
from config import LOG_DIR
import os
import re


def create_logger(logger_name: str, logger_path: str) -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(filename=logger_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def replace_name(string):
    return re.sub(r'<@\w*>', '<b>ods_help_bot</b>', string).replace('<http', 'http').replace('<#', '#')


def prepare_ans(channel, text, ans_text, max_text_len=600):
    question_text = text[: max_text_len]
    ans_text = ans_text[: max_text_len]
    return replace_name(f'<b>{channel}:</b>\n{question_text}\n_____{ans_text}')
    