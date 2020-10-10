# %%
import numpy as np
import pandas as pd
from config import DATA
# %%
df = pd.read_csv(f'{DATA}/ods_slack_all.csv')
# %%
df.head()
# %%
channels = [
    'lang_python',
    'big_data',
    'datasets',
    'lang_r',
    'nlp',
    'deep_learning'
]
# %%
# data = df.query('channel in @channels')
# print('init shape:', len(data))
# data = data.dropna(subset=['text'])
# data = data[~data['text'].str.contains(' has joined the channel')]
# data['text_len'] = data['text'].str.len()
# data = data.query('text_len > 23')
need_cols = ['client_msg_id', 'channel', 'text']
df = df[need_cols]
print('after filtering len:', len(df))
# %%
df.query('channel in @channels').to_csv('/Users/o/PycharmProjects/ods_help_bot/data/channels_posts.csv', index=False)
# %%
df.to_csv('/Users/o/PycharmProjects/ods_help_bot/data/channels_posts_all.csv', index=False)
