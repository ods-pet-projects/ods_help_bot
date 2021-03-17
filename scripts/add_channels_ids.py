import json
import pandas as pd

ifile_path = '../data/opendatascience Slack export Mar 12 2015 - Mar 6 2021/channels.json'


with open(ifile_path) as ifile:
    channels_json = json.load(ifile)
    lines = [{'channel_slack_id': ch['id'], 'channel': ch['name']} for ch in channels_json]

print(len(channels_json))
df = pd.DataFrame(lines)
print(df)

ifile_base_path = '../data/ods_new_base.csv'
df_all = pd.read_csv(ifile_base_path)

df_all_fixed = pd.merge(df_all, df, on='channel', how='left')
df_all_fixed.to_csv('../data/ods_new_base_v4.csv', index=False)
