import json
import os
import re
import requests


def get_slack_answer_ind(query, k=4):
    url = f"https://slack.com/api/search.messages"
    params = {"count": k, "query": query, "pretty": 1}

    headers = {
        'Authorization': f'Bearer {os.environ["USER_TOKEN"]}'
    }

    response = requests.request("GET", url, headers=headers, params=params)
    matches = json.loads(response.text).get('messages').get('matches')
    ind_list = [get_ind(match) for match in matches]
    return ind_list


def get_ind(match):
    user = match.get('user')
    truncated_ts = re.sub(r'\.\d+', r'.0', match.get('ts')) # TODO in labelled_all.csv at the end of the timestamp always zero
    return '_'.join((user, truncated_ts))
