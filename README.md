# ods_help_bot
## About
This project is a telegram bot trained on Open Data Science slack posts history
## Structure
- `chatbot.py` - telegram bot for messaging
- `support_model.py` - question answering AI
- `scripts/prepare_data.py` - dataset parsing
- `models` - answering models
- - `elasticsearch_baseline` - fuzzy search
- - `bert_emb_baseline` - BERT embedding ranking model
- - `bpe_baseline` - BPE embedding ranking model
- Run bot `python chatbot.py`
- Available in telegram by address `@ods_help_bot`
## Features
- answer question
- show other relevant answer (top 4)
- estimate question (Good, Bad)
- show history
## Running
- You need your own telegram TOKEN from BotFather
- You can run bot in `screen`
- add token to system variables `export TOKEN="somerandomtoken"`
- create virtual environment `python -m venv venv` (optionally)
- `pip install -r requirements.txt`
- installed `elasticsearch` 
- running `systemctl restart elasticsearch`
- `python chatbot.py`
## Finite automaton
![](https://github.com/ods-pet-projects/ods_help_bot/blob/master/_chatbot_graph.png)
