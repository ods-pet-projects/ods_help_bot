# ods_help_bot
## About
Telegram bot trained on Open Data Science slack history
- `chatbot.py` - telegram bot for messaging
- `support_model.py` - question answering AI
- `scripts/prepare_data.py` - dataset parsing
- `models` - answering models
- - `elasticsearch_baseline` - fuzzy search
- - `bert_emb_baseline` - BERT embedding ranking model
- - `bpe_baseline` - BPE embedding ranking model
- - `tfidf` - TFIDF classification model
- Run bot `python chatbot.py`
- Available in telegram by address `@ods_help_bot`
## Features
- answer question
- estimate question
- add new question with answer
- show history

## Finite automaton
![](https://github.com/ods-pet-projects/ods_help_bot/blob/master/_chatbot_graph.png)
