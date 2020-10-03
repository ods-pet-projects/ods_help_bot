from transformers import BertTokenizer, BertForQuestionAnswering
import torch

from text_utils.utils import create_logger

logger = create_logger('bert_squad', 'bert_squad.log')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

question, text = "Who was Jim Henson?", "Mr Henson Jim was a nice puppet"
logger.info('text: %s', text)
logger.info('')
logger.info('question: %s', question)
input_ids = tokenizer.encode(question, text)
token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))

logger.info('start_scores: %s', start_scores)
logger.info('end_scores: %s', end_scores)

all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
answer = ' '.join(all_tokens[torch.argmax(start_scores): torch.argmax(end_scores)+1])

logger.info('answer: %s', answer)
assert answer == "a nice puppet"
