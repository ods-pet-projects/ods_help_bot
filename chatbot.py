import logging
import os
from functools import partial

from telegram import ParseMode, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, PicklePersistence, CallbackQueryHandler

from ml_models.elastic_search_baseline import msg_to_doc, add_doc_to_index
from support_model import get_answer, MODEL_NAME, model_name_dict

my_persistence = PicklePersistence(filename='persistence.pickle')

TOKEN = os.environ['TOKEN']

replies = {
    'start': "Please send me query and I'll answer! Send /info to view all commands",
    'info': """Hello\! My name is ODS Support Bot \(call me Sup\)\. 
I can answer on your questions\.
Do not shy ask me about everything, I\'ll try to help\.
Commands:
/start \- show start message
/showmodel \- show current model
/elastic \- `elasticsearch` ranking
/bert \- `bert` embedding ranking
/bpe \- `bpe` embedding ranking
/tfidf \- `tfidf` classification model
/info \- show this info message
/addquestion \- add question and answer to improve bot
/history \- show all user queries
""",
    'assessment_thanks': 'Thank you for your estimation :)',
    'question_added': 'Thank you for adding question!',
    'question_add_info': '''To add answer and question please send msg in format Text \\n Q1? A1 link1 \\n Q2? A2 link2.
    (words Question and Answer not included):
`/addquestion Some information text. First sentence about thing1. Second sentence about thing2.
What about sentence1? thing1_title. thing1. https://link
What about sentence2? thing2_title. thing2. https://link
`''',
    'question_add_failed': ' adding question failed. Please reformat question.',
    'assessment_query': 'Please send chatbot answer assessment:'
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
c_handler = logging.StreamHandler()
os.makedirs('logs', exist_ok=True)
f_handler = logging.FileHandler('logs/chatbot.log')

c_handler.setLevel(logging.DEBUG)
f_handler.setLevel(logging.DEBUG)

logger_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(logger_format)
f_handler.setFormatter(logger_format)
logger.addHandler(c_handler)
logger.addHandler(f_handler)

logger.debug('Support chat bot started')


def start(update, context):
    logger.info('start msg %s', replies['start'])
    context.user_data['model_name'] = MODEL_NAME
    if 'history' not in context.user_data:
        context.user_data['history'] = []
    if 'user_dict' not in context.bot_data:
        context.bot_data['user_dict'] = {}
    context.bot.send_message(chat_id=update.effective_chat.id,
                             text=replies['start'])
    context.bot_data['user_dict'][update.effective_chat.id] = update.message.chat.username


def info(update, context):
    logger.info('info msg %s', replies['info'])
    context.bot.send_message(chat_id=update.effective_chat.id,
                             text=replies['info'],
                             parse_mode=ParseMode.MARKDOWN_V2)


def set_model(update, context, model_name=None):
    if not model_name:
        model_name = update.message.text.replace('/setmodel ', '').replace('/setmodel', '')
    logger.info('setmodel msg set model %s @%s', model_name, update.message.chat.username)

    if not model_name:
        context.bot.send_message(chat_id=update.effective_chat.id,
                                 text='empty model name')
    elif model_name in model_name_dict:
        context.user_data['model_name'] = model_name_dict[model_name]
        context.bot.send_message(chat_id=update.effective_chat.id,
                                 text='set model `{}`'.format(model_name),
                                 parse_mode=ParseMode.MARKDOWN_V2)
    else:
        context.bot.send_message(chat_id=update.effective_chat.id,
                                 text='not found model {}'.format(model_name))


def show_model(update, context):
    logger.info('show model msg %s', context.user_data['model_name'])
    context.bot.send_message(chat_id=update.effective_chat.id,
                             text='current model `{}`'.format(context.user_data['model_name'].value),
                             parse_mode=ParseMode.MARKDOWN_V2)


def reply(update, context):
    query = update.message.text
    model_name = context.user_data['model_name']
    if query.isdigit() and 0 <= int(query) <= 5:
        logger.info('reply user assessment: @%s %s %s', update.message.chat.username, model_name, query)
        context.bot.send_message(chat_id=update.effective_chat.id,
                                 text=replies['assessment_thanks'], parse_mode=ParseMode.HTML)
    else:
        if 'history' not in context.user_data:
            context.user_data['history'] = []
        else:
            context.user_data['history'].append(query)

        answer = get_answer(query, model_name=model_name)
        logger.info('reply msg model_name: `%s`', model_name)
        logger.info('reply msg query: `%s`', query)
        logger.info('reply msg answer: `%s`', answer)
        logger.info('reply msg to user: @%s', update.message.chat.username)
        keyboard = [[InlineKeyboardButton("Good", callback_data='Good'),
                     InlineKeyboardButton("Bad", callback_data='Bad')],
                    [InlineKeyboardButton("Live chat", callback_data='Live chat')]]

        reply_markup = InlineKeyboardMarkup(keyboard)
        context.bot.send_message(chat_id=update.effective_chat.id,
                                 text=answer, parse_mode=ParseMode.HTML)
        context.bot.send_message(chat_id=update.effective_chat.id,
                                 text=replies['assessment_query'], parse_mode=ParseMode.HTML,
                                 reply_markup=reply_markup)


def add_question(update, context):
    input_msg = update.message.text.replace('/addquestion ', '').replace('/addquestion', '')
    if input_msg == '':
        context.bot.send_message(chat_id=update.effective_chat.id,
                                 text=replies['question_add_info'])
    else:
        logger.info('addquestion msg %s @%s', input_msg, update.message.chat.username)
        doc_list = msg_to_doc(input_msg)
        if doc_list:
            for doc in doc_list:
                add_doc_to_index(doc)
                context.bot.send_message(chat_id=update.effective_chat.id,
                                         text=str(doc) + '\n' + replies['question_added'])
                logger.info('addquestion add %s @%s', doc, update.message.chat.username)
        else:
            context.bot.send_message(chat_id=update.effective_chat.id,
                                     text=replies['question_add_failed'])


def assessment_button(update, context):
    query = update.callback_query
    query.answer()
    logger.info('reply assessment from user: @%s %s', update.effective_chat.username, query.data)
    query.edit_message_text(text=f"{replies['assessment_thanks']}")


def show_history(update, context):
    logger.info('show history msg %s', update.message.chat.username)
    history = '\n'.join(context.user_data.get('history', 'empty'))
    context.bot.send_message(chat_id=update.effective_chat.id,
                             text=f'User history:\n{history}')


# create chat bot
updater = Updater(token=TOKEN, use_context=True, persistence=my_persistence)
dispatcher = updater.dispatcher

dispatcher.add_handler(CommandHandler('start', start))
dispatcher.add_handler(CommandHandler('info', info))

dispatcher.add_handler(CommandHandler('setmodel', set_model))
for model_name in model_name_dict.keys():
    dispatcher.add_handler(CommandHandler(model_name, partial(set_model, model_name=model_name)))
dispatcher.add_handler(CommandHandler('showmodel', show_model))
dispatcher.add_handler(CommandHandler('history', show_history))
dispatcher.add_handler(CommandHandler('addquestion', add_question))
dispatcher.add_handler(CallbackQueryHandler(assessment_button))
dispatcher.add_handler(MessageHandler(Filters.text & (~Filters.command), reply))

updater.start_polling()
