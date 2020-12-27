import os
from functools import partial

from telegram import ParseMode, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, PicklePersistence, CallbackQueryHandler

from text_utils.utils import create_logger
from config import API_URL, MODEL_NAME, model_name_dict, MAX_ANSWER_COUNT
import requests

my_persistence = PicklePersistence(filename='persistence.pickle')

TOKEN = os.environ.get('TOKEN')

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
/use \- `sentence transformer` embedding ranking
/info \- show this info message
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
    'assessment_query': 'Please send chatbot answer assessment:',
    'not_found_msg': 'not found :(\nPlease paraphrase your query'
}

logger = create_logger(__name__, 'logs/chatbot.log')

logger.debug('Support chat bot started')


def start(update, context):
    logger.info('start msg %s', replies['start'])
    context.user_data['model_name'] = MODEL_NAME
    if 'history' not in context.user_data:
        context.user_data['history'] = []
    if 'last_answers' not in context.user_data:
        context.user_data['last_answers'] = []
        context.user_data['curr_ans'] = 0
    if 'user_dict' not in context.bot_data:
        context.bot_data['user_dict'] = {}
    context.user_data['assessment_has_set'] = False
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


def get_answer_from_api(query, model_name):
    return requests.get(f'{API_URL}/find', params=dict(q=query, model_name=model_name)).json()


def reply(update, context):
    query = update.message.text
    model_name = context.user_data['model_name']

    if 'history' not in context.user_data:
        context.user_data['history'] = []
    else:
        context.user_data['history'].append(query)

    ans_list = get_answer_from_api(query, model_name=model_name)

    context.user_data['last_answers'] = ans_list
    context.user_data['assessment_has_set'] = False
    context.user_data['curr_ans'] = 0
    logger.info('reply msg model_name: `%s`', model_name)
    logger.info('reply msg query: `%s`', query)
    logger.info('reply msg answer: `%s`', ans_list)
    logger.info('reply msg to user: @%s', update.message.chat.username)

    reply_markup = None
    if ans_list[0] != replies['not_found_msg']:
        keyboard = [
            [InlineKeyboardButton("Next", callback_data='Next')],
            [InlineKeyboardButton("Good", callback_data='Good'), InlineKeyboardButton("Bad", callback_data='Bad')],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=ans_list[0],
        parse_mode=ParseMode.HTML,
        reply_markup=reply_markup
    )


def next_button(update, context):
    query = update.callback_query
    query.answer()

    query.edit_message_text(
        text=context.user_data['last_answers'][context.user_data['curr_ans']],
        parse_mode=ParseMode.HTML,
    )

    context.user_data['curr_ans'] += 1
    text = context.user_data['last_answers'][context.user_data['curr_ans']]
    keyboard = [
        [InlineKeyboardButton("Next", callback_data='Next')],
    ] if (context.user_data['curr_ans'] != len(context.user_data['last_answers']) - 1) else []

    if not context.user_data['assessment_has_set']:
        keyboard.append([InlineKeyboardButton("Good", callback_data='Good'), InlineKeyboardButton("Bad", callback_data='Bad')])

    reply_markup = InlineKeyboardMarkup(keyboard)

    logger.info('reply msg answer: `%s` to user: @%s', text, update.effective_chat.username)
    logger.info('reply assessment from user: @%s %s', update.effective_chat.username, query.data)

    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=text,
        parse_mode=ParseMode.HTML,
        reply_markup=reply_markup
    )
    logger.info('reply assessment from user: @%s %s', update.effective_chat.username, query.data)


def assessment_button(update, context):
    query = update.callback_query
    query.answer()

    reply_markup = None
    if context.user_data['curr_ans'] != MAX_ANSWER_COUNT and context.user_data['curr_ans'] != len(context.user_data['last_answers']) - 1:
        keyboard = [
            [InlineKeyboardButton("Next", callback_data='Next')],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

    query.edit_message_text(
        text=context.user_data['last_answers'][context.user_data['curr_ans']],
        parse_mode=ParseMode.HTML,
        reply_markup=reply_markup
    )

    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"{replies['assessment_thanks']}",
        parse_mode=ParseMode.HTML,
    )
    context.user_data['assessment_has_set'] = True


def show_history(update, context):
    logger.info('show history msg %s', update.message.chat.username)
    history = '\n'.join(context.user_data.get('history', 'empty'))
    context.bot.send_message(chat_id=update.effective_chat.id,
                             text=f'User history:\n{history}')


def main():
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
    dispatcher.add_handler(CallbackQueryHandler(next_button, pattern="Next"))
    dispatcher.add_handler(CallbackQueryHandler(assessment_button, pattern="Good"))
    dispatcher.add_handler(CallbackQueryHandler(assessment_button, pattern="Bad"))
    dispatcher.add_handler(MessageHandler(Filters.text & (~Filters.command), reply))

    updater.start_polling()


if __name__ == '__main__':
    main()
