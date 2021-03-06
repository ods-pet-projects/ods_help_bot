from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
from flask_restplus import Api, Resource
import sys
import os
from config import logger_path, model_name_dict, MODEL_NAME
from text_utils.utils import create_logger

sys.path.append('..')
from support_model import get_answer

# get_post_desc = lambda x: {"questionId": "0", "question": "how are you?", "answer": "I'm fine thanks"}
# get_answer = lambda x, model_name: ["test 1", "test 2", "test 3", "test 4"]

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
cors = CORS(app, resources={r"/api/chat/v1*": {"origins": "*"}})
api = Api(app)
logger = create_logger(__name__, logger_path['app'])


@api.route('/api/v1/find', methods=['GET'])
@api.doc(params={'query': 'text question',
                 'model_name': 'bert, sbert, bpe, elastic'})
class Answer(Resource):
    def get(self):
        q = request.args.get('query', '')
        if len(q) > 1:
            model_name = request.args.get('model_name')
            model_name = model_name_dict.get(model_name, MODEL_NAME)
            logger.info(q)
            ans_list_init = get_answer(q, model_name=model_name)
            logger.info(ans_list_init)
            answer_list = [a for a in ans_list_init]
            return jsonify(answer_list)
        return {'text': "not found",
                'channel_id': '0',
                'timestamp': '0'}


# @api.route('/api/v1/answer', methods=["GET"])
# @api.doc(params={'answer_id': 'answer id'})
# class AnswerDescription(Resource):
#     def get(self):
#         answer_id = request.args.get('answer_id')
#         description = get_post_desc(answer_id)
#         description['questionId'] = str(description['questionId'])
#         return jsonify(description)

'''
GET /find?q=Text+here
[
  {"questionId": 32, "question": "Как сделать X", "score": 0.8989},
  {"questionId": 18, "question": "Как сделать Y", "score": 0.108},
  ...
]

GET /answer/{id}
{
  "questionId": 32,
  "question": "Как сделать X",
  "answer": "Возьмите 10 г обычной советской..."
}
'''


if __name__ == '__main__':
    use_port = os.environ.get('APP_PORT') or 8080
    app.run(debug=True, host='0.0.0.0', port=use_port)
