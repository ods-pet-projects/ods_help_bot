import numpy as np
import pandas as pd
import os
import glob
from gensim.summarization import keywords
from gensim.parsing.preprocessing import remove_stopwords
import enum
from config import DATA, ifile_train_path
from support_model import MODEL_NAME, ModelNames, get_keywords, remove_stop_words_func
from ml_models import elastic_search_baseline, bert_model, bpe_model, use_model

class Evaluator:
    def __init__(self, eval_data_dir, eval_files_dir):
        self.eval_ds = self.create_eval_ds(eval_data_dir, eval_files_dir)
        self.report = None

    def create_eval_ds(self, eval_data_dir, eval_files_dir):
        if not os.path.exists(eval_data_dir):
            files = glob.glob(eval_files_dir)
            df_list = [pd.read_csv(f) for f in files]
            df = pd.concat(df_list, ignore_index=False)
            df['source'] = 'labelled'
            df.to_csv(eval_data_dir, index=False)
        else:
            df = pd.read_csv(eval_data_dir)
        return df
    
    def is_equal(self, true_ans, pred_ans):
        return pred_ans == true_ans
        
    def get_metric(self, true_ans, ans_list):
        acc_top_1 = int(self.is_equal(true_ans, ans_list[0]))
        acc_top_4 = int(any(self.is_equal(true_ans, ans) for ans in ans_list[:4]))
        return acc_top_1, acc_top_4

    def build_report(self, questions, true_answers):
        lines = []
        wrong_ans = []
        train_df = pd.read_csv(ifile_train_path)
        for model_name in ModelNames:
            score_top_1 = 0
            score_top_4 = 0
            n = self.eval_ds.shape[0]
            elapsed_list = []

            for q, true_ans in zip(questions, true_answers):
                start = pd.Timestamp.now()
                ans_list = get_answer_ind(q, model_name=model_name)
                elapsed = pd.Timestamp.now() - start
                if len(ans_list) > 0:
                    acc_top_1, acc_top_4 = self.get_metric(true_ans, ans_list)
                    if acc_top_1 == 0:
                        err = dict(
                            model = model_name,
                            true_answer = train_df[train_df['new_ind'] == true_ans].text,
                            model_answer = train_df[train_df['new_ind'] == ans_list[0]].text
                        )
                        wrong_ans.append(err)
                else:
                    acc_top_1 = 0
                    acc_top_4 = 0

                score_top_1 += acc_top_1
                score_top_4 += acc_top_4
                elapsed_list.append(elapsed)

            line = dict(
                model=model_name.value,
                acc_top_1=score_top_1 / n,
                acc_top_4=score_top_4 / n,
                elapsed=str(np.mean(elapsed_list)).replace("0 days ", "")
            )
            lines.append(line)

        errors = pd.DataFrame(wrong_ans)
        errors.to_csv(DATA + '/errors.csv', index = False)
        report = pd.DataFrame(lines)
        return report
    
    def create_report(self):
        questions = self.eval_ds['question']
        true_answers = self.eval_ds['new_ind']
        self.report = self.build_report(questions, true_answers)

    def print_report(self):
        if self.report is not None:
            print(self.report)
        else:
            print('Report was not built!')


def get_answer_ind(query, use_lower=True, use_keywords=False, use_remove_stopwords=False, model_name=MODEL_NAME):
    if use_lower:
        query = query.lower()
    if use_keywords:
        query = get_keywords(query)
    if use_remove_stopwords:
        query = remove_stop_words_func(query)

    try:
        answer_list = []
        if model_name == ModelNames.ELASTIC:
            answer_list = elastic_search_baseline.get_answer_ind(query)
        if model_name == ModelNames.BERT:
            answer_list = bert_model.get_answer_ind(query)
        if model_name == ModelNames.BPE:
            answer_list = bpe_model.get_answer_ind(query)
        if model_name == ModelNames.USE:
            answer_list = use_model.get_answer_ind(query)
        return answer_list
    except Exception as ex:
        print(ex)
        return ["not found :(\nPlease paraphrase your query"]
