import numpy as np
import pandas as pd
import os
import glob
from support_model import get_answer, ModelNames

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
        return pred_ans in true_ans or true_ans in pred_ans
        
    def get_metric(self, true_ans, ans_list):
        acc_top_1 = int(self.is_equal(true_ans, ans_list[0]))
        acc_top_3 = int(any(self.is_equal(true_ans, ans) for ans in ans_list[:3]))
        return acc_top_1, acc_top_3

    def build_report(self, questions, true_answers):
        lines = []
        wrong_ans = []
        for model_name in ModelNames:
            score_top_1 = 0
            score_top_3 = 0
            n = self.eval_ds.shape[0]
            elapsed_list = []

            for q, true_ans in zip(questions, true_answers):
                start = pd.Timestamp.now()
                ans_list = get_answer(q, model_name=model_name)
                elapsed = pd.Timestamp.now() - start
                if len(ans_list) > 0:
                    acc_top_1, acc_top_3 = self.get_metric(true_ans, ans_list)
                    if acc_top_1 == 0:
                        err = dict(
                            model = model_name,
                            true_answer = true_ans,
                            model_answer = ans_list[0]
                        )
                        wrong_ans.append(err)
                else:
                    acc_top_1 = 0
                    acc_top_3 = 0

                score_top_1 += acc_top_1
                score_top_3 += acc_top_3
                elapsed_list.append(elapsed)

            line = dict(
                model=model_name.value,
                acc_top_1=score_top_1 / n,
                acc_top_3=score_top_3 / n,
                elapsed=str(np.mean(elapsed_list)).replace("0 days ", "")
            )
            lines.append(line)

        errors = pd.DataFrame(wrong_ans)
        errors.to_csv('/home/payonear/ods_help_bot/data/errors.csv', index = False)
        report = pd.DataFrame(lines)
        return report
    
    def create_report(self):
        questions = self.eval_ds['text']
        true_answers = self.eval_ds['proc_answer']
        self.report = self.build_report(questions, true_answers)

    def print_report(self):
        if self.report is not None:
            print(self.report)
        else:
            print('Report was not built!')
