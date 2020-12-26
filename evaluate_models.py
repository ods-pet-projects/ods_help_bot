import glob
import numpy as np
import os
import pandas as pd
from support_model import get_answer
from config import DATA, ModelNames


ifile_all_path = f'{DATA}/labelled_all.csv'
if os.path.exists(ifile_all_path):
    files = glob.glob(f'{DATA}/labelled/*.csv')
    print(files)
    df_list = [pd.read_csv(f) for f in files]
    df = pd.concat(df_list, ignore_index=False)
    df['source'] = 'labelled'
    df.to_csv(ifile_all_path, index=False)
else:
    df = pd.read_csv(ifile_all_path)

print('labelled data shape: ', df.shape)
print('labelled data cols: ', df.columns.tolist())
#%%


# def get_conf_mtx(y_true, y_pred, df):
#     labels = sorted(set(y_true) | set(y_pred))
#     mtx = confusion_matrix(y_true, y_pred, labels=labels)
#     new_dict = {k: v for k, v in zip(df['url_3'].values, df['section_1'].values)}
#     new_labels = [new_dict.get(k, "empty") for k in labels]
#     df_mtx = pd.DataFrame(mtx, columns=new_labels, index=new_labels)
#     new_order = sorted(new_labels)
#     df_mtx = df_mtx.loc[new_order, new_order]
#     return df_mtx

def is_equal(true_ans, pred_ans):
    return pred_ans in true_ans or true_ans in pred_ans
           # or nltk.edit_distance(true_ans, pred_ans) / len(true_ans) < 0.4


def print_scores(questions, true_answers):
    lines = []
    for model_name in ModelNames:
        print(model_name)
        score_top_1 = 0
        score_top_3 = 0
        n = len(df)
        elapsed_list = []
        model_ans_list = []
        true_ans_list = []

        for q, true_ans in zip(questions, true_answers):
            start = pd.Timestamp.now()
            ans_list = get_answer(q, model_name=model_name)
            elapsed = pd.Timestamp.now() - start
            true_ans_list.append(true_ans)
            # print('\t\t', true_ans[:100], ans_list[0][:100])
            if len(ans_list) > 0:
                model_ans_list.append(ans_list)
                acc_top_1 = int(is_equal(true_ans, ans_list[0]))
                acc_top_3 = int(any(is_equal(true_ans, ans) for ans in ans_list[:3]))
                if acc_top_1 or acc_top_3:
                    print('\t\t', q)
                    print('\t\t', true_ans)
                    print('\t\t', ans_list[:3])
            else:
                model_ans_list.append("empty")
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
        print('\tscore_top_1', score_top_1)
        print('\tscore_top_3', score_top_3)
        # mtx_df = get_conf_mtx(true_ans_list, model_ans_list, df)
        # mtx_df.to_csv(f'output/conf_mtx_{model_name}.csv')
        # print(mtx_df.shape)
        # print(mtx_df)
        # print(model_name.value, score_top_1 / n, score_top_3 / n, np.mean(elapsed_list))

    report = pd.DataFrame(lines)
    return report


for slice_name, df_slice in df.groupby('source'):
    df_slice = df_slice[['full question', 'answer']].dropna()

    questions = df_slice['full question'].values
    true_answers = df_slice['answer'].values

    print('start score for slice ', slice_name)
    print('data shape: ', df_slice.shape)
    report = print_scores(questions, true_answers)
    report.to_csv(f'{DATA}/output/report_{slice_name}.csv', index=False)
    print(slice_name)
    print(report)
