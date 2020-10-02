from deeppavlov import build_model, configs
import pandas as pd
from config import DATA
from tqdm import tqdm


model = build_model(configs.squad.squad_bert_infer, download=True)
# config = f'{DATA}/squad_config.json'
# model = build_model(configs.squad.squad, download=True)


df = pd.read_csv(f'{DATA}/all_dataset.csv')
df = df.dropna(subset=['url_3', 'section_text'])
df = df.query('source != "from_org"')
print('found train docs:', len(df))

answer_links = df['url_3'].values
docs = df['section_text'].values


def get_answer(query):
    lines = []
    for answer_link, doc in tqdm(zip(answer_links, docs)):
        similarity = model([doc], [query])
        # line = [x[0] for x in similarity] + [answer_link]
        line = [doc[:100], similarity, answer_link]
        lines.append(line)
    ans = pd.DataFrame(lines, columns=["answer", "similarity", "answer_link"])
    # ans = pd.DataFrame(lines, columns=["answer", "num", "similarity", "answer_link"])
    ans = ans.sort_values('similarity', ascending=False)
    return '\n'.join([' '.join(map(str, line)) for line in ans[['answer', 'answer_link']].head(3).values])


def main():
    questions = ['gantt chart',
                 'export report',
                 'support MS',
                 'share data', 'Machine Learning',
                 'how to create a task',
                 'how to reorder a subtask',
                 'where is user settings',
                 'how to export to Excel',
                 'how to reset a user password',
                 'how to reset a password',
                 'how to move task to another column'
                 ]
    for q in questions:
        print(q)
        ans = get_answer(q)
        print(ans)
        print()
        print()


if __name__ == '__main__':
    main()
