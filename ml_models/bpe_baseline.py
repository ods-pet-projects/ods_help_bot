#!/usr/bin/env python
# coding: utf-8

# In[5]:
import numpy as np
import pandas as pd

from bpemb import BPEmb
from config import DATA

bpemb = BPEmb(lang="en", dim=300)


# # Data preparation

# In[7]:
def extract_features(article):
    parts = article.split("\n")
    parts = [p.strip() for p in parts]
    unique, cs = np.unique(parts, return_counts=True)
    counts = {k: v for k, v in zip(unique, cs) if len(k) > 1}
    headlines = []
    headlines_tmp = []
    for p in parts:
        if counts.get(p, 0) > 1:
            headlines_tmp.append(p)
        else:
            if len(headlines_tmp) > len(headlines):
                headlines = headlines_tmp
            headlines_tmp = []
    headlines = [h for h in headlines if h not in ["Overview", "Important Information"]]
    return headlines


def get_phrase_vector(phrase):
    embs = bpemb.embed(phrase)
    embs = np.array([t / np.linalg.norm(t) for t in embs])
    emb_av = embs.mean(axis=0)
    emb_av = emb_av / np.linalg.norm(emb_av)
    return emb_av


def get_list_vector(l):
    embs = [get_phrase_vector(p) for p in l]
    embs = np.array([t / np.linalg.norm(t) for t in embs])
    emb_av = embs.mean(axis=0)
    emb_av = emb_av / np.linalg.norm(emb_av)
    return emb_av


def load_data():
    faq = pd.read_csv(f"{DATA}/faq.csv")
    faq["subsection"] = faq["article"]
    faq["article"] = faq["text"]
    faq = faq[["section", "subsection", "article", 'article_link']]

    faq["subsubsection"] = faq.article.str.split("\n").apply(lambda x: x[3])
    faq = faq[["section", "subsection", "subsubsection", "article", 'article_link']]

    faq["list_of_subsubsubsections"] = faq.article.apply(extract_features)
    print(f"{len(faq.loc[faq.list_of_subsubsubsections.str.len() < 1])} articles without headlines.")
    faq.loc[faq.list_of_subsubsubsections.str.len() < 1, "list_of_subsubsubsections"] = faq.loc[
        faq.list_of_subsubsubsections.str.len() < 1, "subsubsection"].apply(lambda x: [x])
    faq["subsubsection_embedding"] = faq["list_of_subsubsubsections"].apply(get_list_vector)

    faq_detailed = faq.copy().explode("list_of_subsubsubsections")
    faq_detailed.rename(
        {"list_of_subsubsubsections": "subsubsubsection"}, axis=1, inplace=True)
    faq_detailed["subsubsubsection_embedding"] = faq_detailed["subsubsubsection"].apply(get_phrase_vector)
    print('faq shape', faq.shape)
    print('faq_detailed shape', faq_detailed.shape)
    return faq, faq_detailed


try:
    faq, faq_detailed = load_data()
except:
    pass


def get_answer(request):
    request_embedding = get_phrase_vector(request)
    faq["score"] = faq["subsubsection_embedding"].apply(lambda x: sum(x * request_embedding))
    faq.sort_values("score", ascending=False, inplace=True)
    faq_sub = faq[["section", "subsection", "subsubsection", "list_of_subsubsubsections", "score", 'article_link']].head(3)
    # display(faq[["section", "subsection", "subsubsection", "list_of_subsubsubsections", "score"]].head(5))

    # print()
    # print("Result for subsubsubsection level: ")
    # faq_detailed["score"] = faq_detailed["subsubsubsection_embedding"].apply(lambda x: sum(x * request_embedding))
    # faq_detailed.sort_values("score", ascending=False, inplace=True)
    # faq_det_sub = faq_detailed[["section", "subsection", "subsubsection", "subsubsubsection", "score"]].head(3)
    return '\n'.join((faq_sub['section'] + ' / ' + faq_sub['subsection'] +
                      ' / ' + faq_sub['subsubsection'] + ' URL: ' + faq_sub['article_link']).values)


def main():
    questions = ['move task to another column',
                 'where is Gantt chart?',
                 'how to plot gann chart?',
                 'How to export annual report to Excel?',
                 'where is keyboard shortcuts?',
                 'how to set a user 2 step authorization?']

    for question in questions:
        print(':>', question)
        ans = get_answer(question)
        print('\t\t', ans)


if __name__ == '__main__':
    main()
