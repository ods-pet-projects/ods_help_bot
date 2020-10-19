#!/usr/bin/env python
# coding: utf-8

# In[19]:
import json
import random
from collections import defaultdict

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from config import DATA
REPLIES_DICT = json.load(open(f'{DATA}/dialog_data.json'))

total_stats = {'intent': 0, 'generative': 0, 'failure': 0}
DIALOGS_DATA_PATH = f'{DATA}/chats.txt'
# PROBA_THRESHOLD = 0.09
PROBA_THRESHOLD = 0.7


def fit_intent_models():
    corpus = []
    y = []
    for intent, intent_data in REPLIES_DICT['intents'].items():
        for example in intent_data['examples']:
            corpus.append(example)
            y.append(intent)

    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
    X = vectorizer.fit_transform(corpus)

    clf_proba = LogisticRegression()
    clf_proba.fit(X, y)

    clf = LinearSVC()
    clf.fit(X, y)
    return clf, clf_proba, vectorizer


def get_intent(clf, clf_proba, vectorizer, question):
    best_intent = clf.predict(vectorizer.transform([question]))[0]

    index_of_best_intent = list(clf_proba.classes_).index(best_intent)
    probabilities = clf_proba.predict_proba(vectorizer.transform([question]))[0]

    best_intent_proba = probabilities[index_of_best_intent]
    print('\t\t', question, best_intent, best_intent_proba)
    if best_intent_proba > PROBA_THRESHOLD:
        return best_intent


def get_answer_by_intent(intent):
    phrases = REPLIES_DICT['intents'][intent]['responses']
    return random.choice(phrases)


def clear_question(question):
    question = question.lower().strip()
    alphabet = ' -1234567890йцукенгшщзхъфывапролджэёячсмитьбю'
    question = ''.join(c for c in question if c in alphabet)
    return question


def prepare_data():
    with open(DIALOGS_DATA_PATH) as f:
        content = f.read()

    dialogues = content.split('\n\n')

    questions = set()
    dataset = defaultdict(list)

    for dialogue in dialogues:
        replicas = dialogue.split('\n')[:2]
        if len(replicas) == 2:
            question, answer = replicas
            question = clear_question(question[2:])
            answer = answer[2:]

            if question and question not in questions:
                questions.add(question)
                words = question.split(' ')
                for word in words:
                    dataset[word].append([question, answer])

    too_popular = set()
    for word in dataset:
        if len(dataset[word]) > 10000:
            too_popular.add(word)

    for word in too_popular:
        dataset.pop(word)
    return dataset


def get_generative_answer(dataset, replica):
    replica = clear_question(replica)
    words = replica.split(' ')

    mini_dataset = []
    for word in words:
        if word in dataset:
            mini_dataset += dataset[word]

    candidates = []

    for question, answer in mini_dataset:
        if abs(len(question) - len(replica)) / len(question) < 0.4:
            d = nltk.edit_distance(question, replica)
            diff = d / len(question)
            if diff < 0.4:
                candidates.append([question, answer, diff])
    if candidates:
        winner = min(candidates, key=lambda candidate: candidate[2])
        return winner[1]


def get_failure_phrase():
    phrases = REPLIES_DICT['failure_phrases']
    return random.choice(phrases)


dataset = prepare_data()
clf, clf_proba, vectorizer = fit_intent_models()


def get_full_small_talk_answer(question):
    intent = get_intent(clf, clf_proba, vectorizer, question)
    if intent:
        total_stats['intent'] += 1
        return get_answer_by_intent(intent)
    answer = get_generative_answer(dataset, question)

    if answer:
        total_stats['generative'] += 1
        return answer

    total_stats['failure'] += 1
    return get_failure_phrase()


def check_bot():
    test_queries = ['Что приготовить?',
                    'как дела?',
                    'Привет! как дела?',
                    'Доброй ночи!!',
                    'Как тебя зовут?',
                    'как стать kaggle грандмастером?'
                    ]

    for q in test_queries:
        # print(q, get_full_small_talk_answer(q))
        print(q, get_answer(q))


def get_answer(question):
    intent = get_intent(clf, clf_proba, vectorizer, question)
    if intent:
        return get_answer_by_intent(intent)


def main():
    check_bot()


if __name__ == '__main__':
    main()
