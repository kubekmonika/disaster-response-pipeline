import sys
import sqlite3
import pandas as pd
import numpy as np
import re
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from collections import defaultdict
from joblib import parallel_backend

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')


def load_data(database_filepath):
    """
    Load the data and split it into messages and categories.
    """
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql('SELECT * FROM data', conn)

    X = df['message']
    Y = df.drop(['message', 'original', 'genre'], axis=1)
    category_names = Y.columns.tolist()

    return X, Y, category_names


def tokenize(text):
    """
    Clear and tokenize text data.
    """
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "")

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # remove numbers
    text = re.sub(r"\d+", " ", text.lower())

    # tokenize
    tokens = [word for word in word_tokenize(text) if word not in stop_words]

    # POS TAGS REFERENCE
    # sources: 
    # https://stackoverflow.com/posts/57686805/revisions
    # https://www.nltk.org/_modules/nltk/tag/mapping.html
    pos_refs = defaultdict(lambda: 'n')  # noun is set as default
    pos_refs['VERB'] = 'v'  # verb
    pos_refs['ADJ'] = 'a'  # adjective
    pos_refs['ADV'] = 'r'  # adverb

    # get pos tags for the tokens
    tokens_pos = nltk.pos_tag(tokens, tagset='universal')

    # lemmatize
    tokens = [lemmatizer.lemmatize(word, pos=pos_refs[pos]) for word, pos in tokens_pos]

    return tokens


def build_model(x_train, Y_train):
    """
    Build the ML pipeline and a model.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(
            RidgeClassifier())
        )
    ])

    parameters = {
        'vect__max_df': [.25, .2, .15, .1, .05],
        'vect__min_df': [20, 10, 5, 4],
        'tfidf__norm': ['l1', 'l2'],
        'tfidf__smooth_idf': [True],
        'clf__estimator__alpha': [.01, .1, .2, .3, .4],
        'clf__estimator__normalize': [True],
        'clf__estimator__solver': ['auto'],
        'clf__estimator__class_weight': ['balanced'],
    }
    cv = GridSearchCV(pipeline, parameters, cv=5)

    print('Searching for best params...')
    with parallel_backend('loky', n_jobs=5):
        cv.fit(x_train, Y_train)

    best_params = {key: [val] for key, val in cv.best_params_.items()}
    print(cv.best_params_)

    print('Training model...')
    with parallel_backend('loky', n_jobs=5):
        model = GridSearchCV(pipeline, best_params, cv=10)
        model.fit(x_train, Y_train)

    return model


def evaluate_model(model, x_test, Y_test, category_names):
    """
    Score the model and show a classification report for each target category.
    """
    print('Score:', model.score(x_test, Y_test))
    print()

    Y_pred = model.predict(x_test)

    for i in range(Y_pred.shape[1]):
        report = classification_report(Y_test.values[:,i], Y_pred[:,i], zero_division=0)
        print(f'====={category_names[i]}=====')
        print(report)


def save_model(model, model_filepath):
    """
    Save the model as a pickle file.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()