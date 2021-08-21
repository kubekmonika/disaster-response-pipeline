import sys
import joblib
import re
import json
import sqlite3
import plotly
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
import plotly.graph_objs as gobj

from collections import defaultdict


app = Flask(__name__)


def get_pos_refs():
    """
    Pos tags reference.

    Sources: 
    https://stackoverflow.com/posts/57686805/revisions
    https://www.nltk.org/_modules/nltk/tag/mapping.html
    """
    pos_refs = defaultdict(lambda: 'n')  # noun is set as default
    pos_refs['VERB'] = 'v'  # verb
    pos_refs['ADJ'] = 'a'  # adjective
    pos_refs['ADV'] = 'r'  # adverb

    return pos_refs


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

    # get pos tags for the tokens
    tokens_pos = nltk.pos_tag(tokens, tagset='universal')

    # lemmatize
    tokens = [lemmatizer.lemmatize(word, pos=pos_refs[pos]) for word, pos in tokens_pos]

    return tokens


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # create graphs
    graphs = []

    # 1
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    genre_counts = genre_counts.tolist()

    data1 = gobj.Bar(
        x=genre_names,
        y=genre_counts
    )
    layout1 = {
        'title': 'Distribution of Message Genres',
        'yaxis': {'title': "Count"},
        'xaxis': {'title': "Genre"}
    }

    graphs.append(dict(data=data1, layout=layout1))

    # 2
    df_categories = df.drop(['message', 'original', 'genre'], axis=1).sum().sort_values(ascending=True)
    categories_data = df_categories.tolist()
    categories_names = list(df_categories.index.str.replace('_', ' '))

    data2 = gobj.Bar(
        y=categories_names,
        x=categories_data,
        orientation="h"
    )
    layout2 = {
        'title': 'Distribution of Target Categories',
        'xaxis': {'title': "Count"},
        'yaxis': {'title': "Categories", 'automargin': True}
    }
    graphs.append(dict(data=data2, layout=layout2))

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    database_filepath, model_filepath = sys.argv[1:]

    global df
    global model
    global pos_refs

    # load data
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql('SELECT * FROM data', conn)

    # load model
    model = joblib.load(model_filepath)

    # get pos references
    pos_refs = get_pos_refs()

    # run the app
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()