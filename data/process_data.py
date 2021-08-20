import sys
import sqlite3

import pandas as pd


def load_data(messages_filepath, categories_filepath):
    """
    Load data and return it as a data frame.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    return pd.merge(messages, categories, on='id')


def clean_data(df):
    """
    Clean the data.
    """
    categories = df['categories'].str.split(';', expand=True)
    categories.columns = categories.iloc[0].str.split('-').str.get(0).tolist()

    for col in categories.columns:
        categories[col] = categories[col].str.split('-').str.get(1).astype(int)

    # remove rows with wrong labels
    mask_rows_proper_values = (categories.isin([0, 1])).all(axis=1)
    categories = categories[mask_rows_proper_values]

    categories.drop('child_alone', axis=1, inplace=True)
    categories.drop('related', axis=1, inplace=True)

    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], join='inner', axis=1)

    return df.drop_duplicates(subset=['message']).drop('id', axis=1)


def save_data(df, database_filename):
    """
    Save the data frame to the database
    """
    conn = sqlite3.connect(database_filename)
    df.to_sql('data', conn, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()