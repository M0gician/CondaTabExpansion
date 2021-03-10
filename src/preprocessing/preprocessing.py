from collections.abc import Iterable

import pandas as pd
import numpy as np
import gzip
import json
import os

import nltk
from collections import Counter


DATASET_DIR = "/mnt/nfs/scratch1/neerajsharma/amazon_data/"
META_PREFIX = "meta_"
REVIEW_PREFIX = "review_"
DATASET_NAME = {
    'phone': 'Cell_Phones_and_Accessories.json.gz',
    'beauty': 'All_Beauty.json.gz',                     # haven't downloaded yet
    'food': 'Grocery_and_Gourmet_Food.json.gz'
}
nltk.download('punkt')


def parse(path: str):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)

def getDF(path: str, num_entry='all') -> pd.DataFrame:
    df = {}
    if num_entry == 'all':
        for i, d in enumerate(parse(path)):
            df[i] = d
    elif isinstance(num_entry, int):
        for i, d in enumerate(parse(path)):
            if i > num_entry - 1:
                break
            df[i] = d
    else:
        raise TypeError("'num_entry' can either be an int or 'all'")

    return pd.DataFrame.from_dict(df, orient='index')

def getAmazonData(data_name: str, num_entry='all') -> pd.DataFrame:
    try:
        path = os.path.join(DATASET_DIR, DATASET_NAME[data_name])
        return getDF(path, num_entry)
    except KeyError:
        print(f"Dataset '{data_name}' is not supported!")

def save_review_dataset(data_name: str, agg_func=list, save_path="./", min_usr_review=5) -> None:
    df = getAmazonData(data_name)
    df = df[df['reviewerID'].map(df['reviewerID'].value_counts()) >= min_usr_review]
    agg_df = df.groupby('reviewerID').agg(
        summary=pd.NamedAgg(column='summary', aggfunc=agg_func),
        reviewText=pd.NamedAgg(column='reviewText', aggfunc=agg_func)
    )
    agg_df.to_json(
        path_or_buf=save_path+REVIEW_PREFIX+f"{data_name}.json.gz",
        orient="index",
        compression='gzip'
    )

def save_meta_dataset(data_name: str, agg_func=lambda x: x, save_path="./") -> None:
    df = getAmazonData(data_name)
    agg_df = df.groupby('asin').agg(
        title = pd.NamedAgg(column='title', aggfunc=agg_func),
        category = pd.NamedAgg(column='category', aggfunc=agg_func),
        brand = pd.NamedAgg(column='brand', aggfunc=agg_func),
        also_buy = pd.NamedAgg(column='also_buy', aggfunc=agg_func),
        also_view = pd.NamedAgg(column='also_view', aggfunc=agg_func),
        price = pd.NamedAgg(column='price', aggfunc=agg_func)
    )
    agg_df.to_json(
        path_or_buf=save_path+META_PREFIX+f"{data_name}.json.gz",
        orient="index",
        compression='gzip'
    )

def describe_dataset(data_name: str, dataset=None, is_meta=False) -> None:
    dataset = getAmazonData(data_name) if dataset is None else dataset
    columns = list(dataset.columns)

    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("input 'dataset' is not a dataframe")
    print(f"Dataset name: {DATASET_NAME['food']}")
    print(f"Dataset size: {len(dataset)}")
    print("Dataset columns:")
    for col in columns:
        if dataset[col].dtypes in [float, int]:
            print(f"\t'{col}'--Unique: {dataset[col].nunique()}")
            print(" "*(len(col)+2) + f"\t--max: {dataset[col].max()}, mean: {dataset[col].mean()}, min: {dataset[col].min()}")
        else:
            try:
                print(f"\t'{col}'--Unique: {dataset[col].nunique()}")
            except:
                print(f"\t'{col}'--Total: {dataset[col].count()}")

def top_phrases(dataset: pd.DataFrame, column: str, phrase_length: int or Iterable, print_top50=False) -> pd.Series:
    if dataset[column].dtypes != object:
        raise TypeError(f"The targeted column '{column}' doesn't have the correct dtype 'object'!")
    if isinstance(phrase_length, int):
        n = [phrase_length]
    elif isinstance(phrase_length, Iterable):
        n = phrase_length
    else:
        raise TypeError(f"'phrase_length' must be an integer or an iterable of integers!")
    data_split = [y for x in dataset[column] for y in str(x).split()]
    phrase_counter = pd.Series(
        [' '.join(y) for x in n for y in nltk.ngrams(data_split, x)]
    ).value_counts()
    
    if print_top50:
        with pd.option_context('display.max_rows', 50, 'display.max_columns', 2):
            if len(phrase_counter) >= 50:
                print(phrase_counter.iloc[:50])
            else:
                print(phrase_counter)
    return phrase_counter

def sentence_count(dataset: pd.DataFrame, column: str, print_top50=False) -> pd.Series:
    if dataset[column].dtypes != object:
        raise TypeError(f"The targeted column '{column}' doesn't have the correct dtype 'object'!")

    tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
    sentence_counts = dataset[column].apply(lambda x: len(tokenizer.tokenize(str(x)))).value_counts()

    if print_top50:
        with pd.option_context('display.max_rows', 50, 'display.max_columns', 3):
            if len(sentence_counts) >= 50:
                top_50 = sentence_counts.iloc[:50]
                df = pd.DataFrame({'Sentence Length':top_50.index, 'Counts':top_50.values})
                print(df)
            else:
                df = pd.DataFrame({'Sentence Length':sentence_counts.index, 'Counts':sentence_counts.values})
                print(df)
    return sentence_counts


if __name__ == '__main__':
    save_review_dataset('phone')
    # df = getDF('./review_phone.json.gz', num_entry=100)
