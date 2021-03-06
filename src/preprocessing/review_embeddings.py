from transformers import BertTokenizer, BertModel
from nltk import tokenize as nltk_tokenize
import torch

import nltk
nltk.download('punkt')
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

f = open("./src/data/review_embeddings.txt", "w")

def tokenize(first_sentence, second_sentence):
    tokenized_first = ['[CLS]'] + tokenizer.tokenize(first_sentence) + ['[SEP]']
    tokenized_second = tokenizer.tokenize(second_sentence)+ ['[SEP]'] if  len(second_sentence) > 0 else []
    tokenized_text = tokenized_first + tokenized_second
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0 for i in range(len(tokenized_first))]
    segments_ids += [1 for i in range(len(tokenized_second))]
    print(tokenized_text)
    print(segments_ids)
    return tokenized_text, segments_ids, indexed_tokens


def get_review_embedding(review):
    review_lines = nltk_tokenize.sent_tokenize(review)
    first, last = review_lines[0], review_lines[-1]
    prev_line = first
    # first sentence alone
    tokenized_review, segments_ids, indexed_tokens = tokenize(first, '')
    # all pairs of sentences
    for line in review_lines[1:]:
        first_sentence = prev_line
        second_sentence = line
        tokenized_review, segments_ids, indexed_tokens = tokenize(first_sentence, second_sentence)
        prev_line = line
    # last sentence alone
    tokenized_review, segments_ids, indexed_tokens = tokenize(last, '')

example_review = 'I love this product. \
        I mean honestly, who doesn\'t love chocolate? \
        Only sociopaths, I reckon. \
        I\'d eat this every day if I could. '
        

get_review_embedding(example_review)

# f.write(tokenized_review)
f.close()