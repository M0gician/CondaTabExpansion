from transformers import BertTokenizer, BertModel
from nltk import tokenize as nltk_tokenize
import numpy as np
import torch

import nltk
nltk.download('punkt')
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()
useGPU = False
if useGPU:
    model.to('cuda')

f = open("./src/data/review_embeddings.txt", "w")

def tokenize(first_sentence, second_sentence):
    tokenized_first = ['[CLS]'] + tokenizer.tokenize(first_sentence) + ['[SEP]']
    tokenized_second = tokenizer.tokenize(second_sentence)+ ['[SEP]'] if  len(second_sentence) > 0 else []
    tokenized_text = tokenized_first + tokenized_second
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0 for i in range(len(tokenized_first))]
    segments_ids += [1 for i in range(len(tokenized_second))]
    if useGPU:
        print(tokenized_text)
        print(segments_ids)
    return tokenized_text, segments_ids, indexed_tokens

def get_sentence_embedding(first_sentence, second_sentence):
    tokenized_review, segments_ids, indexed_tokens = tokenize(first_sentence, second_sentence)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    if useGPU:
        tokens_tensor = tokens_tensor.to('cuda')
        segments_tensors = segments_tensors.to('cuda')
    with torch.no_grad():
        outputs = model(tokens_tensor, token_type_ids=segments_tensors)
        embedding = outputs[0] #[0]
    assert tuple(embedding.shape) == (1, len(indexed_tokens), model.config.hidden_size)
    return embedding

def get_review_embeddings(review):
    embeddings = []
    review_lines = nltk_tokenize.sent_tokenize(review)
    first_sentence, last_sentence = review_lines[0], review_lines[-1]
    # first sentence alone
    embedding = get_sentence_embedding(first_sentence, '')
    embeddings.append(embedding)
    # all pairs of sentences
    for line in review_lines[1:]:
        second_sentence = line
        embedding = get_sentence_embedding(first_sentence, second_sentence)        
        embeddings.append(embedding)
        first_sentence = line
    # last sentence alone
    embedding = get_sentence_embedding(last_sentence, '')        
    embeddings.append(embedding)
    return embeddings

example_review = 'I love this product. \
        I mean honestly, who doesn\'t love chocolate? \
        Only sociopaths, I reckon. \
        I\'d eat this every day if I could. '

reviews = [example_review]
        
for review in reviews:
    embeddings = get_review_embeddings(review)
    # result = torch.stack(embeddings, dim=0).sum(dim=0).sum(dim=0)
    # print(result.shape)
    # # f.write(embedding)

f.close()