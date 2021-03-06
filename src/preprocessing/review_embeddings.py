from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")


def get_review_embedding(review):
    inputs = tokenizer(review, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state
