from transformers import BertTokenizer, BertModel
from nltk import tokenize as nltk_tokenize
import numpy as np
import torch
import nltk

# Downloads
nltk.download("punkt")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Settings
useGPU = False
sentence_embedding_type = "CLS"  # "avg"
review_embedding_type = "avg"  # other?

model.eval()
if useGPU:
    model.to("cuda")

f = open("./src/data/review_embeddings.txt", "w")


def process(first_sentence, second_sentence):
    """Pre-process sentence(s) for BERT. Returns:
        - tokenized text (with [CLS] and [SEP] tokens)
        - segment sentence ids ([0s & 1s])
        - indexed tokens """
    tokenized_first = ["[CLS]"] + tokenizer.tokenize(first_sentence) + ["[SEP]"]
    if second_sentence:
        tokenized_second = tokenizer.tokenize(second_sentence) + ["[SEP]"]
    else:
        tokenized_second = []
    tokenized_text = tokenized_first + tokenized_second
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0 for i in range(len(tokenized_first))]
    segments_ids += [1 for i in range(len(tokenized_second))]
    return tokenized_text, segments_ids, indexed_tokens


def get_sentence_embedding(sentence_pair):
    """returns sentence (pair) embedding of size [1, 768]"""
    first_sentence, second_sentence = sentence_pair
    tokenized_review, segments_ids, indexed_tokens = process(
        first_sentence, second_sentence
    )
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    if useGPU:
        tokens_tensor = tokens_tensor.to("cuda")
        segments_tensors = segments_tensors.to("cuda")
    with torch.no_grad():
        output = model(tokens_tensor, token_type_ids=segments_tensors)

    # get embedding using specified method
    if sentence_embedding_type == "CLS":
        return output.pooler_output
    elif sentence_embedding_type == "avg":
        return output.last_hidden_state.mean(axis=1)
    else:
        return None


def create_sentence_pairs(review):
    """ returns array of sentence pair tuples"""
    pairs = []
    review_sentences = nltk_tokenize.sent_tokenize(review)
    first_sentence, last_sentence = review_sentences[0], review_sentences[-1]
    # first sentence alone
    pairs.append((first_sentence, ""))
    # all pairs of sentences
    for line in review_sentences[1:]:
        second_sentence = line
        pairs.append((first_sentence, second_sentence))
        first_sentence = line
    # last sentence alone
    pairs.append((last_sentence, ""))
    return pairs


def get_review_embedding(review):
    """returns review embedding of size [1, 768]"""
    sentence_pairs = create_sentence_pairs(review)
    sentence_embeddings = map(get_sentence_embedding, sentence_pairs)  # [pairs, 1, 768]
    if review_embedding_type == "avg":
        # avg over all pairs [pairs, 1, 768] => [1, 768]
        mean = torch.mean(torch.stack(list(sentence_embeddings)), axis=0)


# def get_review_embedding(review):
#     """returns review embedding of size [1, 768]"""
#     embeddings = []
#     review_sentences = nltk_tokenize.sent_tokenize(review)
#     first_sentence, last_sentence = review_sentences[0], review_sentences[-1]

#     # first sentence alone
#     embedding = get_sentence_embedding(first_sentence, "")
#     embeddings.append(embedding)

#     # all pairs of sentences
#     for line in review_sentences[1:]:
#         second_sentence = line
#         embedding = get_sentence_embedding(first_sentence, second_sentence)
#         embeddings.append(embedding)
#         first_sentence = line

#     # last sentence alone
#     embedding = get_sentence_embedding(last_sentence, "")
#     embeddings.append(embedding)
#     np.array(embeddings)
#     if review_embedding_type == "avg":
#         print(embeddings.size())
#         print(embeddings.mean(axis=0).size())
#         print(embeddings.mean(axis=1).size())
#         return embeddings.mean()
#     else:
#         return None


example_review = "I love this product. \
        I mean honestly, who doesn't love chocolate? \
        Only sociopaths, I reckon. \
        I'd eat this every day if I could. "

reviews = [example_review]
for review in reviews:
    review_embedding = get_review_embedding(review)
    # f.write(review_embedding)

f.close()
