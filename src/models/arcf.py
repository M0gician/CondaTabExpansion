import argparse
import os
import numpy as np
import math
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch


# rating discriminator for user inputs dimensions:
# rating vector: (total_number_of_items, 1)
# c_u : (total_number_of_users, 1)
# review_embedding: (final_embedding_size, 1)

class UserEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(UserEmbedding, self).__init__()
        self.input_layer = nn.Embedding(input_dim, 1024)
        self.first_layer = nn.Linear(1024, 2048)
        self.output_layer = nn.Linear(2048, output_dim)

    def forward(self, user_one_hot_vector):
        user_embedding = self.input_layer(user_one_hot_vector)
        user_embedding = self.first_layer(user_embedding)
        user_embedding = nn.functional.relu(user_embedding)
        return self.output_layer(user_embedding)


class ReviewEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, reviwes):
        # define pre trained BERT
        pass


class Discriminator(nn.Module):
    def __init__(self, input_size, c_embedding_size, review_embedding_size):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(input_size + c_embedding_size + review_embedding_size, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 128),
            nn.ReLU(True),
            nn.Linear(128, 16),
            nn.ReLU(True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, rating_vector, c_vector, user_reviews):
        c_embedding = UserEmbedding(c_vector)
        review_embedding = ReviewEmbedding(user_reviews)
        data_c = torch.cat((rating_vector, c_embedding, review_embedding), 1)
        result = self.dis(data_c)
        return result


class Generator(nn.Module):
    def __init__(self, input_size, c_embedding_size, review_embedding_size):
        self.input_size = input_size
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(self.input_size + c_embedding_size + review_embedding_size, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.input_size),
            nn.Sigmoid()
        )

    def forward(self, noise_vector, c_vector, user_reviews):
        c_embedding = UserEmbedding(c_vector)
        review_embedding = ReviewEmbedding(user_reviews)
        G_input = torch.cat([noise_vector, c_embedding, review_embedding], 1)
        result = self.gen(G_input)
        return result


class ItemEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ItemEmbedding, self).__init__()
        self.input_layer = nn.Embedding(input_dim, 1024)
        self.first_layer = nn.Linear(1024, 2048)
        self.output_layer = nn.Linear(2048, output_dim)

    def forward(self, item_one_hot_vector):
        item_embedding = self.input_layer(item_one_hot_vector)
        item_embedding = self.first_layer(item_embedding)
        item_embedding = nn.functional.relu(item_embedding)
        return self.output_layer(item_embedding)
