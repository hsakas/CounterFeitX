"""
@author: aswamy
@github: hsakas
"""

from time import time

import torch
import torch.nn as nn
from annoy import AnnoyIndex
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings
from flair.embeddings import StackedEmbeddings
from flair.embeddings import WordEmbeddings

# pylint: disable=invalid-name
sentence = "nikee adidas AIR FORCE 1 Board Shoes Men's Shoes Women's Shoes 2020 New High-top Deconstruction White Shoes Casual Shoes Sports Shoes Board Shoes".lower()

sentence = Sentence(sentence)
word_embedding = WordEmbeddings('glove')

# embedding
flair_forward = FlairEmbeddings('news-forward-fast')
backward_flair = FlairEmbeddings('news-backward-fast')

# stacked embedding
stacked_embeddings = StackedEmbeddings(embeddings=[
    # flair_forward,
    # backward_flair
    word_embedding
])

# global
EMBEDDING_SIZE = 100

# brand names
keywords = 'nike adidas puma rebook'
keywords = Sentence(keywords)

# embed both sentences
stacked_embeddings.embed(sentence)
stacked_embeddings.embed(keywords)

# set cosine sim function
cos = nn.CosineSimilarity(dim=1, eps=1e-6)

EMBEDDING_SIZE = int([token.embedding.shape for token in sentence][0][0])
# build a tree for all brands
t = AnnoyIndex(EMBEDDING_SIZE, 'angular')

for i, token in enumerate(keywords):
    t.add_item(i, token.embedding)

t.build(100)

if __name__ == "__main__":
    start = time()
    found_match = False

    for i, token in enumerate(sentence):
        match = t.get_nns_by_vector(token.embedding, 1)[0]
        sim_score = float(cos(token.embedding.view(-1, EMBEDDING_SIZE),
                              torch.tensor(t.get_item_vector(match)).view(-1, EMBEDDING_SIZE)))
        print(sim_score, match, token)
        if 0.6 <= sim_score <= 0.8:
            print(f'Found Counterfeit')
            print(f"Found Counterfeit {token} with a {round(sim_score, 2)} similarity to brand {keywords[match]}")
            found_match = True

        if found_match:
            break

    end = time()
    print(end - start)
