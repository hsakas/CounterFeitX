"""
@author: aswamy
"""
# pylint disable=invalid-name

from time import time

import torch.nn as nn
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings
from flair.embeddings import StackedEmbeddings
from flair.embeddings import WordEmbeddings

sentence = "nike adidadas"
sentence = Sentence(sentence)
word_embedding = WordEmbeddings('glove')

flair_forward = FlairEmbeddings('news-forward-fast')

stacked_embeddings = StackedEmbeddings(embeddings=[
    flair_forward,
])

keywords = 'adidas ' * 1012
keywords = Sentence(keywords)

stacked_embeddings.embed(sentence)
stacked_embeddings.embed(keywords)

cos = nn.CosineSimilarity(dim=1, eps=1e-6)

if __name__ == "__main__":
    start = time()
    # pylint disable=line-too-long
    [[cos(token.embedding.view(-1, 1024), token2.embedding.view(-1, 1024)) for token in keywords] for token2 in
     sentence]
    end = time()
    print(end - start)
