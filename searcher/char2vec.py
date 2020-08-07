"""
@author: aswamy
@github: hsakas
"""
from time import time
from typing import List, Set

import torch
import torch.nn as nn
from flair.data import Sentence
from flair.embeddings import WordEmbeddings
from torch import Tensor

# word embedder
word_embedder = WordEmbeddings('glove')


# word splitter to char
def split_word(word: str) -> Set[str]:
    """

    :param word: a string, must be a single word
    :return: set of char separated into chars
    """
    return set([i for i in word])


# embed single char into a vec
def char2vec(chars: List[str]) -> Tensor:
    """

    :param chars: list of string of characters
    :return: tensor of shape defined by word embedding algorithm
    """
    # since we are getting a list, we need to convert it into a sentence
    keywords = Sentence(' '.join(chars))
    word_embedder.embed(keywords)
    return torch.mean(torch.stack([token.embedding for token in keywords]), dim=0)


cos = nn.CosineSimilarity(dim=1, eps=1e-6)

if __name__ == '__main__':
    start = time()
    x = char2vec(split_word('different'))
    y = char2vec(split_word('similar'))
    print(cos(x.view(-1, 100), y.view(-1, 100)))
    end2 = time()
    print(f'Time Taken -> {end2 - start}')
