"""
@author: aswamy
@github: hsakas
"""
from typing import List, Dict

import torch
from flair.data import Sentence
from flair.embeddings import WordEmbeddings
from torch import Tensor

# type annotations
Word = List[str]

# word embedder
word_embedder = WordEmbeddings('glove')


# word splitter to char
def split_word(word: str) -> List[str]:
    """

    :param word: a string, must be a single word
    :return: set of char separated into chars
    """
    return [i for i in word]


# embed single char into a vec
def char2vec(chars: Word) -> Tensor:
    """

    :param chars: list of string of characters
    :return: tensor of shape defined by word embedding algorithm
    """
    # since we are getting a list, we need to convert it into a sentence
    keywords = Sentence(' '.join(chars))
    if not keywords:
        raise ValueError(f'Passed Empty Keyword')
    word_embedder.embed(keywords)
    return torch.mean(torch.stack([token.embedding for token in keywords]), dim=0)


def sentence_char2vec(sentence: str) -> Dict[str, Tensor]:
    """

    :param sentence:
    :return:
    """
    # split the sentence into a list
    _sentence = sentence.split(' ')
    return {word: char2vec(split_word(word)) for word in _sentence}
