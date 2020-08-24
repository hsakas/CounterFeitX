"""
@author: aswamy
"""
from typing import List, Union, Any

import flair.embeddings
import numpy as np
import torch
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings, StackedEmbeddings
from torch.utils.data import Dataset


# pylint: disable=too-few-public-methods
class EmbedSentence:
    """
    EmbedSentence class helps in embeddings a sentence

    """

    def __init__(self):
        """
        initialize the word embedding and document embedding classes
        """
        self.word_embedding = flair.embeddings.WordEmbeddings('glove')
        self.doc_embedding = flair.embeddings.DocumentPoolEmbeddings([self.word_embedding])

        # embedding
        self.flair_forward = FlairEmbeddings('news-forward-fast')
        self.backward_flair = FlairEmbeddings('news-backward-fast')

        # stacked embedding
        self.stacked_embedding = StackedEmbeddings(embeddings=[
            self.flair_forward,
            self.backward_flair])

    def embed_str(self, sentence: str) -> torch.Tensor:
        """
        This function converts a sentence to a Tensor of embeddings
        :param sentence: str, for example: 'hello world'
        :return: returns a tensor, of shape already predefined by flair
        """
        __sentence = Sentence(sentence)
        self.doc_embedding.embed(__sentence)
        return __sentence.embedding

    def stacked_embed(self, sentence: str, return_sentence: bool = False) -> Union[torch.Tensor, Sentence]:
        """

        :param sentence:
        :param return_sentence:
        :return:
        """
        __sentence = Sentence(sentence)
        self.stacked_embedding.embed(__sentence)

        if return_sentence:
            return __sentence
        else:
            return __sentence.embedding


class SentenceDataset(Dataset):
    """
    Sentence Dataset will embed any incoming sentence into a tensor of default size

    """

    def __init__(self,
                 sentences: np.ndarray,
                 stacked: bool = False,
                 transform=None):
        """

        :param sentences: a numpy array, for example: np.array(['hello world', 'this is world news'])
        :param transform: any torch transform functions
        """
        self.sentences = sentences
        self.transform = transform
        self.stacked = stacked
        self.embedding = EmbedSentence()

    def __len__(self) -> int:
        """
        :return: the length of the sentences
        """
        return len(self.sentences)

    def __getitem__(self, idx) -> torch.Tensor:
        """
        :param idx: index of sentence
        :return: torch tensor for the specified idx
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.sentences[idx]

        if self.transform:
            sample = self.transform(sample)

        if self.stacked:
            return self.embedding.stacked_embed(sample, return_sentence=False)
        else:
            return self.embedding.embed_str(sample)


class SimpleDataset(Dataset):
    """
    Sentence Dataset will embed any incoming sentence into a tensor of default size

    """

    def __init__(self,
                 sentences: np.ndarray,
                 transform=None):
        """

        :param sentences: a numpy array, for example: np.array(['hello world', 'this is world news'])
        :param transform: any torch transform functions
        """
        self.sentences = sentences
        self.transform = transform

    def __len__(self) -> int:
        """
        :return: the length of the sentences
        """
        return len(self.sentences)

    def __getitem__(self, idx) -> Any:
        """
        :param idx: index of sentence
        :return: torch tensor for the specified idx
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.sentences[idx]

        if self.transform:
            sample = self.transform(sample)
        return sample


class SentencesDataset(Dataset):
    """
    Sentence Dataset will embed any incoming sentence into a tensor of default size

    """

    def __init__(self, sentences_list: List[np.ndarray]):
        """

        :param sentences_list: a numpy array, for example: [np.array(['hello world', 'this is world news']),
        np.array(['hello world', 'this is world news'])]
        """
        self.sentences_list = sentences_list
        self.embedding = EmbedSentence()

    def __len__(self) -> int:
        """
        :return: the length of the sentences
        """
        return len(self.sentences_list)

    def __getitem__(self, idx, stacked: bool = False) -> torch.Tensor:
        """
        :param idx: index of sentence
        :return: torch tensor for the specified idx
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        samples = self.sentences_list[idx]

        temp_sample = torch.zeros(100)

        for sample in samples:
            if stacked:
                sample = self.embedding.stacked_embed(sample)
            else:
                sample = self.embedding.embed_str(sample)
            temp_sample += sample

        return temp_sample
