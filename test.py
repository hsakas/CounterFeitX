'''
@author: aswamy
'''
import flair
import torch
from flair.data import Sentence

from flair.embeddings import WordEmbeddings, DocumentEmbeddings
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


class EmbedSentence:
    def __init__(self):
        self.embedder = WordEmbeddings('glove')
        self.document_embedding = DocumentEmbeddings(self.embedder)

    def transform(self, sentence: str):
        sentence = Sentence(sentence)
        return self.document_embedding.embed(sentence)


class SentenceDataset(Dataset):
    def __init__(self, root, filenames, transform=None):
        self.root = root
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        item = self.root + '/' + self.filenames[idx]

        if self.transform:
            item = self.transform(item)

        return item


dataset = SentenceDataset(root='./', filenames=[], transform=None)
dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)
sentence_embedder = EmbedSentence()

embeddings = torch.Tensor()

for batch in tqdm(dataloader):
    embeddings = torch.cat((embeddings, sentence_embedder.embedder(batch)), dim=0)
