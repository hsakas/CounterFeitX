"""
@author: aswamy

"""
# pylint: disable=invalid-name
import pandas as pd
from annoy import AnnoyIndex
from torch.utils.data import DataLoader
from tqdm import tqdm

from textsim.utils import EmbedSentence, SentenceDataset, SentencesDataset

# create a sentence embedder
embedder = EmbedSentence()

# read the csv file
df = pd.read_csv('missing_data.csv')
# create a dataloader
sentence_dataset = SentenceDataset(sentences=df['title'].to_numpy())
desc_dataset = SentenceDataset(sentences=df['description'].to_numpy())
dataset = SentencesDataset(sentences_list=df.filter(['title', 'description']).to_numpy())
# anything in the column

sentence_dataloader = DataLoader(dataset=sentence_dataset, batch_size=10, shuffle=False)
desc_dataloader = DataLoader(dataset=desc_dataset, batch_size=10, shuffle=False)
dataloader = DataLoader(dataset=dataset, batch_size=10, shuffle=False)


# create a annoy tree
title_tree = AnnoyIndex(100, 'angular')
desc_tree = AnnoyIndex(100, 'angular')
tree = AnnoyIndex(100, 'angular')


if __name__ == '__main__':
    counter_i = 0
    for batch in tqdm(sentence_dataloader):
        for _, vec in enumerate(batch):
            title_tree.add_item(counter_i, vec)
            counter_i += 1

    counter_j = 0
    for batch in tqdm(desc_dataloader):
        for _, vec in enumerate(batch):
            desc_tree.add_item(counter_j, vec)
            counter_j += 1

    counter_k = 0
    for batch in tqdm(dataloader):
        for _, vec in enumerate(batch):
            tree.add_item(counter_k, vec)
            counter_k += 1

    # build the tree
    title_tree.build(100)
    desc_tree.build(100)
    tree.build(100)

    # query using a index
    print(title_tree.get_nns_by_item(0, 10))
    print(desc_tree.get_nns_by_item(0, 10))
    print(tree.get_nns_by_item(0, 10))

    for i in tree.get_nns_by_item(164, 2):
        print(f'id -> {i}')
        print('value -> ' + df.filter(['title', 'description']).iloc[i])
        print('brand suggestion -> ', df.filter(['brand']).iloc[i])