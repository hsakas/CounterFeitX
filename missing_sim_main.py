"""
@author: aswamy
"""
# pylint: disable=invalid-name
import pandas as pd
from annoy import AnnoyIndex
from torch.utils.data import DataLoader
from tqdm import tqdm

from textsim.utils import EmbedSentence, SentencesDataset

# create a sentence embedder
embedder = EmbedSentence()

# read the csv file
df = pd.read_csv('missing_data.csv')
# create a dataloader
dataset = SentencesDataset(sentences_list=df.filter(['title', 'description']).to_numpy())

dataloader = DataLoader(dataset=dataset, batch_size=10, shuffle=False)

# create a annoy tree
tree = AnnoyIndex(100, 'angular')

if __name__ == '__main__':
    counter_k = 0
    for batch in tqdm(dataloader):
        for _, vec in enumerate(batch):
            tree.add_item(counter_k, vec)
            counter_k += 1

    # build the tree
    tree.build(100)

    # query using a index
    print(tree.get_nns_by_item(0, 10))

    for i in tree.get_nns_by_item(164, 2):
        print(f'id -> {i}')
        print('value -> ' + df.filter(['title', 'description']).iloc[i])
        print('brand suggestion -> ', df.filter(['brand']).iloc[i])
