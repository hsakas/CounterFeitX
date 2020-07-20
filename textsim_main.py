"""
@author: aswamy

"""
# pylint: disable=invalid-name
import pandas as pd
from annoy import AnnoyIndex
from torch.utils.data import DataLoader
from tqdm import tqdm

from textsim.utils import EmbedSentence, SentenceDataset

# create a sentence embedder
embedder = EmbedSentence()

# read the csv file
df = pd.read_csv('raw_product.csv')
# create a dataloader
sentence_dataset = SentenceDataset(sentences=df['title'].to_numpy())
dataloader = DataLoader(dataset=sentence_dataset, batch_size=10, shuffle=False)


# create a annoy tree
t = AnnoyIndex(100, 'angular')

if __name__ == '__main__':
    counter = 0
    for batch in tqdm(dataloader):
        for _, vec in enumerate(batch):
            t.add_item(counter, vec)
            counter += 1

    # build the tree
    t.build(100)

    # query using a index
    print(t.get_nns_by_item(0, 10))

    # query using a vector
    # print(t.get_nns_by_vector(..., 10))