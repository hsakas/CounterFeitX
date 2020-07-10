"""
@author: aswamy

"""
# pylint: disable=invalid-name
import pandas as pd
from annoy import AnnoyIndex
from torch.utils.data import DataLoader
from tqdm import tqdm

from testsim.utils import EmbedSentence, SentenceDataset

# create a sentence embedder
embedder = EmbedSentence()

# read the csv file
df = pd.read_csv('raw_product.csv')
# create a dataloader
sentence_dataset = SentenceDataset(sentences=df['title'].to_numpy())
dataloader = DataLoader(dataset=sentence_dataset, batch_size=32, shuffle=False)


# create a annoy tree
t = AnnoyIndex(100, 'angular')

if __name__ == '__main__':
    for batch in tqdm(dataloader):
        print(batch.shape)
        for i, vec in enumerate(batch):
            t.add_item(i, vec)

    # build the tree
    t.build(100)

    print(t.get_nns_by_item(0, 10))
