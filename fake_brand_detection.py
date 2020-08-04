"""
@author: aswamy
"""

# pylint: disable=invalid-name
from time import time

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from searcher.finder import FakeDetector
from textsim.utils import EmbedSentence
from textsim.utils import SimpleDataset

# actual brand names
keywords = 'nike adidas puma rebook ucla hrx'

embedder = EmbedSentence()

# read the csv file
df = pd.read_csv('raw_product2.csv')

# create a dataloader
sentence_dataset = SimpleDataset(sentences=df['title'].to_numpy())
dataloader = DataLoader(dataset=sentence_dataset, batch_size=30, shuffle=False)

if __name__ == "__main__":
    start = time()
    detector = FakeDetector()
    detector.build(brand_names=keywords)

    for batch in tqdm(dataloader):
        for _, vec in enumerate(batch):
            detector.fake_detector(text=vec, detection_range=(0.7, 0.8))
    end = time()
    print(end - start)
