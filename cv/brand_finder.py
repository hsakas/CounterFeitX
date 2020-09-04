"""
@author: aswamy
@github: hsakas
"""

import os
from typing import Tuple, List, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from annoy import AnnoyIndex
from torch import Tensor

from cv.utils import slices2batch, open_resize


class BrandDetector:
    """
    #TODO: complete the desc.
    """

    def __init__(self, brands: Tensor, brand_names: List[str], n_tree: int = 100):
        """
        Logo tree and model is defined here
        """
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.model = models.resnet50(pretrained=True)

        assert len(brands) == len(brand_names), f'Mismatched brand tensor and brand names. \
        Expected {len(brands)} brand names for tensors but found {len(brand_names)}'

        self.brand_names = brand_names
        self.brand_vecs = self.model(brands.float())

        self.logo_tree = self.build(n_tree)

    @staticmethod
    def get_brand_tensors(path: str = './cv/brands') -> Tensor:
        """

        :param path: direct to the path where are the brand logos are saved
        :return: returns a batch of tensor of all the logos converted into tensors
        """
        return torch.tensor([open_resize(path + i) for i in os.listdir(path)])

    def idx_to_brand(self, idx: int):
        return self.brand_names[idx]

    def build(self, n_tree: int = 100):
        """

        :param n_tree: number of tree to be created in annoy, default=100
        :return:
        """
        embedding_size = self.brand_vecs.size(-1)
        logo_tree = AnnoyIndex(embedding_size, 'angular')

        for value, _token in enumerate(self.brand_vecs):
            logo_tree.add_item(value, _token)

        logo_tree.build(n_tree)
        return logo_tree

    def brand_detect(self,
                     slices: Tensor,
                     detection_threshold: float = 0.55) -> Union[Tuple[int, str, float], Any]:
        """
        :param slices:
        :param detection_threshold:
        :return:
        """

        # convert slices to a batch
        batch = slices2batch(slices)
        output = self.model(batch.float())
        match = [self.logo_tree.get_nns_by_vector(output[i], 1)[0] for i in range(len(output))]
        matches_score = np.array([self.cos(output[i], self.brand_vecs[match[i]]) for i in range(len(output))])
        matched_brand_index = int(np.argmax(matches_score))

        if matches_score[matched_brand_index] >= detection_threshold:
            return matched_brand_index, self.idx_to_brand(match[matched_brand_index]), matches_score[
                matched_brand_index]

        return None, None, None
