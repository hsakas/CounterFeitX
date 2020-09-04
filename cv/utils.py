"""
@author: aswamy
@github: hsakas
"""
from typing import Union

from PIL import Image
import numpy as np
import torch
from skimage.color import rgb2gray
from torch import Tensor

from cv.GLOBALS import STD_IMG_TUPLE_MULTI, STD_IMG_TUPLE_SINGLE, STD_IMG_SIZE


# open image and resize
def open_resize(img_path: str, size: int = 512) -> np.ndarray:
    """

    :param img_path: path to the specified image
    :param size: size to be resized into
    :return: numpy array
    """
    img = Image.open(img_path)
    img = img.resize((size, size))
    img = rgb2gray(np.asarray(img))
    return img


# slicer algorithm
def slicer(inp: Tensor, window: int, stride: int, device='cpu') -> torch.Tensor:
    """
    This function will return slices of a image tensor

    :param inp: input image tensor
    :param window: window size to be sliced
    :param stride: distance from the previous slice
    :param device: 'cpu' for cpu enabled processing, select 'gpu' for gpu enabled processing of the image tensor
    :return: sliced tensor of shape [B, C, W, H], B -> batch, C -> channel, W -> width, H -> height
    """
    inp = inp.to(device)
    slices = inp.unfold(0, window, stride)
    return slices.reshape(-1, window, window).permute(0, 2, 1)


def single2multi(inp: Tensor, get_np: bool = False) -> Union[Tensor, np.ndarray]:
    """
    Convert a single channel tensor to multi-channel vector

    :param inp: a tensor of single channel
    :param get_np: specify "True" to get numpy arrays, "False" for pytorch tensor
    :return: a tensor or numpy array of multi-channel
    """
    if get_np:
        return inp.unsqueeze_(0).repeat(3, 1, 1).view(STD_IMG_TUPLE_MULTI).cpu().numpy()
    else:
        return inp.unsqueeze_(0).repeat(3, 1, 1).view(STD_IMG_TUPLE_MULTI)


def slices2batch(slices: Tensor) -> Tensor:
    """
    :param slices: slices of a single image converted into a tensor
    :return: batch of slices
    """

    def __expand(x: Tensor) -> Tensor: return x.view(STD_IMG_TUPLE_SINGLE).expand(-1, -1, 3)

    return torch.tensor(torch.stack([__expand(slices[i]) for i in range(slices.size(0))], dim=0)).\
        view(4, 3, STD_IMG_SIZE, STD_IMG_SIZE)
