from typing import Any, Tuple

import torch
from torch import Tensor

import cv2 as cv
import numpy as np
from PIL import Image


class DepthToTensor(object):
    def __call__(self,
                 depth: Any) -> Tensor:
        return torch.from_numpy(np.array(depth, dtype=np.int32)).long()


class Normalize(object):
    def __call__(self,
                 depth_data: Tensor) -> Tensor:
        depth_data = depth_data.float() / 1000.0
        return depth_data


class View(object):
    def __init__(self,
                 size: Tuple[int, ...]) -> None:
        self.size = size

    def __call__(self,
                 depth_data: Tensor) -> Tensor:
        return depth_data.view(self.size)


class UnSqueeze(object):
    def __init__(self,
                 dim: int) -> None:
        self.dim = dim

    def __call__(self,
                 depth_data: Tensor) -> Tensor:
        return depth_data.unsqueeze(dim=self.dim)


class Resize(object):
    def __init__(self,
                 size: int) -> None:
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

    def __call__(self,
                 depth: Any) -> Any:
        return depth.resize(self.size, Image.NEAREST)


def visualize_depth(depth: Any) -> Any:
    depth_mat = depth
    depth_color = cv.applyColorMap(cv.convertScaleAbs(depth_mat, alpha=0.03), cv.COLORMAP_JET)
    return depth_color
