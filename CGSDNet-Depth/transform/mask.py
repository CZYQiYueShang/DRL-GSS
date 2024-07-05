from typing import Any, Tuple

import torch
from torch import Tensor

import numpy as np
from PIL import Image


class MaskToTensor(object):
    def __call__(self,
                 mask: Any) -> Tensor:
        return torch.from_numpy(np.array(mask, dtype=np.int32)).long()


class BinaryNormalize(object):
    def __call__(self,
                 mask_data: Tensor) -> Tensor:
        mask_data = torch.where(mask_data >= 127.5, 1, 0).float()
        return mask_data


class MultiNormalize(object):
    def __call__(self,
                 mask_data: Tensor) -> Tensor:
        mask_data = torch.where(mask_data >= 1, 1, 0).float()
        return mask_data


class View(object):
    def __init__(self,
                 size: Tuple[int, ...]) -> None:
        self.size = size

    def __call__(self,
                 mask_data: Tensor) -> Tensor:
        return mask_data.view(self.size)


class Resize(object):
    def __init__(self,
                 size: int) -> None:
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

    def __call__(self,
                 mask: Any) -> Any:
        return mask.resize(self.size, Image.NEAREST)


class UnSqueeze(object):
    def __init__(self,
                 dim: int) -> None:
        self.dim = dim

    def __call__(self,
                 mask_data: Tensor) -> Tensor:
        return mask_data.unsqueeze(dim=self.dim)


class MultiStandardize(object):
    def __call__(self,
                 mask: Any) -> Any:
        mask_data = np.array(mask)
        mask_data[mask_data >= 1] = 255
        mask_image = Image.fromarray(mask_data, mode='L')
        return mask_image
