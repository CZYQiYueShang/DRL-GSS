from typing import Any

from torchvision.transforms import *


class NullTransform(object):
    def __call__(self,
                 image: Any) -> Any:
        return image
