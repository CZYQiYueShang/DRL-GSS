import random
from typing import Union, Tuple, List, Any

from PIL import Image


class Resize(object):
    def __init__(self,
                 size: Union[int, Tuple[int, int]]) -> None:
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

    def __call__(self,
                 image: Any,
                 mask: Any) -> Tuple[Any, Any]:
        assert image.size == mask.size
        w, h = image.size
        if w == h and image.size == self.size:
            return image, mask
        return image.resize(self.size, Image.BICUBIC), mask.resize(self.size, Image.BICUBIC)


class BatchResize(object):
    def __init__(self,
                 size: Union[int, Tuple[int, int]]) -> None:
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

    def __call__(self,
                 *images: Tuple[Any]) -> List[Any]:
        resized_images = [image.resize(self.size, Image.BICUBIC) for image in images]
        return resized_images


class RandomHorizontallyFlip(object):
    def __init__(self,
                 flip_pro: float = 0.5) -> None:
        self.flip_pro = flip_pro

    def __call__(self,
                 image: Any,
                 mask: Any) -> Tuple[Any, Any]:
        if random.random() < self.flip_pro:
            return image.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return image, mask


class BatchRandomHorizontallyFlip(object):
    def __init__(self,
                 flip_pro: float = 0.5) -> None:
        self.flip_pro = flip_pro

    def __call__(self,
                 *images: Tuple[Any]) -> Union[Tuple[Any], List[Any]]:
        if random.random() < self.flip_pro:
            flipped_images = [image.transpose(Image.FLIP_LEFT_RIGHT) for image in images]
            return flipped_images
        return images


class BatchRandomVerticallyFlip(object):
    def __init__(self,
                 flip_pro: float = 0.5) -> None:
        self.flip_pro = flip_pro

    def __call__(self,
                 *images: Tuple[Any]) -> Union[Tuple[Any], List[Any]]:
        if random.random() < self.flip_pro:
            flipped_images = [image.transpose(Image.FLIP_TOP_BOTTOM) for image in images]
            return flipped_images
        return images


class BatchRandomSelect(object):
    def __init__(self,
                 transform_fn_1: Any,
                 transform_fn_2: Any,
                 pro: float = 0.5) -> None:
        self.transform_fn_1 = transform_fn_1
        self.transform_fn_2 = transform_fn_2
        self.pro = pro

    def __call__(self,
                 *images: Tuple[Any]) -> Union[Tuple[Any], List[Any]]:
        if random.random() < self.pro:
            return self.transform_fn_1(*images)
        return self.transform_fn_2(*images)


class BatchCompose(object):
    def __init__(self,
                 transform_fns: List[Any]) -> None:
        self.transform_fns = transform_fns

    def __call__(self,
                 *images: Tuple[Any]) -> Union[Tuple[Any], List[Any]]:
        for transform_fn in self.transform_fns:
            images = transform_fn(*images)
        return images


class BatchRandomResizeCrop(object):
    def __init__(self,
                 size: Union[int, Tuple[int, int]],
                 scale: float = 0.6) -> None:
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        self.scale = scale

    def __call__(self, *images: Tuple[Any]) -> List[Any]:
        cropped_images: List[Any, ...] = []
        crop_params = self.get_random_crop_params(images[0].size, self.scale)
        for image in images:
            cropped_image = image.crop(crop_params)
            cropped_images.append(cropped_image)

        resized_images = [image.resize(self.size, Image.BICUBIC) for image in cropped_images]
        return resized_images

    @staticmethod
    def get_random_crop_params(ori_size: Tuple[int, int],
                               scale: float = 0.6) -> Tuple[int, int, int, int]:
        width, height = ori_size
        crop_scale = random.uniform(scale, 1.0)
        crop_width = int(width * crop_scale)
        crop_height = int(height * crop_scale)

        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)
        right = left + crop_width
        bottom = top + crop_height
        return left, top, right, bottom


class BatchRandomResize(object):
    def __init__(self,
                 sizes: List[int],
                 max_size: int = 1024) -> None:
        assert isinstance(sizes, list)
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self,
                 *images: Tuple[Any]) -> List[Any]:
        size = random.choice(self.sizes)
        resized_images = [resize_with_ratio(image, size, self.max_size) for image in images]
        return resized_images


def resize_with_ratio(image: Any,
                      size: int,
                      max_size: int) -> Any:
    width, height = image.size

    min_ori_size = float(min((width, height)))
    max_ori_size = float(max((width, height)))

    small_size = size
    big_size = int(round(max_ori_size / min_ori_size * small_size))
    if big_size > max_size:
        small_size = int(round(max_size * min_ori_size / max_ori_size))
        big_size = max_size

    if width < height:
        resized_width = small_size
        resized_height = big_size
    else:
        resized_width = big_size
        resized_height = small_size

    resized_image = image.resize((resized_width, resized_height), Image.BICUBIC)
    return resized_image
