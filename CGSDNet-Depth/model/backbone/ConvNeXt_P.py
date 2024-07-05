from functools import partial
from typing import Any, Callable, List, Optional, Sequence

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torchvision.ops.misc import ConvNormActivation
from torchvision.ops.stochastic_depth import StochasticDepth  # Drop path


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class Permute(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)


class CNBlock(nn.Module):
    def __init__(self,
                 dim,
                 layer_scale: float,
                 stochastic_depth_prob: float,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.block = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
                                   Permute([0, 2, 3, 1]),
                                   norm_layer(dim),
                                   nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
                                   nn.GELU(),
                                   nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
                                   Permute([0, 3, 1, 2]))
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)  # 与输出size相同的缩放tensor
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")  # Drop path

    def forward(self, input: Tensor) -> Tensor:
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)  # Drop path
        result += input
        return result


class CNBlockConfig:
    # Stores information listed at Section 3 of the ConvNeXt paper
    def __init__(
        self,
        input_channels: int,
        out_channels: Optional[int],
        num_layers: int,
    ) -> None:
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "input_channels={input_channels}"
        s += ", out_channels={out_channels}"
        s += ", num_layers={num_layers}"
        s += ")"
        return s.format(**self.__dict__)


class ConvNeXt(nn.Module):
    def __init__(self,
                 block_setting: List[CNBlockConfig],
                 stochastic_depth_prob: float = 0.0,
                 layer_scale: float = 1e-6,
                 num_classes: int = 1000,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 **kwargs: Any) -> None:
        super(ConvNeXt, self).__init__()

        self.stochastic_depth_prob = stochastic_depth_prob

        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (isinstance(block_setting, Sequence) and all([isinstance(s, CNBlockConfig) for s in block_setting])):
            raise TypeError("The block_setting should be List[CNBlockConfig]")

        if block is None:
            block = CNBlock

        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)

        layers: List[nn.Module] = []

        # Stem
        firstconv_output_channels = block_setting[0].input_channels
        layers.append(ConvNormActivation(3,
                                         firstconv_output_channels,
                                         kernel_size=4,
                                         stride=4,
                                         padding=0,
                                         norm_layer=norm_layer,
                                         activation_layer=None,
                                         bias=True))
        # layers.append(nn.Sequential(nn.Conv2d(3, firstconv_output_channels, kernel_size=4, padding=0, bias=True),
        #                             norm_layer(firstconv_output_channels)))

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            # Bottlenecks
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(cnf.input_channels, layer_scale, sd_prob))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if cnf.out_channels is not None:
                # Downsampling
                layers.append(nn.Sequential(norm_layer(cnf.input_channels),
                                            nn.Conv2d(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2)))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        lastblock = block_setting[-1]
        lastconv_output_channels = (lastblock.out_channels if lastblock.out_channels is not None
                                    else lastblock.input_channels)
        self.classifier = nn.Sequential(norm_layer(lastconv_output_channels),
                                        nn.Flatten(1),
                                        nn.Linear(lastconv_output_channels, num_classes))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _convnext(arch: str,
              block_setting: List[CNBlockConfig],
              stochastic_depth_prob: float,
              num_classes: int = 1000,
              **kwargs: Any) -> ConvNeXt:
    model = ConvNeXt(block_setting, stochastic_depth_prob=stochastic_depth_prob, num_classes=num_classes, **kwargs)
    return model


def convnext_tiny(*, num_classes=1000, **kwargs: Any) -> ConvNeXt:
    # https://download.pytorch.org/models/convnext_tiny-983f1562.pth
    block_setting = [CNBlockConfig(96, 192, 3),
                     CNBlockConfig(192, 384, 3),
                     CNBlockConfig(384, 768, 9),
                     CNBlockConfig(768, None, 3)]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
    return _convnext("convnext_tiny", block_setting, stochastic_depth_prob, num_classes, **kwargs)


def convnext_small(*, num_classes=1000, **kwargs: Any) -> ConvNeXt:
    # https://download.pytorch.org/models/convnext_small-0c510722.pth
    block_setting = [CNBlockConfig(96, 192, 3),
                     CNBlockConfig(192, 384, 3),
                     CNBlockConfig(384, 768, 27),
                     CNBlockConfig(768, None, 3)]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.4)
    return _convnext("convnext_small", block_setting, stochastic_depth_prob, num_classes, **kwargs)


def convnext_base(*, num_classes=1000, **kwargs: Any) -> ConvNeXt:
    # https://download.pytorch.org/models/convnext_base-6075fbad.pth
    block_setting = [CNBlockConfig(128, 256, 3),
                     CNBlockConfig(256, 512, 3),
                     CNBlockConfig(512, 1024, 27),
                     CNBlockConfig(1024, None, 3)]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return _convnext("convnext_base", block_setting, stochastic_depth_prob, num_classes, **kwargs)


def convnext_large(*, num_classes=1000, **kwargs: Any) -> ConvNeXt:
    # https://download.pytorch.org/models/convnext_large-ea097f82.pth
    block_setting = [CNBlockConfig(192, 384, 3),
                     CNBlockConfig(384, 768, 3),
                     CNBlockConfig(768, 1536, 27),
                     CNBlockConfig(1536, None, 3)]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return _convnext("convnext_large", block_setting, stochastic_depth_prob, num_classes, **kwargs)


if __name__ == "__main__":
    ConvNeXt_B = convnext_base(num_classes=1000)
    x = torch.randn(1, 3, 224, 224)
    torch.onnx.export(ConvNeXt_B, x, 'ConvNeXt_B.onnx', verbose=True)
