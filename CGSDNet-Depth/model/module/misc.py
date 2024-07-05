import torch
from torch import nn, Tensor
from torch.nn import functional as F

from typing import Callable, Optional, Union, List, Tuple, Any

from model.module.CBAM import CBAM


def upsample_size(x: Tensor,
                  size: Any,
                  align_corners: bool = True) -> Tensor:
    return F.interpolate(x, size, mode="bilinear", align_corners=align_corners)


def upsample_scale(x: Tensor,
                   scale: Any,
                   align_corners: bool = True) -> Tensor:
    return F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=align_corners)


class ConvNormAct(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: Optional[int] = None,
                 group: int = 1,
                 dilation: int = 1,
                 bias: Optional[bool] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
                 act_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
                 inplace: bool = False) -> None:
        super(ConvNormAct, self).__init__()

        layers: List[nn.Module] = []

        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation,
                         groups=group, bias=bias)
        layers.append(conv)

        if norm_layer is not None:
            norm = norm_layer(out_channels)
            layers.append(norm)

        if act_layer is not None:
            if hasattr(act_layer, 'inplace'):
                act = act_layer(inplace=inplace)
            else:
                act = act_layer()
            layers.append(act)

        self.layers = nn.Sequential(*layers)

    def forward(self,
                x: Tensor) -> Tensor:
        x = self.layers(x)
        return x


class PPM(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 pool_dims: Tuple[int, ...] = (1, 2, 3, 6),
                 pool_layer: Optional[Callable[..., nn.Module]] = nn.AdaptiveAvgPool2d,
                 act_layer: Optional[Callable[..., nn.Module]] = nn.ReLU) -> None:
        super(PPM, self).__init__()

        inter_channels = in_channels // len(pool_dims)
        ppm_layers: List[nn.Module] = []

        for dim in pool_dims:
            ppm_layers.append(
                nn.Sequential(pool_layer(dim),
                              ConvNormAct(in_channels, inter_channels, 1, 1, 0, act_layer=act_layer))
            )

        self.ppm_layers = nn.Sequential(*ppm_layers)
        self.ppm_last_layer = ConvNormAct(in_channels * 2, out_channels, 1, 1, 0, act_layer=act_layer)

    def forward(self,
                x: Tensor) -> Tensor:
        size = x.shape[2:]
        ppm_outs: List[Tensor] = [x]
        for ppm_layer in self.ppm_layers:
            ppm_outs.append(upsample_size(ppm_layer(x), size))

        ppm_out = self.ppm_last_layer(torch.cat(ppm_outs, dim=1))
        return ppm_out


class CAP(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 atrous_rates: Tuple[int, ...] = (1, 2, 4, 8, 16),
                 act_layer: Optional[Callable[..., nn.Module]] = nn.ReLU) -> None:
        super(CAP, self).__init__()

        cap_layers: List[nn.Module] = []

        for atrous_rate in atrous_rates:
            cap_layers.append(
                nn.Sequential(ConvNormAct(in_channels, out_channels, 3, 1, 1, act_layer=act_layer),
                              ConvNormAct(out_channels, out_channels, 3, 1, padding=atrous_rate,
                                          dilation=atrous_rate, act_layer=act_layer),
                              ConvNormAct(out_channels, out_channels, 3, 1, 1, act_layer=act_layer))
            )

        cap_layers.append(
            nn.Sequential(nn.AdaptiveAvgPool2d(1),
                          ConvNormAct(in_channels, out_channels, 1, 1, 0, act_layer=act_layer))
        )

        self.cap_layers = nn.Sequential(*cap_layers)
        self.fuse = nn.Sequential(CBAM(len(self.cap_layers) * out_channels),
                                  ConvNormAct(len(self.cap_layers) * out_channels, out_channels, 1, 1, 0,
                                              act_layer=act_layer),
                                  nn.Dropout(0.5))
        self.out_channels = out_channels

    def forward(self,
                x: Tensor) -> Tensor:
        size = x.shape[2:]
        cap_outs: List[Tensor] = []
        cap_flow = 0

        for cap_layer in self.cap_layers[:-1]:
            cap_out = cap_layer[0](x)
            cap_out = cap_out + cap_flow
            cap_out = cap_layer[1:](cap_out)
            cap_flow = cap_out
            cap_outs.append(cap_out)
        cap_outs.append(upsample_size(self.cap_layers[-1](x), size))

        cap_out = self.fuse(torch.cat(cap_outs, dim=1))
        return cap_out


class HBD(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 act_layer: Optional[Callable[..., nn.Module]] = nn.ReLU) -> None:
        super(HBD, self).__init__()

        hbd_layers: List[nn.Module] = [ConvNormAct(in_channels, out_channels, 1, 1, 0, act_layer=act_layer)]
        self.hbd_layers = nn.Sequential(*hbd_layers)

    def forward(self,
                hbd_input: Tensor) -> Tensor:
        hbd_output = self.hbd_layers(hbd_input)
        return hbd_output


class Permute(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)


class DepthDecoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 act_layer: Optional[Callable[..., nn.Module]] = nn.ReLU) -> None:
        super(DepthDecoder, self).__init__()

        self.fuse = nn.Sequential(CBAM(gate_channels=in_channels),
                                  ConvNormAct(in_channels, out_channels, 1, 1, 0, act_layer=act_layer),
                                  ConvNormAct(out_channels, out_channels, 3, 1, 1, act_layer=act_layer))

        self.cnblock = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,
                                               groups=out_channels, bias=True),
                                     Permute([0, 2, 3, 1]),
                                     nn.LayerNorm(out_channels, eps=1e-6),
                                     nn.Linear(in_features=out_channels, out_features=4 * out_channels, bias=True),
                                     act_layer(),
                                     nn.Linear(in_features=4 * out_channels, out_features=out_channels, bias=True),
                                     Permute([0, 3, 1, 2]))

        self.upconv1 = ConvNormAct(out_channels, out_channels, 3, 1, 1, act_layer=act_layer)
        self.upconv2 = ConvNormAct(out_channels, out_channels // 2, 3, 1, 1, act_layer=act_layer)
        self.upconv3 = ConvNormAct(out_channels // 2, out_channels // 4, 3, 1, 1, act_layer=act_layer)

    def forward(self,
                x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.fuse(x)

        depth_out = self.cnblock(x)

        depth_out = self.upconv1(depth_out)
        depth_out = upsample_scale(depth_out, 2)
        depth_out = self.upconv2(depth_out)
        depth_out = upsample_scale(depth_out, 2)
        depth_out = self.upconv3(depth_out)
        return depth_out


class FPN(nn.Module):
    def __init__(self,
                 in_channels: Tuple[int, ...],
                 out_channels: int,
                 skip_top: bool = True,
                 context_layer: Optional[Callable[..., nn.Module]] = CAP,
                 boundary_layer: Optional[Callable[..., nn.Module]] = HBD,
                 depth_layer: Optional[Callable[..., nn.Module]] = None,
                 act_layer: Optional[Callable[..., nn.Module]] = nn.ReLU) -> None:
        super(FPN, self).__init__()

        self.skip_top = skip_top
        self.context_layer = context_layer
        self.boundary_layer = boundary_layer
        self.depth_layer = depth_layer
        self.ppm = PPM(in_channels=in_channels[-1], out_channels=out_channels, pool_dims=(1, 2, 3, 6),
                       act_layer=act_layer)

        fpn_in_layers: List[nn.Module] = []
        fpn_out_layers: List[nn.Module] = []
        if self.context_layer is not None:
            context_layers: List[nn.Module] = []
        if self.boundary_layer is not None:
            boundary_layers: List[nn.Module] = [self.boundary_layer(in_channels=out_channels, out_channels=1,
                                                                    act_layer=act_layer)]
        if self.depth_layer is not None:
            backbone_depth_layers: List[nn.Module] = [ConvNormAct(in_channels[0], out_channels // 4, 1, 1, 0,
                                                                  act_layer=nn.ELU)]
            context_depth_layers: List[nn.Module] = [ConvNormAct(out_channels, out_channels // 4, 1, 1, 0,
                                                                 act_layer=nn.ELU)]
            depth_layers: List[nn.Module] = [self.depth_layer(in_channels=out_channels // 2, out_channels=out_channels,
                                                              act_layer=nn.ELU)]

        for i in range(len(in_channels) - self.skip_top):
            fpn_in_layers.append(ConvNormAct(in_channels[i], out_channels, 1, 1, 0, act_layer=act_layer))
            fpn_out_layers.append(ConvNormAct(out_channels, out_channels, 3, 1, 1, act_layer=act_layer))
            if self.context_layer is not None:
                context_layers.append(self.context_layer(in_channels=out_channels, out_channels=out_channels,
                                                         act_layer=act_layer))
            if self.boundary_layer is not None:
                boundary_layers.append(self.boundary_layer(in_channels=out_channels, out_channels=1,
                                                           act_layer=act_layer))
            if self.depth_layer is not None:
                backbone_depth_layers.append(ConvNormAct(in_channels[i + 1], out_channels // 4, 1, 1, 0,
                                                         act_layer=nn.ELU))
                context_depth_layers.append(ConvNormAct(out_channels, out_channels // 4, 3, 1, 1, act_layer=nn.ELU))
                depth_layers.append(self.depth_layer(in_channels=out_channels * 3 // 4, out_channels=out_channels,
                                                     act_layer=nn.ELU))

        self.fpn_in_layers = nn.Sequential(*fpn_in_layers)
        self.fpn_out_layers = nn.Sequential(*fpn_out_layers)
        if self.context_layer is not None:
            self.context_layers = nn.Sequential(*context_layers)
        if self.boundary_layer is not None:
            self.boundary_layers = nn.Sequential(*boundary_layers)
        if self.depth_layer is not None:
            self.backbone_depth_layers = nn.Sequential(*backbone_depth_layers)
            self.context_depth_layers = nn.Sequential(*context_depth_layers)
            self.depth_layers = nn.Sequential(*depth_layers)

    def forward(self,
                fpn_input: List[Tensor]) -> Union[Tuple[List[Tensor], List[Tensor], List[Tensor]],
                                                  Tuple[List[Tensor], List[Tensor]], List[Tensor]]:
        fpn_out: List[Tensor] = []

        backbone_f = fpn_input[-1]
        f = self.ppm(backbone_f)
        fpn_out.insert(0, f)

        if self.boundary_layer is not None:
            boundary_outs: List[Tensor] = []
            boundary_out = self.boundary_layers[-1](f)
            boundary_outs.insert(0, boundary_out)

        if self.depth_layer is not None:
            depth_outs: List[Tensor] = []
            backbone_depth_f = self.backbone_depth_layers[-1](backbone_f)
            context_depth_f = self.context_depth_layers[-1](f)
            depth_out = self.depth_layers[0](torch.cat([backbone_depth_f, context_depth_f], dim=1))
            depth_outs.insert(0, depth_out)

        # Cascaded Network Architecture
        for i in range(len(self.fpn_in_layers)):
            backbone_f = fpn_input[-(2 + i)]
            x = self.fpn_in_layers[-(1 + i)](backbone_f)
            f = upsample_size(f, x.shape[2:])
            f = x + f

            if self.boundary_layer is not None:
                boundary_out = upsample_size(boundary_out, x.shape[2:])
                f = f + torch.sigmoid(boundary_out) * f

            if self.context_layer is not None:
                f = self.context_layers[-(1 + i)](f)
                if self.boundary_layer is not None:
                    boundary_out = self.boundary_layers[-(2 + i)](f)
                    boundary_outs.insert(0, boundary_out)
            fpn_out.insert(0, self.fpn_out_layers[-(1 + i)](f))

            if self.depth_layer is not None:
                backbone_depth_f = self.backbone_depth_layers[-(2 + i)](backbone_f)
                context_depth_f = self.context_depth_layers[-(2 + i)](f)
                depth_out = self.depth_layers[i + 1](torch.cat([backbone_depth_f, context_depth_f,
                                                                upsample_scale(depth_out, 0.5)], dim=1))
                depth_outs.insert(0, depth_out)

        if self.boundary_layer is not None and self.depth_layer is not None:
            return fpn_out, boundary_outs, depth_outs
        elif self.boundary_layer is not None:
            return fpn_out, boundary_outs
        elif self.depth_layer is not None:
            return fpn_out, depth_outs
        else:
            return fpn_out


if __name__ == "__main__":
    testModule = CAP(in_channels=1, out_channels=1)
    x = torch.randn(3, 1, 13, 13)
    print(testModule(x).size())
    print(testModule.cap_layers[0][1:](x).size())
