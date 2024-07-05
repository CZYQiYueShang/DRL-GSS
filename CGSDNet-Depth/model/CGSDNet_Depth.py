import torch
from torch import nn, Tensor

from typing import Callable, Optional, List

from model.backbone.Backbone import Backbone
from model.module.misc import upsample_size, ConvNormAct, CAP, HBD, DepthDecoder, FPN


class CGSDNetDepth(nn.Module):
    def __init__(self,
                 backbone_path: str,
                 skip_top: bool = True,
                 context_layer: Optional[Callable[..., nn.Module]] = CAP,
                 boundary_layer: Optional[Callable[..., nn.Module]] = HBD,
                 depth_layer: Optional[Callable[..., nn.Module]] = DepthDecoder,
                 act_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
                 max_depth: float = 10.0,
                 device: torch.device = 'cpu') -> None:
        super(CGSDNetDepth, self).__init__()
        # params
        self.skip_top = skip_top
        self.boundary_layer = boundary_layer
        self.max_depth = max_depth

        if 'resnet' in backbone_path or 'resnext' in backbone_path:
            basic_dim = 256
        elif 'convnext' in backbone_path:
            basic_dim = 128
        else:
            raise ValueError('No such Backbone model for CGSDNet!')

        # backbone
        net = Backbone(backbone_path, device=device)
        self.layer0 = net.layer0
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        # PPM and FPN
        self.fpn = FPN(in_channels=(basic_dim, 2 * basic_dim, 4 * basic_dim, 8 * basic_dim), out_channels=basic_dim,
                       skip_top=self.skip_top, context_layer=context_layer, boundary_layer=boundary_layer,
                       depth_layer=depth_layer, act_layer=act_layer)

        self.fuse = ConvNormAct(in_channels=(5 - self.skip_top) * basic_dim, out_channels=basic_dim, kernel_size=1,
                                stride=1, padding=0, act_layer=act_layer)
        self.segment = nn.Sequential(ConvNormAct(in_channels=basic_dim, out_channels=basic_dim, kernel_size=1, stride=1,
                                                 padding=0, act_layer=act_layer),
                                     nn.Conv2d(basic_dim, 1, 1, 1, 0, bias=True))

        self.predict_layers = nn.ModuleList([nn.Conv2d(basic_dim, 1, 3, 1, 1),
                                             nn.Conv2d(basic_dim, 1, 3, 1, 1),
                                             nn.Conv2d(basic_dim, 1, 3, 1, 1),
                                             nn.Conv2d(basic_dim, 1, 3, 1, 1)])
        if self.skip_top is not True:
            self.predict_layers.append(nn.Conv2d(basic_dim, 1, 3, 1, 1))

        self.final_predict = nn.Conv2d(1, 1, 3, 1, 1)

        if self.boundary_layer is not None:
            self.boundary_fusion = ConvNormAct(5 - self.skip_top, 1, 1, 1, 0, act_layer=act_layer)
            self.boundary_final_predict = nn.Conv2d(1, 1, 3, 1, 1)

        self.depth_fuse = nn.Sequential(ConvNormAct(in_channels=(5 - self.skip_top) * basic_dim // 4,
                                                    out_channels=basic_dim, kernel_size=1, stride=1, padding=0,
                                                    act_layer=nn.ELU))
        self.depth_predict_layers = nn.ModuleList([nn.Conv2d(basic_dim // 4, 1, 3, 1, 1),
                                                   nn.Conv2d(basic_dim // 4, 1, 3, 1, 1),
                                                   nn.Conv2d(basic_dim // 4, 1, 3, 1, 1),
                                                   nn.Conv2d(basic_dim // 4, 1, 3, 1, 1)])
        if self.skip_top is not True:
            self.depth_predict_layers.append(nn.Conv2d(basic_dim // 4, 1, 3, 1, 1))
        self.depth_final_predict = nn.Sequential(ConvNormAct(in_channels=basic_dim, out_channels=basic_dim,
                                                             kernel_size=1, stride=1, padding=0, act_layer=nn.ELU),
                                                 nn.Conv2d(basic_dim, 1, 1, 1, 0, bias=True),
                                                 nn.Conv2d(1, 1, 3, 1, 1))

    def forward(self,
                x: Tensor) -> List[Tensor]:
        size = x.shape[2:]

        layer0_feature = self.layer0(x)
        layer1_feature = self.layer1(layer0_feature)
        layer2_feature = self.layer2(layer1_feature)
        layer3_feature = self.layer3(layer2_feature)
        layer4_feature = self.layer4(layer3_feature)

        if self.skip_top is True:
            fpn_in_features = [layer1_feature, layer2_feature, layer3_feature, layer4_feature]
        else:
            fpn_in_features = [layer1_feature, layer2_feature, layer3_feature, layer4_feature, layer4_feature]

        if self.boundary_layer is None:
            fpn_out_features, depth_out_features = self.fpn(fpn_in_features)
        else:
            fpn_out_features, boundary_out_features, depth_out_features = self.fpn(fpn_in_features)
            boundary_fuse_features: List[Tensor] = [boundary_out_features[0]]

        fuse_size = fpn_out_features[0].shape[2:]
        fuse_features: List[Tensor] = [fpn_out_features[0]]
        predicts: List[Tensor] = [torch.sigmoid(upsample_size(self.predict_layers[0](fpn_out_features[0]), size))]

        depth_fuse_size = depth_out_features[0].shape[2:]
        depth_fuse_features: List[Tensor] = [depth_out_features[0]]
        depth_predicts: List[Tensor] = [torch.sigmoid(upsample_size(self.depth_predict_layers[0](depth_out_features[0]),
                                                                    size)) * self.max_depth]

        for i in range(1, len(fpn_out_features)):
            fpn_out_feature = upsample_size(fpn_out_features[i], fuse_size)
            fuse_features.append(fpn_out_feature)
            predict = self.predict_layers[i](fpn_out_features[i])
            predict = upsample_size(predict, size)
            predicts.append(torch.sigmoid(predict))
            if self.boundary_layer is not None:
                boundary_fuse_features.append(upsample_size(boundary_out_features[i], fuse_size))
            depth_fuse_features.append(upsample_size(depth_out_features[i], depth_fuse_size))
            depth_predicts.append(torch.sigmoid(upsample_size(self.depth_predict_layers[i](depth_out_features[i]),
                                                              size)) * self.max_depth)

        fuse = self.fuse(torch.cat(fuse_features, dim=1))
        if self.boundary_layer is not None:
            boundary_fuse = self.boundary_fusion(torch.cat(boundary_fuse_features, dim=1))
            boundary_final_predict = self.boundary_final_predict(boundary_fuse)
            boundary_final_predict = upsample_size(boundary_final_predict, size)
            boundary_predict = torch.sigmoid(boundary_final_predict)
            predicts.append(boundary_predict)

            boundary_final_fuse = torch.sigmoid(boundary_fuse) * fuse
            fuse = fuse + boundary_final_fuse

        depth_fuse = self.depth_fuse(torch.cat(depth_fuse_features, dim=1))
        depth_final_predict = self.depth_final_predict(depth_fuse)
        depth_final_predict = upsample_size(depth_final_predict, size)
        depth_final_predict = torch.sigmoid(depth_final_predict) * self.max_depth

        depth_predicts.append(depth_final_predict)
        predicts.extend(depth_predicts)

        segment = self.segment(fuse)
        final_predict = self.final_predict(segment)
        final_predict = upsample_size(final_predict, size)

        predicts.append(torch.sigmoid(final_predict))
        return predicts


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    backbone_path = '../ckpt/pretrained/convnext_base'
    net = CGSDNetDepth(backbone_path=backbone_path, skip_top=True, context_layer=CAP, boundary_layer=HBD,
                       act_layer=nn.ReLU, max_depth=10.0, device=device)

    num_params = sum(p.numel() for p in net.parameters())
    print(num_params)
