import torch
from torch import nn, Tensor

from model.backbone.ResNet_D import resnet101
from model.backbone.ConvNeXt_P import convnext_base


class Backbone(nn.Module):
    def __init__(self,
                 backbone_path: str,
                 device: torch.device = 'cpu') -> None:
        super(Backbone, self).__init__()
        if 'resnet101_deep' in backbone_path:
            net = resnet101()
        elif 'convnext_base' in backbone_path:
            net = convnext_base(num_classes=1000)
        else:
            raise ValueError('No such Backbone model for path: %s!' % backbone_path)

        if 'pth' in backbone_path:
            pretrained_dict = torch.load(backbone_path, map_location=device)
            net_dict = net.state_dict()
            # print([k for k in net_dict.keys() if (k not in pretrained_dict)])
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict and 'Prediction' not in k)}
            net_dict.update(pretrained_dict)
            net.load_state_dict(net_dict)
            print("Successfully loaded pretrained weights of Backbone!")

        if 'resnet' in backbone_path:
            self.layer0 = nn.Sequential(net.conv1, net.bn1, net.relu)
            self.layer1 = nn.Sequential(net.maxpool, net.layer1)
            self.layer2 = net.layer2
            self.layer3 = net.layer3
            self.layer4 = net.layer4
            print("Successfully got ResNet Backbone model!")
        elif 'convnext' in backbone_path:
            self.layer0 = nn.Identity()
            self.layer1 = nn.Sequential(net.features[0], net.features[1])
            self.layer2 = nn.Sequential(net.features[2], net.features[3])
            self.layer3 = nn.Sequential(net.features[4], net.features[5])
            self.layer4 = nn.Sequential(net.features[6], net.features[7])
            print("Successfully got ConvNeXt Backbone model!")

    def forward(self,
                x: Tensor) -> Tensor:
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer4


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    backbone_path = '../../ckpt/pretrained/resnet101_deep.pth'
    net = Backbone(backbone_path=backbone_path, device=device)
