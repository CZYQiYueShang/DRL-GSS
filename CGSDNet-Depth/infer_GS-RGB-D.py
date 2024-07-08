import os
import sys
import argparse
from typing import List, Tuple, Any

import cv2 as cv
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable

from model.CGSDNet import CGSDNet
from model.CGSDNet_Depth import CGSDNetDepth
from transform.depth import visualize_depth


parser = argparse.ArgumentParser(description='Inference for GS RGB-D')
# read images
parser.add_argument('--root_path', type=str, default='../ORB-SLAM2-GSD/GS RGB-D/corridor_chair_day')
parser.add_argument('--image_scale', type=int, default=416)
# network
parser.add_argument('--network', type=str, default='CGSDNet-Depth')
parser.add_argument('--backbone', type=str, default='convnext_base')
parser.add_argument('--trained_model', type=str, default='./ckpt/trained/CGSDNet-Depth/ConvNeXt_all-200.pth')
parser.add_argument('--skip_top', action='store_true', default=True)
parser.add_argument('--max_depth', type=float, default=10.0)
parser.add_argument('--device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

parser.add_argument('--save_boundary', action='store_true', default=False)
parser.add_argument('--save_depth', action='store_true', default=True)

args = parser.parse_args()
args.model_name = args.trained_model.split('/')[-1][:-4]
print(args)


def read(root_path: str) -> Tuple[List[str], List[Any]]:
    path = os.path.join(root_path, 'rgb')
    if os.path.exists(path):
        items = os.listdir(path)
        items.sort()
        read_bar = tqdm(items, file=sys.stdout)

        images: List[Any] = []
        for step, item in enumerate(read_bar):
            image = cv.imread(os.path.join(path, item))
            images.append(image)
            read_bar.desc = "reading images"
        print('Complete images reading!')
        return items, images
    else:
        raise ValueError("The path does not exist!")


def get_model(network: str,
              backbone: str = 'resnext',
              skip_top: bool = True,
              max_depth: float = 10.0,
              trained_model: str = None,
              device: str = 'cpu') -> nn.Module:
    if network == 'CGSDNet':
        model = CGSDNet(backbone_path=backbone, skip_top=skip_top, device=device)
    elif network == 'CGSDNet-Depth':
        model = CGSDNetDepth(backbone_path=backbone, skip_top=skip_top, max_depth=max_depth, device=device)
    else:
        raise ValueError('No such network model!')

    if trained_model is not None:
        print("Load %s's trained model %s for inference" % (network, trained_model))
        trained_dict = torch.load(trained_model, map_location=device)
        net_dict = model.state_dict()
        trained_dict = {k: v for k, v in trained_dict.items() if (k in net_dict and 'Prediction' not in k)}
        net_dict.update(trained_dict)
        model.load_state_dict(net_dict)
        print("Successfully loaded trained model %s!" % trained_model)
    else:
        print("No such trained model %s!" % trained_model)

    model.to(device)
    print("Successfully got network model of %s!" % network)
    return model


def folder_detect(network: str,
                  image_names: List[str],
                  images: List[Any],
                  model: nn.Module,
                  image_transform: transforms.Compose,
                  root_path: str,
                  save_boundary: bool = True,
                  save_depth: bool = True,
                  device: str = 'cpu'):
    model.eval()
    detect_bar = tqdm(images, file=sys.stdout)
    save_path = root_path

    if not os.path.exists(os.path.join(save_path, 'mask')):
        os.makedirs(os.path.join(save_path, 'mask'))
    if save_boundary:
        if not os.path.exists(os.path.join(save_path, 'boundary')):
            os.makedirs(os.path.join(save_path, 'boundary'))
    if save_depth:
        if not os.path.exists(os.path.join(save_path, 'depth_GS')):
            os.makedirs(os.path.join(save_path, 'depth_GS'))
        if not os.path.exists(os.path.join(save_path, 'depth_GS_vis')):
            os.makedirs(os.path.join(save_path, 'depth_GS_vis'))

    for step, input in enumerate(detect_bar):
        height, width, channel = input.shape

        with torch.no_grad():
            input_RGB = cv.cvtColor(input, cv.COLOR_BGR2RGB)
            image = Image.fromarray(input_RGB)
            image_var = Variable(image_transform(image).unsqueeze(0)).to(device)

            if network == 'CGSDNet':
                predicts = model(image_var)
                output_mask = predicts[-1]
                output_mask = output_mask.data.squeeze(0).cpu()
                output_mask = np.array(transforms.Resize((height, width), interpolation=Image.BICUBIC)
                                       (transforms.ToPILImage()(output_mask)))
                Image.fromarray(output_mask).save(os.path.join(save_path, 'mask', image_names[step]))
                if save_boundary:
                    output_boundary = predicts[-2]
                    output_boundary = output_boundary.data.squeeze(0).cpu()
                    output_boundary = np.array(transforms.Resize((height, width), interpolation=Image.BICUBIC)
                                               (transforms.ToPILImage()(output_boundary)))
                    Image.fromarray(output_boundary).save(os.path.join(save_path, 'boundary', image_names[step]))
            elif network == 'CGSDNet-Depth':
                predicts = model(image_var)
                output_mask = predicts[-1]
                output_mask = output_mask.data.squeeze(0).cpu()
                output_mask = np.array(transforms.Resize((height, width), interpolation=Image.BICUBIC)
                                       (transforms.ToPILImage()(output_mask)))
                Image.fromarray(output_mask).save(os.path.join(save_path, 'mask', image_names[step]))
                if save_boundary:
                    output_boundary = predicts[4]
                    output_boundary = output_boundary.data.squeeze(0).cpu()
                    output_boundary = np.array(transforms.Resize((height, width), interpolation=Image.BICUBIC)
                                               (transforms.ToPILImage()(output_boundary)))
                    Image.fromarray(output_boundary).save(os.path.join(save_path, 'boundary', image_names[step]))
                if save_depth is True:
                    output_depth = predicts[-2] * 1000.0
                    output_depth = F.interpolate(output_depth, size=(height, width), mode='bicubic',
                                                 align_corners=False)
                    output_depth = output_depth.data.squeeze(0).squeeze(0).cpu()
                    output_depth = np.array(output_depth).astype(np.uint16)
                    cv.imwrite(os.path.join(save_path, 'depth_GS', image_names[step]), output_depth)
                    depth_color = visualize_depth(output_depth)
                    cv.imwrite(os.path.join(save_path, 'depth_GS_vis', image_names[step]), depth_color)
        detect_bar.desc = "detecting glass"


def main():
    image_names, images = read(args.root_path)

    img_transform = transforms.Compose([
        transforms.Resize((args.image_scale, args.image_scale), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model = get_model(network=args.network, backbone=args.backbone, skip_top=args.skip_top, max_depth=args.max_depth,
                      trained_model=args.trained_model, device=args.device)

    folder_detect(network=args.network, image_names=image_names, images=images, model=model,
                  image_transform=img_transform, root_path=args.root_path, save_boundary=args.save_boundary,
                  save_depth=args.save_depth, device=args.device)


if __name__ == '__main__':
    main()
