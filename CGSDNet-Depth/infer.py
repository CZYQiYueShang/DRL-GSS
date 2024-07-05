import os
import sys
import time
import argparse

import torch
from torch import nn
from torchvision import transforms
from torch.autograd import Variable

import numpy as np
from PIL import Image
from tqdm import tqdm

from model.CGSDNet import CGSDNet


parser = argparse.ArgumentParser(description='Inference for Glass Surface Detection')
# network
parser.add_argument('--network', type=str, default='CGSDNet')
parser.add_argument('--backbone', type=str, default='convnext_base')
parser.add_argument('--ckpt_name', type=str, default='ConvNeXt-200')
# parser.add_argument('--backbone', type=str, default='resnet101_deep')
# parser.add_argument('--ckpt_name', type=str, default='ResNet-200')
parser.add_argument('--skip_top', action='store_true', default=True)
parser.add_argument('--device', type=str, default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
# inference
parser.add_argument('--dataset_root', type=str, default='./dataset/GDD')
parser.add_argument('--boundary_map', action='store_true', default=True)
parser.add_argument('--image_scale', type=int, default=416)
args = parser.parse_args()


image_transform_fn = transforms.Compose([transforms.Resize((args.image_scale, args.image_scale), interpolation=Image.BICUBIC),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
to_pil = transforms.ToPILImage()


def get_model(network: str,
              backbone: str,
              ckpt_name: str,
              skip_top: bool = True,
              device: torch.device = 'cpu') -> nn.Module:
    if network == 'CGSDNet':
        model = CGSDNet(backbone_path=backbone, skip_top=skip_top, device=device)
    else:
        raise ValueError('No such network model!')

    trained_model_path = os.path.join('./ckpt/trained', network, ckpt_name + '.pth')
    if os.path.exists(trained_model_path):
        print("Load %s's trained model %s for testing" % (network, ckpt_name))
        trained_dict = torch.load(trained_model_path, map_location=device)
        model_dict = model.state_dict()
        trained_dict = {k: v for k, v in trained_dict.items() if (k in model_dict and 'Prediction' not in k)}
        model_dict.update(trained_dict)
        model.load_state_dict(model_dict)
        print("Successfully loaded trained model %s!" % trained_model_path)
    else:
        print("No such trained model %s!" % trained_model_path)

    model.to(device)
    print("Successfully got network model of %s!" % network)
    return model


def infer(network: str,
          model: nn.Module,
          ckpt_name: str,
          dataset_root: str,
          boundary_map: bool = True,
          device: torch.device = 'cpu') -> None:
    image_root = os.path.join(dataset_root, 'test', 'image')
    image_list = os.listdir(image_root)

    save_root = os.path.join(dataset_root, 'result', network, ckpt_name)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    if boundary_map is True:
        boundary_save_root = os.path.join(save_root, 'boundary')
        if not os.path.exists(boundary_save_root):
            os.makedirs(boundary_save_root)

    infer_bar = tqdm(image_list, file=sys.stdout)

    model.eval()
    with torch.no_grad():
        start_time = time.time()
        for step, image_name in enumerate(infer_bar):
            image = Image.open(os.path.join(image_root, image_name))
            if image.mode != 'RGB':
                image = image.convert('RGB')
                print("%s is a gray image" % image_name)
            w, h = image.size

            if network == 'CGSDNet':
                image_var = Variable(image_transform_fn(image).unsqueeze(0)).to(device)
                outputs = model(image_var)
                output = outputs[-1]
                output = output.data.squeeze(0).cpu()
                output = np.array(transforms.Resize((h, w), interpolation=Image.BICUBIC)(to_pil(output)))
                Image.fromarray(output).save(os.path.join(save_root, image_name[:-4] + ".png"))
                if boundary_map is True:
                    output_boundary = outputs[-2]
                    output_boundary = output_boundary.data.squeeze(0).cpu()
                    output_boundary = np.array(
                        transforms.Resize((h, w), interpolation=Image.BICUBIC)(to_pil(output_boundary)))
                    Image.fromarray(output_boundary).save(
                        os.path.join(boundary_save_root, image_name[:-4] + "_boundary.png"))
            else:
                raise ValueError('No such network model!')

        end_time = time.time()
        average_time = (end_time - start_time) / len(image_list)
        print("Average time is: %.3f" % average_time)


def main():
    model = get_model(args.network, args.backbone, args.ckpt_name, args.skip_top, args.device)

    infer(args.network, model, args.ckpt_name, args.dataset_root, args.boundary_map, args.device)


if __name__ == '__main__':
    main()
