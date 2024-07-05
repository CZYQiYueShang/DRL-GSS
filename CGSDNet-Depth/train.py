import os
import sys
import argparse
from typing import Optional, Tuple, List

from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer, lr_scheduler

from dataset import GSD
from transform import common, image, mask
from model.CGSDNet import CGSDNet
from loss import CGSDNetLoss
from optimizer import SGD, AdamW, PolyScheduler, NullScheduler
from eval import get_val_confmat, confmat_to_metric


parser = argparse.ArgumentParser(description='Training for Glass Surface Detection')
# dataset
parser.add_argument('--image_scale', type=int, default=416)
parser.add_argument('--dataset_root', type=str, default='./dataset/GDD')
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--boundary_map', action='store_true', default=True)
parser.add_argument('--boundary_thickness', type=int, default=8)
parser.add_argument('--remove_edge', action='store_true', default=False)
parser.add_argument('--save_boundary', action='store_true', default=False)
# network
parser.add_argument('--network', type=str, default='CGSDNet')
# parser.add_argument('--backbone_path', type=str, default='./ckpt/pretrained/resnet101_deep.pth')
parser.add_argument('--backbone_path', type=str, default='./ckpt/pretrained/convnext_base.pth')
parser.add_argument('--skip_top', action='store_true', default=True)
parser.add_argument('--act_layer', type=nn.Module, default=nn.ReLU)
parser.add_argument('--device', type=str, default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
parser.add_argument('--pretrained_model', type=str, default=None)
# loss function
parser.add_argument('--loss_weight', type=Tuple[int], default=(1, 1, 1, 1, 3, 2))
parser.add_argument('--dice_smooth', type=float, default=1)
# optimizer
parser.add_argument('--optimizer_type', type=str, default='SGD')
parser.add_argument('--total_epochs', type=int, default=200)
parser.add_argument('--base_lr', type=float, default=0.001)
parser.add_argument('--lr_power', type=float, default=0.9)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--use_warmup', action='store_true', default=True)
parser.add_argument('--warmup_epochs', type=int, default=10)
parser.add_argument('--warmup_factor', type=float, default=0.001)
parser.add_argument('--scheduler_type', type=str, default='epoch')

args = parser.parse_args()
args.num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
# train
args.save_root = './ckpt/trained/%s' % args.network
if not os.path.exists(args.save_root):
    os.makedirs(args.save_root)
print(args)


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)


def get_data_loader(network: str,
                    image_scale: int,
                    dataset_root: str,
                    batch_size: int = 1,
                    num_workers: int = 1,
                    boundary_map: bool = False,
                    boundary_thickness: int = 8,
                    remove_edge: bool = False,
                    save_boundary: bool = False) -> Tuple[DataLoader, DataLoader]:
    train_common_transform_fn = [common.Resize(image_scale),
                                 common.RandomHorizontallyFlip(0.5)]
    val_common_transform_fn = [common.Resize(image_scale)]

    image_transform_fn = [image.ToTensor(),
                          image.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

    if network == 'CGSDNet':
        mask_transform_fn = [mask.MaskToTensor(),
                             mask.BinaryNormalize(),
                             mask.View((-1, image_scale, image_scale))]
    else:
        raise ValueError("No such data loader for %s!" % network)

    train_loader = GSD.get_data_loader(network, os.path.join(dataset_root, 'train'), batch_size=batch_size,
                                       num_workers=num_workers, common_transform_fn=train_common_transform_fn,
                                       image_transform_fn=image_transform_fn, mask_transform_fn=mask_transform_fn,
                                       boundary_map=boundary_map, boundary_thickness=boundary_thickness,
                                       remove_edge=remove_edge, save_boundary=save_boundary, is_train=True)
    val_loader = GSD.get_data_loader(network, os.path.join(dataset_root, 'test'), batch_size=batch_size,
                                     num_workers=num_workers, common_transform_fn=val_common_transform_fn,
                                     image_transform_fn=image_transform_fn, mask_transform_fn=mask_transform_fn)

    print("Successfully got Glass Surfaces' data loader for %s!" % network)
    return train_loader, val_loader


def get_model(network: str,
              backbone_path: str,
              skip_top: bool = True,
              act_layer: nn.Module = nn.ReLU,
              device: torch.device = 'cpu',
              pretrained_model: Optional[str] = None) -> nn.Module:
    if network == 'CGSDNet':
        model = CGSDNet(backbone_path=backbone_path, skip_top=skip_top, act_layer=act_layer, device=device)
    else:
        raise ValueError('No such network model!')

    if pretrained_model is not None:
        pretrained_dict = torch.load(pretrained_model, map_location=device)
        model_dict = model.state_dict()
        # print([k for k in model_dict.keys() if (k not in pretrained_dict)])
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'Prediction' not in k)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("Successfully loaded pretrained weights of %s!" % network)

    model.to(device)
    print("Successfully got network model of %s!" % network)
    return model


def get_loss_function(network: str,
                      device: torch.device = 'cpu',
                      loss_weight: Tuple[int] = (1, 1, 1, 1, 3, 2),
                      boundary_map: bool = True,
                      dice_smooth: float = 1) -> nn.Module:
    if network == 'CGSDNet':
        loss_fn = CGSDNetLoss(weight=loss_weight, boundary_map=boundary_map, dice_smooth=dice_smooth)
    else:
        raise ValueError("No such loss function for %s!" % network)

    loss_fn.to(device)
    print("Successfully got loss function for %s!" % network)
    return loss_fn


def get_optimizer(network: str,
                  optimizer_type: str,
                  model: nn.Module,
                  train_loader: DataLoader,
                  total_epochs: int,
                  base_lr: float = 0.001,
                  lr_power: float = 0.9,
                  momentum: float = 0.9,
                  weight_decay: float = 0.0005,
                  use_warmup: bool = False,
                  warmup_epochs: int = 10,
                  warmup_factor: float = 0.001,
                  scheduler_type: str = 'batch') -> Tuple[Optimizer, lr_scheduler.LambdaLR]:
    params = [p for p in model.parameters() if p.requires_grad]

    if optimizer_type == 'SGD':
        optim = SGD(params, lr=base_lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = PolyScheduler(optim, total_epochs, len(train_loader), lr_power=lr_power, use_warmup=use_warmup,
                                  warmup_epochs=warmup_epochs, warmup_factor=warmup_factor,
                                  scheduler_type=scheduler_type).get_poly_scheduler()
    elif optimizer_type == 'AdamW':
        optim = AdamW(params, lr=base_lr)
        scheduler = NullScheduler(optim).get_null_scheduler()
    else:
        raise ValueError("No such optimizer for %s!" % network)

    print("Successfully got %s optimizer for %s!" % (optimizer_type, network))
    return optim, scheduler


def train(network: str,
          train_loader: DataLoader,
          val_loader: DataLoader,
          model: nn.Module,
          loss_fn: nn.Module,
          optim: Optimizer,
          scheduler: lr_scheduler.LambdaLR,
          epochs: int,
          save_root: str,
          boundary_map: bool = True,
          device: torch.device = 'cpu') -> None:
    print("Start training process for %s!" % network)
    best_PA: float = 0.0
    best_log_out: str = ''
    train_steps = len(train_loader)

    # graph
    train_loss_list: List[float] = []
    IoU_list: List[float] = []
    PA_list: List[float] = []
    FB_list: List[float] = []

    # train and evaluation in one epoch
    for epoch in range(epochs):
        torch.cuda.empty_cache()

        # train
        model.train()
        running_loss: float = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)

        for step, data in enumerate(train_bar):
            if boundary_map is False:
                names, images, masks = data
                targets = masks.to(device)
            else:
                names, images, masks, masks_edge, masks_body = data
                targets = (masks.to(device), masks_edge.to(device), masks_body.to(device))

            predicts = model(images.to(device))
            loss = loss_fn(predicts, targets)

            optim.zero_grad()
            loss.backward()
            optim.step()

            running_loss += loss.item()
            train_bar.desc = "train epoch[%d/%d] loss: %.4f  learning_rate: %.6f" % \
                             (epoch + 1, epochs, loss, optim.param_groups[0]['lr'])

        # eval
        model.eval()
        confmat = 0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for data in val_bar:
                names, images, gt_masks = data

                predicts = model(images.to(device))
                if network == 'CGSDNet':
                    confmat += get_val_confmat(predicts[-1].cpu(), gt_masks)
                else:
                    raise ValueError('No such net model!')

                val_bar.desc = "val epoch[%d/%d]" % (epoch + 1, epochs)

        train_loss = running_loss / train_steps
        IoU, PA, FB = confmat_to_metric(confmat)
        log_out = '[epoch %d] train_loss: %.4f  IoU: %.4f  PA: %.4f  FB: %.4f  learning_rate: %.6f' % \
                  (epoch + 1, train_loss, IoU, PA, FB, optim.param_groups[0]['lr'])
        print(log_out)

        train_loss_list.append(train_loss)
        IoU_list.append(IoU)
        PA_list.append(PA)
        FB_list.append(FB)

        model_save_path = os.path.join(save_root, str(epoch + 1) + '.pth')
        if PA > best_PA:
            best_PA = PA
            torch.save(model.state_dict(), model_save_path)
            print('%dth model is saved' % (epoch + 1))
            best_log_out = log_out
        else:
            print('best model: %s' % best_log_out)

        scheduler.step()

        map_save_path = os.path.join(save_root, '%d_map.png' % epochs)
        plt.clf()
        plt.plot(train_loss_list, label='Training Loss')
        plt.plot(IoU_list, label='IoU')
        plt.plot(PA_list, label='PA')
        plt.plot(FB_list, label='FB')
        plt.legend()
        plt.savefig(map_save_path)

    print("Training process for %s has finished!" % network)


def main() -> None:
    train_loader, val_loader = get_data_loader(args.network, args.image_scale, args.dataset_root, args.batch_size,
                                               args.num_workers, args.boundary_map, args.boundary_thickness,
                                               args.remove_edge, args.save_boundary)

    model = get_model(args.network, args.backbone_path, args.skip_top, args.act_layer, args.device,
                      args.pretrained_model)

    loss_fn = get_loss_function(args.network, args.device, args.loss_weight, args.boundary_map, args.dice_smooth)

    optim, scheduler = get_optimizer(args.network, args.optimizer_type, model, train_loader, args.total_epochs,
                                     args.base_lr, args.lr_power, args.momentum, args.weight_decay, args.use_warmup,
                                     args.warmup_epochs, args.warmup_factor, args.scheduler_type)

    train(args.network, train_loader, val_loader, model, loss_fn, optim, scheduler, args.total_epochs, args.save_root,
          args.boundary_map, args.device)


if __name__ == '__main__':
    main()
