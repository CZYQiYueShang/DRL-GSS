from typing import Union, Tuple, List

import torch
from torch import nn, Tensor


# Binary Cross-Entropy (BCE) loss
class BCELoss(nn.BCELoss):
    def __init__(self,
                 reduction: str = 'mean') -> None:
        super(BCELoss, self).__init__(reduction=reduction)


# Intersection over Union (IoU) loss
class IoULoss(nn.Module):
    def __init__(self,
                 smooth: float = 0.0) -> None:
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self,
                inputs: Tensor,
                targets: Tensor) -> Tensor:
        # 每个batch的图片拼接在一起计算IoU
        # input = input.view(-1)
        # target = target.view(-1)

        # 每个batch的图片分别计算IoU后取平均值
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + self.smooth) / (union + self.smooth)
        IoU_loss = 1 - IoU
        return IoU_loss


# Dice loss
# Code Adapted from:
# https://github.com/hehao13/EBLNet/blob/main/loss.py
class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target, valid_mask):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        device = predict.device
        target = target.contiguous().view(target.shape[0], -1)
        target_gpu = target.clone().cuda(device=device)
        valid_mask_gpu = valid_mask.clone().cuda(device=device)
        valid_mask_gpu = valid_mask_gpu.contiguous().view(valid_mask.shape[0], -1)

        num = torch.sum(torch.mul(predict, target_gpu) * valid_mask_gpu, dim=1) * 2 + self.smooth
        den = torch.sum((predict.pow(self.p) + target_gpu.pow(self.p)) * valid_mask_gpu, dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


# Context loss for CAP module
class ContextLoss(nn.Module):
    def __init__(self,
                 bce_reduction: str = 'mean') -> None:
        super(ContextLoss, self).__init__()
        self.bce_loss_fn = BCELoss(reduction=bce_reduction)
        self.iou_loss_fn = IoULoss()

    def forward(self,
                inputs: Tensor,
                targets: Tensor) -> Tensor:
        context_loss = self.bce_loss_fn(inputs, targets) + self.iou_loss_fn(inputs, targets)
        return context_loss


# Boundary loss for HBD module
class BoundaryLoss(nn.Module):
    def __init__(self,
                 bce_reduction: str = 'mean',
                 dice_smooth: float = 1,
                 dice_reduction: str = 'mean') -> None:
        super(BoundaryLoss, self).__init__()
        self.bce_loss_fn = BCELoss(reduction=bce_reduction)
        self.dice_loss_fn = BinaryDiceLoss(smooth=dice_smooth, reduction=dice_reduction)

    def forward(self,
                inputs: Tensor,
                targets: Tensor,
                valid: Tensor) -> Tensor:
        boundary_loss = self.bce_loss_fn(inputs, targets) + self.dice_loss_fn(inputs, targets, valid)
        return boundary_loss


# overall loss for CGSDNet
class CGSDNetLoss(nn.Module):
    def __init__(self,
                 weight: Tuple[int, ...] = (1, 1, 1, 1, 3, 2),
                 boundary_map: bool = True,
                 bce_reduction: str = 'mean',
                 dice_smooth: float = 1,
                 dice_reduction: str = 'mean') -> None:
        super(CGSDNetLoss, self).__init__()
        self.weight = weight
        self.boundary_map = boundary_map
        self.context_loss_fn = ContextLoss(bce_reduction=bce_reduction)
        self.boundary_loss_fn = BoundaryLoss(bce_reduction=bce_reduction, dice_smooth=dice_smooth,
                                             dice_reduction=dice_reduction)

    def forward(self,
                inputs: Tuple[Tensor],
                targets: Union[Tuple[Tensor], Tensor]) -> Tensor:
        total_loss = 0
        if self.boundary_map is False:
            for i in range(len(inputs)):
                total_loss = total_loss + self.context_loss_fn(inputs[i], targets) * self.weight[i]
        else:
            valid = torch.ones_like(targets[1])
            for i in range(4):
                total_loss = total_loss + self.context_loss_fn(inputs[i], targets[0]) * self.weight[i]

            # total_loss = total_loss + self.context_loss_fn(inputs[-2], targets[1]) * self.weight[-2]
            total_loss = total_loss + self.boundary_loss_fn(inputs[-2], targets[1], valid) * self.weight[-2]
            total_loss = total_loss + self.context_loss_fn(inputs[-1], targets[0]) * self.weight[-1]
        return total_loss


class SilogLoss(nn.Module):
    def __init__(self,
                 variance_focus: float = 0.85,
                 log_depth_error: bool = True,
                 min_depth: float = 0.2,
                 max_depth: float = 10.0) -> None:
        super(SilogLoss, self).__init__()
        self.variance_focus = variance_focus
        self.log_depth_error = log_depth_error
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self,
                inputs: Tensor,
                targets: Tensor) -> Tensor:
        valid = (targets >= self.min_depth) & (targets < self.max_depth)
        if self.log_depth_error:
            d = torch.log(inputs[valid]) - torch.log(targets[valid])
            return ((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10
        else:
            valid_inputs = inputs[valid]
            valid_targets = targets[valid]
            d = (valid_inputs + torch.log(valid_inputs)) - (valid_targets + torch.log(valid_targets))
            return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10


# overall loss for CGSDNet-Depth
class CGSDNetDepthLoss(nn.Module):
    def __init__(self,
                 weight: Tuple[int, ...] = (1, 1, 1, 1, 3, 5, 2),
                 boundary_map: bool = True,
                 bce_reduction: str = 'mean',
                 dice_smooth: float = 1,
                 dice_reduction: str = 'mean',
                 variance_focus: float = 0.85,
                 log_depth_error: bool = True,
                 min_depth: float = 0.2,
                 max_depth: float = 10.0) -> None:
        super(CGSDNetDepthLoss, self).__init__()
        self.weight = weight
        self.boundary_map = boundary_map
        self.context_loss_fn = ContextLoss(bce_reduction=bce_reduction)
        self.boundary_loss_fn = BoundaryLoss(bce_reduction=bce_reduction, dice_smooth=dice_smooth,
                                             dice_reduction=dice_reduction)
        self.depth_loss_fn = SilogLoss(variance_focus=variance_focus, log_depth_error=log_depth_error,
                                       min_depth=min_depth, max_depth=max_depth)

    def forward(self,
                inputs: Tuple[Tensor],
                targets: Union[Tuple[Tensor], Tensor]) -> Tuple[Tensor, List[Tensor]]:
        loss_list: List[Tensor] = []
        for i in range(4):
            loss_list.append(self.context_loss_fn(inputs[i], targets[0]) * self.weight[i])
        if self.boundary_map is False:
            for i in range(5):
                loss_list.append(self.depth_loss_fn(inputs[4 + i], targets[1]) * self.weight[4 + i])
        else:
            valid = torch.ones_like(targets[1])
            loss_list.append(self.boundary_loss_fn(inputs[4], targets[1], valid) * self.weight[4])
            for i in range(5):
                loss_list.append(self.depth_loss_fn(inputs[5 + i], targets[3]) * self.weight[5 + i])
        loss_list.append(self.context_loss_fn(inputs[-1], targets[0]) * self.weight[-1])
        total_loss = sum(loss_list)
        return total_loss, loss_list
