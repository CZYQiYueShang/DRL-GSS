import os
import sys
import argparse
from typing import Tuple, Any

import numpy as np
from tqdm import tqdm

from torch import Tensor
from torch.utils.data import Dataset

from dataset import GSD
from transform import mask


parser = argparse.ArgumentParser(description='Evaluation for Glass Surface Detection')
parser.add_argument('--dataset_root', type=str, default='./dataset/GDD')
parser.add_argument('--network', type=str, default='CGSDNet')
parser.add_argument('--ckpt_name', type=str, default='ConvNeXt-200')
parser.add_argument('--image_scale', type=int, default=416)
parser.add_argument('--class_num', type=int, default=2)
parser.add_argument('--beta_square', type=float, default=0.3)
args = parser.parse_args()


def compute_fast_hist(label_true: np.ndarray,
                      label_pred: np.ndarray,
                      n_class: int) -> np.ndarray:
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(n_class * label_true[mask].astype(int) +
                       label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def get_confmat(gt_mask_set: Dataset,
                predict_mask_set: Dataset,
                class_num: int) -> Tuple[Any, float, float]:
    confmat = 0
    MAE_sum: float = 0
    BER_sum: float = 0

    mask_num = len(gt_mask_set)
    print('There are %d pictures in the test set totally' % mask_num)
    eval_bar = tqdm(range(mask_num), file=sys.stdout)
    for step, data in enumerate(eval_bar):
        gt_mask = np.array(gt_mask_set[data][1], dtype=np.float32)
        predict_mask = np.array(predict_mask_set[data][1], dtype=np.float32)
        gt_mask = np.where(gt_mask >= 127.5, 1, 0).astype(np.float32)
        predict_mask = np.where(predict_mask >= 127.5, 1, 0).astype(np.float32)

        MAE_sum += np.mean(abs(predict_mask - gt_mask)).item()

        N_p = np.sum(gt_mask)
        N_n = np.sum(np.logical_not(gt_mask))

        TP = np.sum(np.logical_and(predict_mask, gt_mask))
        TN = np.sum(np.logical_and(np.logical_not(predict_mask), np.logical_not(gt_mask)))

        if N_p == 0:
            N_p += 1
            TP += 1
        if N_n == 0:
            N_n += 1
            TN += 1

        BER_sum += 100 * (1 - 0.5 * ((TP / N_p) + (TN / N_n)))

        label_true = np.array(gt_mask, dtype=np.int32)
        label_pred = np.array(predict_mask, dtype=np.int32)
        confmat += compute_fast_hist(label_true, label_pred, class_num)
    MAE = MAE_sum / mask_num
    BER = BER_sum / mask_num
    return confmat, MAE, BER


def get_val_confmat(predict_masks: Tensor,
                    gt_masks: Tensor,
                    class_num:int = 2) -> Any:
    predict_masks = np.array(predict_masks, dtype=np.float32)
    predict_masks = np.where(predict_masks >= 0.5, 1, 0).astype(np.float32)

    label_pred = np.array(predict_masks, dtype=np.int32)
    label_true = np.array(gt_masks, dtype=np.int32)
    confmat = compute_fast_hist(label_true, label_pred, class_num)
    return confmat


def confmat_to_metric(confmat: Any,
                      beta_square: float = 0.3) -> Tuple[float, float, float]:
    TP = confmat[1][1]
    TN = confmat[0][0]
    FP = confmat[0][1]
    FN = confmat[1][0]
    IoU = float(TP / (TP + FN + FP))
    PA = float((TP + TN) / (TP + FN + FP + TN))
    precision = float(TP / (TP + FP))
    recall = float(TP / (TP + FN))
    FB = float(((1 + beta_square) * precision * recall) / (beta_square * precision + recall))
    return IoU, PA, FB


def get_eval_metric(gt_mask_set: Dataset,
                    predict_mask_set: Dataset,
                    class_num: int = 2,
                    beta_square: float = 0.3) -> Tuple[float, float, float, float, float]:
    confmat, MAE, BER = get_confmat(gt_mask_set, predict_mask_set, class_num)
    IoU, PA, FB = confmat_to_metric(confmat, beta_square)
    print('IoU: %.4f  PA: %.4f  FB: %.4f  MAE: %.4f  BER: %.4f' % (IoU, PA, FB, MAE, BER))
    return IoU, PA, FB, MAE, BER


def main():
    predict_mask_set = GSD.MaskDataset(os.path.join(args.dataset_root, 'result', args.network),
                                       folder_name=args.ckpt_name, transform_fn=[mask.Resize(args.image_scale)])
    gt_mask_set = GSD.MaskDataset(os.path.join(args.dataset_root, 'test'), transform_fn=[mask.Resize(args.image_scale)])

    get_eval_metric(gt_mask_set, predict_mask_set, args.class_num, args.beta_square)


if __name__ == '__main__':
    main()

    '''
    CGSDNet
    convnext_base  200 (IoU: 0.8970  PA: 0.9496  FB: 0.9461  MAE: 0.0504  BER: 4.7951)
    
    resnet101_deep 200 (IoU: 0.8914  PA: 0.9468  FB: 0.9436  MAE: 0.0532  BER: 5.3486)
    '''
