import os
import sys
import argparse
from typing import Tuple

import numpy as np
from tqdm import tqdm

from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from dataset import GSD, GWDepth
from transform import mask, depth
import eval


parser = argparse.ArgumentParser(description='Evaluation for Glass Surface Detection and Depth Estimation')
parser.add_argument('--dataset_root', type=str, default='./dataset/GW-Depth')
parser.add_argument('--network', type=str, default='CGSDNet-Depth')
parser.add_argument('--ckpt_name', type=str, default='ConvNeXt-200')
parser.add_argument('--image_scale', type=int, default=416)
parser.add_argument('--class_num', type=int, default=2)
parser.add_argument('--beta_square', type=float, default=0.3)
parser.add_argument('--min_depth', type=float, default=0.001)
parser.add_argument('--max_depth', type=float, default=10.0)
args = parser.parse_args()


def compute_depth_errors(gt_depth: np.array,
                         predict_depth: np.array) -> Tuple[float, float, float, float, float, float, float]:
    thresh = np.maximum((gt_depth / predict_depth), (predict_depth / gt_depth))
    sigma_1 = (thresh < 1.25).mean()
    sigma_2 = (thresh < 1.25 ** 2).mean()
    sigma_3 = (thresh < 1.25 ** 3).mean()

    REL = np.mean(np.abs(gt_depth - predict_depth) / gt_depth)
    # sq_rel = np.mean(((gt - pred) ** 2) / gt)

    RMS = (gt_depth - predict_depth) ** 2
    RMS = np.sqrt(RMS.mean())

    RMS_log = (np.log(gt_depth) - np.log(predict_depth)) ** 2
    RMS_log = np.sqrt(RMS_log.mean())

    # err = np.log(pred) - np.log(gt)
    # silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(predict_depth) - np.log10(gt_depth))
    log_10 = np.mean(err)
    return sigma_1, sigma_2, sigma_3, REL, RMS, RMS_log, log_10


def compute_val_metric(predict_depths: Tensor,
                       gt_depths: Tensor,
                       min_depth: float = 0.001,
                       max_depth: float = 10.0) -> Tuple[float, float, float]:
    predict_depths = np.array(predict_depths, dtype=np.float32)
    gt_depths = np.array(gt_depths, dtype=np.float32)

    predict_depths[predict_depths < min_depth] = min_depth
    predict_depths[predict_depths > max_depth] = max_depth
    predict_depths[np.isnan(predict_depths)] = min_depth
    predict_depths[np.isinf(predict_depths)] = max_depth
    valid_masks = np.logical_and(gt_depths > min_depth, gt_depths < max_depth)
    predict_depths = predict_depths[valid_masks]
    gt_depths = gt_depths[valid_masks]

    thresh = np.maximum((gt_depths / predict_depths), (predict_depths / gt_depths))
    sigma_1 = (thresh < 1.25).mean()

    REL = np.mean(np.abs(gt_depths - predict_depths) / gt_depths)

    RMS = (gt_depths - predict_depths) ** 2
    RMS = np.sqrt(RMS.mean())
    return sigma_1, REL, RMS


def get_eval_metric(gt_depth_set: Dataset,
                    predict_depth_set: Dataset,
                    min_depth: float = 0.001,
                    max_depth: float = 10.0) -> Tuple[float, float, float, float, float, float, float]:
    sigma_1_sum: float = 0
    sigma_2_sum: float = 0
    sigma_3_sum: float = 0
    REL_sum: float = 0
    RMS_sum: float = 0
    RMS_log_sum: float = 0
    log_10_sum: float = 0

    depth_num = len(gt_depth_set)
    print('There are %d pictures in the test set totally' % depth_num)
    eval_bar = tqdm(range(depth_num), file=sys.stdout)
    for step, data in enumerate(eval_bar):
        gt_depth = np.array(gt_depth_set[data][1], dtype=np.float32)
        predict_depth = np.array(predict_depth_set[data][1], dtype=np.float32)
        gt_depth = gt_depth / 1000.0
        predict_depth = predict_depth / 1000.0

        predict_depth[predict_depth < min_depth] = min_depth
        predict_depth[predict_depth > max_depth] = max_depth
        predict_depth[np.isnan(predict_depth)] = min_depth
        predict_depth[np.isinf(predict_depth)] = max_depth
        valid_mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)
        sigma_1, sigma_2, sigma_3, REL, RMS, RMS_log, log_10 = \
            compute_depth_errors(gt_depth[valid_mask], predict_depth[valid_mask])

        sigma_1_sum += sigma_1
        sigma_2_sum += sigma_2
        sigma_3_sum += sigma_3
        REL_sum += REL
        RMS_sum += RMS
        RMS_log_sum += RMS_log
        log_10_sum += log_10

    sigma_1 = sigma_1_sum / depth_num
    sigma_2 = sigma_2_sum / depth_num
    sigma_3 = sigma_3_sum / depth_num
    REL = REL_sum / depth_num
    RMS = RMS_sum / depth_num
    RMS_log = RMS_log_sum / depth_num
    log_10 = log_10_sum / depth_num
    print('S1: %.4f  S2: %.4f  S3: %.4f  REL: %.4f  RMS: %.4f  RMS_log: %.4f  log_10: %.4f' %
          (sigma_1, sigma_2, sigma_3, REL, RMS, RMS_log, log_10))
    return sigma_1, sigma_2, sigma_3, REL, RMS, RMS_log, log_10


def main():
    gt_mask_transform_fn = [mask.Resize(args.image_scale),
                            mask.MultiStandardize()]
    predict_mask_set = GSD.MaskDataset(os.path.join(args.dataset_root, 'result', args.network, args.ckpt_name),
                                       transform_fn=[mask.Resize(args.image_scale)])
    gt_mask_set = GWDepth.MaskDataset(args.dataset_root, dataset_type='val', transform_fn=gt_mask_transform_fn)
    eval.get_eval_metric(gt_mask_set, predict_mask_set, args.class_num, args.beta_square)

    depth_transform_fn = [depth.Resize(args.image_scale)]
    predict_depth_set = GWDepth.DepthDataset(args.dataset_root, dataset_type='result',
                                             folder_name=os.path.join(args.network, args.ckpt_name, 'depth'),
                                             transform_fn=depth_transform_fn)
    gt_depth_set = GWDepth.DepthDataset(args.dataset_root, dataset_type='val', transform_fn=depth_transform_fn)
    get_eval_metric(gt_depth_set, predict_depth_set, min_depth=args.min_depth, max_depth=args.max_depth)


if __name__ == '__main__':
    main()

    '''
    CGSDNet-Depth
    convnext_base 200 (IoU: 0.9528  PA: 0.9645  FB: 0.9765  MAE: 0.0355  BER: 5.9469)
                      (S1: 0.8935  S2: 0.9843  S3: 0.9970  REL: 0.1089  RMS: 0.3082  RMS_log: 0.1221  log_10: 0.0474)
    '''
