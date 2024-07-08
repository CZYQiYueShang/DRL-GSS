# DRL-GSS

## Dense Reconstruction and Localization in Scenes with Glass Surfaces Based on ORB-SLAM2

### Abstract
In recent years, Visual Simultaneous Localization and Mapping (SLAM) research has made significant strides, particularly in the domain of RGB-D SLAM. However, the prevalent presence of glass surfaces poses a substantial challenge, impeding the effective performance of RGB-D SLAM in modern indoor environments. This challenge stems from the transparent, refractive, and reflective properties of glass surfaces, causing RGB-D cameras to struggle to obtain accurate depth information, consequently negatively impacting on the estimation of camera trajectories and the reconstruction of glass surfaces. In this paper, we propose a novel network designed for simultaneous glass surface segmentation and depth estimation called CGSDNet-Depth. Leveraging a novel Context Guided Depth Decoder (CGDD), CGSDNet-Depth generates depth information guided by the contextual information of glass surfaces. Subsequently, based on ORB-SLAM2, we introduce a new method named ORB-SLAM2-GSD that utilizes the segmentation and depth estimation results from CGSDNet-Depth to alleviate the adverse effects of glass surfaces on camera trajectory estimation and dense reconstruction. Additionally, we construct the first RGB-D dataset for glass surface scenes, comprising 8 image sequences, called GS RGB-D. Extensive experiments demonstrate that our method outperforms other State-of-the-Art (SOTA) methods in glass surface segmentation and improves ORB-SLAM2 performance in glass surface scenes.

### Test
Please check the two folders [CGSDNet-Depth](https://github.com/CZYQiYueShang/DRL-GSS/tree/main/CGSDNet-Depth) and [ORB-SLAM2-GSD](https://github.com/CZYQiYueShang/DRL-GSS/tree/main/ORB-SLAM2-GSD) respectively.

### Contact
E-Mail: chen.zeyuan.tkb_gu@u.tsukuba.ac.jp
