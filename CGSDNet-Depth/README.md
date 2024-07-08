# CGSDNet-Depth

### Dataset
Please download from the corresponding pages of [GDD](https://github.com/Mhaiyang/CVPR2020_GDNet) and [GW-Depth](https://github.com/ViktorLiang/GW-Depth) and put them in the `dataset` folder.

### Requirements
* Python 3.9
* CUDA 11.7
```
pip install -r requirements.txt
```

### Model
Place models in the `ckpt` folder.
| Network | Backbone | Download |
|:---------------|:----:|:---:|
| CGSDNet | ResNet-101 | [Google Drive] |
| CGSDNet | ConvNeXt-B | [Google Drive] |
| CGSDNet-Depth | ConvNeXt-B | [Google Drive] |
| CGSDNet-Depth (for ORB-SLAM2-GSD) | ConvNeXt-B | [Google Drive] |

### Test
Use `infer_depth.py` to test after downloading the datasets and models.
```
python infer_depth.py
```
