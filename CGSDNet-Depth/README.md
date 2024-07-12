# CGSDNet-Depth

### Dataset
Please download from the corresponding pages of [GDD](https://github.com/Mhaiyang/CVPR2020_GDNet) and [GW-Depth](https://github.com/ViktorLiang/GW-Depth) and put them in the `dataset` folder.

### Requirements
* Python 3.10
* CUDA 11.7
```
pip install -r requirements.txt
```

### Model
Place models in the `ckpt` folder.
| Network | Backbone | Download |
|:---------------|:----:|:---:|
| CGSDNet | ResNet-101 | [Google Drive](https://drive.google.com/file/d/16EF26d35_UfAeLvtU2G8mp_2gHB5kiHC/view?usp=sharing) |
| CGSDNet | ConvNeXt-B | [Google Drive](https://drive.google.com/file/d/1EIatOGUwqvOgMIpf-0bKPR5XvrgYzpII/view?usp=drive_link) |
| CGSDNet-Depth | ConvNeXt-B | [Google Drive](https://drive.google.com/file/d/1GvsAb2rhFtBJxXlr54cTSrIiOOIh79IS/view?usp=drive_link) |
| CGSDNet-Depth (for ORB-SLAM2-GSD) | ConvNeXt-B | [Google Drive](https://drive.google.com/file/d/1kIbamJen_XBf_wThNiHnVrgjt9Efyv4s/view?usp=drive_link) |

### Test
Use `infer_depth.py` to test after downloading the datasets and models.
```
python infer_depth.py
```
