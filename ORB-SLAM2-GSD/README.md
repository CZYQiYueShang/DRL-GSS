# ORB-SLAM2-GSD
This is a project based on [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2).

### Dataset
Please download our `GS RGB-D` dataset at [here](https://drive.google.com/file/d/1GJxv5ICyocRUQhu3hG2LMnpaq1ee0dvG/view?usp=drive_link).

Unzip it in the current folder after downloading.

Use `infer_GS-RGB-D.py` in [CGSDNet-Depth](https://github.com/CZYQiYueShang/DRL-GSS/tree/main/CGSDNet-Depth) to perform glass surface segmentation and depth estimation on the sequences in `GS RGB-D`.

### Build
The build process is the same as [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2).

Since point cloud related operations have been added, the PCL library file needs to be installed.
```
sudo apt install libpcl-dev
```

### Test
Run the `test.sh` to test.
```
./test.sh
```
