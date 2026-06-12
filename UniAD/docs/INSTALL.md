# Installation
> Modified from bevformer and mmdetection3d.

**a. Env: Create a conda virtual environment and activate it.**
```shell
conda create -n uniad2.0 python=3.9 -y
conda activate uniad2.0
```

**b. Torch: Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

**c. Install mmcv-series packages.**
```shell
pip install -v mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
pip install mmdet==2.26.0 mmsegmentation==0.29.1 mmdet3d==1.0.0rc6
```


**d. Install UniAD.**
```shell
cd ~
git clone https://github.com/OpenDriveLab/UniAD.git
cd UniAD
pip install -r requirements.txt
```


**e. Prepare pretrained weights.**

We release our pretrained weights in [HuggingFace::OpenDriveLab/UniAD2.0_R101_nuScenes](https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/tree/main/ckpts)

```shell
mkdir ckpts & cd ckpts
# r101_dcn_fcos3d_pretrain.pth (from bevformer)
wget https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/resolve/main/ckpts/r101_dcn_fcos3d_pretrain.pth

# bevformer_r101_dcn_24ep.pth
wget https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/resolve/main/ckpts/bevformer_r101_dcn_24ep.pth

# uniad_base_track_map.pth
wget https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/resolve/main/ckpts/uniad_base_track_map.pth

# uniad_base_e2e.pth
wget https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/resolve/main/ckpts/uniad_base_e2e.pth
```

---
-> Next Page: [Prepare The Dataset](./DATA_PREP.md)