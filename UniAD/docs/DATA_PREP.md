

# nuScenes
Download nuScenes V1.0 full dataset data, CAN bus and map(v1.3) extensions [HERE](https://www.nuscenes.org/download), then follow the steps below to prepare the data.


**Download nuScenes, CAN_bus and Map extensions**
```shell
cd UniAD
mkdir data
# Download nuScenes V1.0 full dataset data directly to (or soft link to) UniAD/data/
# Download CAN_bus and Map(v1.3) extensions directly to (or soft link to) UniAD/data/nuscenes/
```

**Prepare UniAD data info**

*Option1: We have already prepared the off-the-shelf data infos for you in [HuggingFace::OpenDriveLab/UniAD2.0_R101_nuScenes](https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/tree/main/data):*
```shell
cd UniAD/data
mkdir infos && cd infos
wget https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/resolve/main/data/nuscenes_infos_temporal_train.pkl # train_infos

wget https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/resolve/main/data/nuscenes_infos_temporal_val.pkl # val_infos
```


*Option2: You can also generate the data infos by yourself:*
> The generated data path will contain the root directory. Please remember to change the `data_root` to empty in config files if using your generated pkl. Refer to https://github.com/OpenDriveLab/UniAD/issues/13.
```shell
cd UniAD/data
mkdir infos
./tools/uniad_create_data.sh
# This will generate nuscenes_infos_temporal_{train,val}.pkl
```

**Prepare Motion Anchors**

We already upload motion anchors in [HuggingFace::OpenDriveLab/UniAD2.0_R101_nuScenes](https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/tree/main/data)
```shell
cd UniAD/data
mkdir others && cd others
wget https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/resolve/main/data/motion_anchor_infos_mode6.pkl
```

**The Overall Structure**

*Please make sure the structure of UniAD is as follows:*
```
UniAD
├── projects/
├── tools/
├── ckpts/
│   ├── bevformer_r101_dcn_24ep.pth
│   ├── uniad_base_track_map.pth
|   ├── uniad_base_e2e.pth
├── data/
│   ├── nuscenes/
│   │   ├── can_bus/
│   │   ├── maps/
│   │   ├── lidarseg/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
│   │   ├── v1.0-trainval/
│   │   ├── v1.0-mini/ 
│   ├── infos/
│   │   ├── nuscenes_infos_temporal_train.pkl
│   │   ├── nuscenes_infos_temporal_val.pkl
│   ├── others/
│   │   ├── motion_anchor_infos_mode6.pkl
```
---
<- Last Page:  [Installation](./INSTALL.md)

-> Next Page: [Train/Eval UniAD](./TRAIN_EVAL.md)