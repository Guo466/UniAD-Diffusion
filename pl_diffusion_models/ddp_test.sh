#!/usr/bin/env bash
CONFIG=$1
CKPT=$2
WORK_DIRS=${3:-./work_dirs}
PY_ARGS=${@:4}

# export NCCL_DEBUG=INFO
export NCCL_IB_GID_INDEX=5
export NCCL_P2P_DISABLE=1

#example original use:  bash ddp_test.sh ./config/dataset/LitDiffusionDataset_ego_navi_fix_distance_path_oss_dlp.yaml ./ckpts/2025_12_25_10_00_00.pth
#example use case dir and noise dir num_samples:  bash ddp_test.sh ./config/dataset/LitDiffusionDataset_ego_navi_fix_distance_path_oss_dlp.yaml ./ckpts/2025_12_25_10_00_00.pth work_dirs --case_dir=./2025_01_01_05_04 --noise_dir ./noise
#example use case dir and num_samples:  bash ddp_test.sh ./config/dataset/LitDiffusionDataset_ego_navi_fix_distance_path_oss_dlp.yaml ./ckpts/2025_12_25_10_00_00.pth work_dirs --case_dir=./2025_01_01_05_04 --num_samples 16

# infer
python \
    -m torch.distributed.launch \
    --nnodes 1 \
    --nproc_per_node 1 \
    --use_env \
    --master_port 29506 \
    tools/test_dlp.py --work_dir=${WORK_DIRS} --config=${CONFIG} --ckpt=${CKPT} ${PY_ARGS} 2>&1|tee $log_file

python \
    -m torch.distributed.launch \
    --nnodes 1 \
    --node_rank 0 \
    --master_port 29507 \
    --nproc_per_node 1 \
    --use_env \
    tools/infer_visualization.py --work_dir=${WORK_DIRS} 2>&1|tee $log_file
