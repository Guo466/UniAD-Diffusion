#!/usr/bin/env bash
CONFIG=$1
CKPT=$2
WORK_DIRS=$3
SDK_DUMP_DIR=$4

if [[ -z $SDK_DUMP_DIR ]]; then
    echo "Need specify sdk dump dir."
    exit 1
fi

# export NCCL_DEBUG=INFO
export NCCL_IB_GID_INDEX=5
export NCCL_P2P_DISABLE=1

WORK_DIRS=${WORK_DIRS}/work_dirs/infer_test_dump
mkdir -p $WORK_DIRS
log_file=$WORK_DIRS/test.log

python \
    -m torch.distributed.launch \
    --nnodes 1 \
    --nproc_per_node 1 \
    --use_env \
    --master_port 29506 \
    tools/test_dlp.py --work_dir=${WORK_DIRS} --config=${CONFIG} --ckpt=${CKPT}  --sdk_dump_dir=$SDK_DUMP_DIR 2>&1|tee $log_file

python \
    -m torch.distributed.launch \
    --nnodes 1 \
    --node_rank 0 \
    --master_port 29507 \
    --nproc_per_node 1 \
    --use_env \
    tools/infer_visualization.py --work_dir=${WORK_DIRS} --copy_img 2>&1|tee $log_file

ffmpeg -y -i ${WORK_DIRS}/infer/full.mp4 -vcodec h264 ${WORK_DIRS}/infer/full1.mp4 -nostdin

python \
    -m torch.distributed.launch \
    --nnodes 1 \
    --nproc_per_node 1 \
    --use_env \
    --master_port 29506 \
    tools/compare_torch_and_sdk.py --work_dir=${WORK_DIRS} 2>&1|tee $log_file

