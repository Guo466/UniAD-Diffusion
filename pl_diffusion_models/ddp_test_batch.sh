#!/usr/bin/env bash
# use: bash ddp_test_batch.sh  dif_09a_occ_1225 /mnt/afs/liuzhaoyang1/diffusion_codes/dif_09a_occ_1225/lightning_logs/version_1/checkpoints/last.ckpt
MODEL_NAME=${1:-""}
CKPT_PATH=${2:-""}
CONFIG=${3:-./config/dataset/LitDiffusionDataset_ego_navi_fix_distance_path_oss_dlp.yaml}
PY_ARGS=${@:4}
WORK_DIRS=./work_dirs/
echo "MODEL_NAME: $MODEL_NAME"
echo "CKPT_PATH: $CKPT_PATH"
echo "CONFIG: $CONFIG"

tmp_data_dir_test_list=(
    /mnt/afs/liangzhihui/dlp-data-factory/2025_11_30_09_50_22
    /mnt/afs/liangzhihui/dlp-data-factory/2025_11_30_13_18_09
    # /mnt/afs/liuzhaoyang1/split_scene/daily1206_roadedge_pro/2025_12_05_21_14_25
    # /mnt/afs/liuzhaoyang1/split_scene/daily1202_pro/2025_11_28_18_16_35
    # /mnt/afs/liuzhaoyang1/split_scene/daily1202_pro/2025_11_28_08_27_27
    # /mnt/afs/liuzhaoyang1/split_scene/daily1216_pro/2025_12_02_17_36_53
)

# 参数检查
if [ -z "$CKPT_PATH" ] || [ -z "$MODEL_NAME" ]; then
    echo "Usage: <CKPT_PATH>  <MODEL_NAME>"
    exit 1
fi

# 文件存在性检查
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file $CONFIG does not exist."
    exit 1
fi

# 文件存在性检查
if [ ! -f "$CKPT_PATH" ]; then
    echo "Error: Checkpoint file $CKPT_PATH does not exist."
    exit 1
fi


for tmp_data_dir_test in "${tmp_data_dir_test_list[@]}"; do
    export NCCL_IB_GID_INDEX=5   
    export NCCL_P2P_DISABLE=1  

    basename_dir=$(basename "$tmp_data_dir_test")
    MP4NAME=${MODEL_NAME}_${basename_dir}
    echo "生成的视频文件名: $MP4NAME, 数据集名称: $basename_dir, 当前数据集目录: $tmp_data_dir_test"

    date_split_file_test=${tmp_data_dir_test}/date_split.json
    sed -i \
    "s|^\(\s*tmp_data_dir_test\s*:\s*\).*|\1${tmp_data_dir_test}|" \
    "$CONFIG"

    sed -i \
    "s|^\(\s*date_split_file_test\s*:\s*\).*|\1${date_split_file_test}|" \
    "$CONFIG"

    # infer
    python \
        -m torch.distributed.launch \
        --nnodes 1 \
        --nproc_per_node 1 \
        --use_env \
        --master_port 29506 \
        tools/test_dlp.py --work_dir=${WORK_DIRS} --config=${CONFIG} --ckpt=${CKPT_PATH} ${PY_ARGS} 2>&1|tee $log_file

    # visualization
    python \
        -m torch.distributed.launch \
        --nnodes 1 \
        --node_rank 0 \
        --master_port 29507 \
        --nproc_per_node 1 \
        --use_env \
        tools/infer_visualization.py --work_dir=${WORK_DIRS} --no_draw_fixed 2>&1|tee $log_file

    mv ${WORK_DIRS}/infer/full.mp4 ${WORK_DIRS}/infer/${MP4NAME}.mp4

    echo "------------------------------------------------------------"

done
