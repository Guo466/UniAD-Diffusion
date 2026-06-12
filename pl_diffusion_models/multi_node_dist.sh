#!/bin/bash
echo "NCCL_DEBUG = ${NCCL_DEBUG:-INFO}"
echo "NCCL_IB_TIMEOUT = ${NCCL_IB_TIMEOUT:-23}"
echo "NCCL_IB_RETRY_CNT = ${NCCL_IB_RETRY_CNT:-7}"
echo "MLP_WORKER_GPU = ${MLP_WORKER_GPU:-1}"
echo "MLP_WORKER_NUM = ${MLP_WORKER_NUM:-1}"
echo "MLP_ROLE_INDEX = ${MLP_ROLE_INDEX:-0}"
echo "MLP_WORKER_0_HOST = ${MLP_WORKER_0_HOST:-127.0.0.1}"
echo "MLP_WORKER_0_PORT = ${MLP_WORKER_0_PORT:-1234}"

# fit
torchrun \
    --nnodes $MLP_WORKER_NUM \
    --node_rank $MLP_ROLE_INDEX \
    --nproc_per_node $MLP_WORKER_GPU \
    --master_addr $MLP_WORKER_0_HOST \
    --master_port $MLP_WORKER_0_PORT \
    launch.py --compute_metrics False $@

# read config path from $@, match by --config   
CONFIG_PATH=$(echo $@ | grep -o -- '--config.*')
# echo "CONFIG_PATH: ${CONFIG_PATH}"

# test
# read best model path txt
# only run on worker 0
if [ ${MLP_ROLE_INDEX} == 0 ]; then
  BEST_MODEL_PATH=$(cat best_model_path.txt) && \
    python launch.py test ${CONFIG_PATH} --trainer.num_nodes 1  --trainer.devices 1 --ckpt_path ${BEST_MODEL_PATH} 
    rm best_model_path.txt # remove best_model_path.txt
  exit 1
fi