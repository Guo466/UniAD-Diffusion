#!/usr/bin/env bash
# ===========================================================================
# 单卡训练启动脚本（笔记本 8GB 显存专用）
# 用法：bash tools/run_train_single.sh <config_file> [extra_args...]
#
# 功能：
#   1. 自动创建带时间戳的 log 文件（work_dir/logs/train_YYYYMMDD_HHMMSS.log）
#   2. 同时输出到终端 + 写入 log 文件（tee）
#   3. 记录训练开始时间、结束时间、退出状态
#   4. 训练结束后自动打印 loss 曲线摘要（只需 grep 几个关键字段）
#   5. 若挂掉，退出码 != 0，log 末尾会写 "CRASHED" 便于快速定位
#
# 示例：
#   bash tools/run_train_single.sh \
#       projects/configs/stage2_e2e/base_e2e_diffusion.py
# ===========================================================================

set -euo pipefail

CFG=${1:?"用法: bash $0 <config_file>"}
shift  # 剩余参数传给 train.py

# ---------- 路径配置 ----------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(dirname "$SCRIPT_DIR")"

# work_dir：把 configs → work_dirs，去掉后缀
WORK_DIR=$(echo "${CFG%.*}" | sed -e "s/configs/work_dirs/g")
LOG_DIR="${WORK_DIR}/logs"
mkdir -p "${LOG_DIR}"

# 带时间戳的 log 文件名
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

# ---------- 打印启动信息 ----------
echo "=========================================="
echo "  UniAD 单卡训练启动"
echo "  配置文件: ${CFG}"
echo "  工作目录: ${WORK_DIR}"
echo "  日志文件: ${LOG_FILE}"
echo "  启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# ---------- 启动训练（同时输出到终端和 log 文件）----------
{
    echo "====== 训练开始: $(date '+%Y-%m-%d %H:%M:%S') ======"
    echo "配置文件: ${CFG}"
    echo "Python 可执行: $(which python)"
    echo ""

    # 单卡训练（无 torchrun，无 --launcher pytorch）
    PYTHONPATH="${PROJ_ROOT}:${PYTHONPATH:-}" \
    python "${SCRIPT_DIR}/train.py" \
        "${CFG}" \
        --work-dir "${WORK_DIR}" \
        --deterministic \
        "$@"

    EXIT_CODE=$?

} 2>&1 | tee "${LOG_FILE}"

# 注意：set -e 在管道中只对左侧生效，需要单独捕获退出码
EXIT_CODE=${PIPESTATUS[0]}

# ---------- 训练结束摘要 ----------
{
    echo ""
    echo "=========================================="
    if [ "${EXIT_CODE}" -eq 0 ]; then
        echo "  ✅ 训练正常结束"
    else
        echo "  ❌ 训练异常终止 (exit code: ${EXIT_CODE})"
        echo "  >> 查看末尾 50 行找原因: tail -50 ${LOG_FILE}"
    fi
    echo "  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""

    # ---------- 自动提取 loss 摘要 ----------
    echo "====== Loss 摘要（每个 Epoch 最后一条记录）======"
    # 提取包含 "Epoch [" 的行，每 Epoch 只保留最后一条（按 Epoch 号去重）
    grep -E "Epoch \[[0-9]+" "${LOG_FILE}" 2>/dev/null \
        | awk -F'Epoch \\[' '{print "Epoch ["$2}' \
        | awk -F'\\]\\[' '{epoch=$1; rest=$0; print epoch, rest}' \
        | sort -t'[' -k2 -n \
        | awk '
            BEGIN { last_epoch = -1 }
            {
                match($0, /Epoch \[([0-9]+)/, arr)
                epoch = arr[1] + 0
                if (epoch != last_epoch) {
                    if (last_epoch != -1) print last_line
                    last_epoch = epoch
                }
                last_line = $0
            }
            END { if (last_epoch != -1) print last_line }
        ' \
        | grep -oE "Epoch \[[0-9/]+\].*loss: [0-9.]+" \
        | head -30 \
        || echo "（未找到 loss 记录，可能训练未产生有效日志）"

    echo ""
    echo "====== planning.loss_flow_matching 趋势 ======"
    grep -oE "planning\.loss_flow_matching: [0-9.]+" "${LOG_FILE}" 2>/dev/null \
        | awk -F': ' 'NR % 5 == 1 {printf "iter %-4d → %s\n", NR, $2}' \
        | head -40 \
        || echo "（未找到 flow_matching loss 记录）"

    echo ""
    echo "====== grad_norm 趋势（异常值检测）======"
    grep -oE "grad_norm: [0-9.nai]+" "${LOG_FILE}" 2>/dev/null \
        | awk -F': ' '{
            v = $2
            if (v == "nan" || v+0 > 500) flag="⚠️  异常"
            else flag=""
            printf "iter %-4d  grad_norm: %-10s %s\n", NR, v, flag
        }' \
        | head -40 \
        || echo "（未找到 grad_norm 记录）"

    echo ""
    echo "====== 错误/警告行（前 20 条）======"
    grep -iE "error|traceback|exception|cuda out of memory|nan|inf" "${LOG_FILE}" 2>/dev/null \
        | grep -v "^2[0-9].*INFO" \
        | head -20 \
        || echo "（未发现明显错误）"

    echo ""
    echo "日志完整路径: ${LOG_FILE}"
    echo "====== 摘要结束 ======"

} | tee -a "${LOG_FILE}"

exit "${EXIT_CODE}"