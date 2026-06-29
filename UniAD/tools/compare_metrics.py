"""
compare_metrics.py
==================
用途：对比 UniAD 原版 vs DiffusionPlanningHead 的规划指标。

使用方法：
  python tools/compare_metrics.py \
      --uniad_pkl /home/guo/VLA/UniAD/results/e2e_results.pkl \
      --dit_pkl   /home/guo/UniAD-Diffusion/UniAD-Diffusion/UniAD/output/vis_results.pkl \
      --out_csv   output/compare.csv
"""

import argparse
import pickle
import numpy as np
import os
import csv
import torch


# ─────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────

def to_numpy(x):
    """
    将任意输入安全转为 numpy array，兼容：
      - torch.Tensor（CPU 或 GPU）
      - list / tuple 内部嵌套 torch.Tensor（如 [cuda_tensor]）
      - numpy.ndarray、Python list of numbers
      - None
    """
    if x is None:
        return None
    # 直接是 Tensor
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    # list 或 tuple：可能内部嵌套 Tensor，先递归展开再 stack
    if isinstance(x, (list, tuple)):
        converted = [to_numpy(item) for item in x]
        return np.array(converted)
    # numpy array 或其他数值类型
    return np.array(x)


def parse_single_result(result):
    """
    从单个样本 dict 中提取预测轨迹和 GT 轨迹。

    返回：
        pred : (T, 2)  预测轨迹 (x, y)
        gt   : (T, 2)  GT 轨迹 (x, y)
    兼容的 shape：(1,T,2) / (T,2) / (1,1,T,2)，会自动去掉多余的前置维度。
    GT 可能是 (T,3)，包含 heading，只取前两列。
    """
    pred = to_numpy(result.get('planning_traj', None))
    gt   = to_numpy(result.get('planning_traj_gt', None))

    if pred is None or gt is None:
        return None, None

    # 去掉多余的 batch / mode 维度，直到剩 (T, D)
    while pred.ndim > 2:
        pred = pred[0]
    while gt.ndim > 2:
        gt = gt[0]

    # GT 可能含 heading 列，只保留 x, y
    gt = gt[:, :2]

    return pred, gt


def compute_l2_per_step(pred, gt):
    """
    计算每个时间步的欧氏距离（L2 误差）。
    pred, gt : (T, 2)
    返回     : (T,)
    """
    return np.sqrt(((pred[:, :2] - gt[:, :2]) ** 2).sum(axis=-1))


# ─────────────────────────────────────────────────────────────────────
# 读取 pkl，计算 L2
# ─────────────────────────────────────────────────────────────────────

def compute_metrics_from_pkl(pkl_path, n_future=6):
    """
    读取 pkl，对每个样本计算 L2，返回 (N, T) ndarray。
    支持格式：
      - dict(bbox_results=[sample_dict, ...])   ← test.py 标准输出
      - list([sample_dict, ...])
    """
    print(f"\n读取: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict) and 'bbox_results' in data:
        results = data['bbox_results']
    elif isinstance(data, list):
        results = data
    else:
        raise ValueError(f"无法解析 pkl 顶层格式，type={type(data)}, "
                         f"keys={list(data.keys())[:6] if isinstance(data, dict) else 'N/A'}")

    print(f"  总样本数: {len(results)}")

    l2_list = []
    skipped = 0

    for r in results:
        if not isinstance(r, dict):
            skipped += 1
            continue

        pred, gt = parse_single_result(r)
        if pred is None or gt is None:
            skipped += 1
            continue

        T = min(pred.shape[0], gt.shape[0], n_future)
        l2 = compute_l2_per_step(pred[:T], gt[:T])

        # 步数不足时用最后一步填充
        if len(l2) < n_future:
            pad = np.full(n_future - len(l2), l2[-1] if len(l2) > 0 else 0.0)
            l2 = np.concatenate([l2, pad])

        l2_list.append(l2[:n_future])

    if skipped > 0:
        print(f"  跳过无规划字段样本: {skipped}")

    if len(l2_list) == 0:
        raise RuntimeError(
            f"pkl 中没有找到任何有效的 planning_traj / planning_traj_gt 字段！\n"
            f"路径: {pkl_path}"
        )

    print(f"  有效规划样本: {len(l2_list)}")
    return np.stack(l2_list, axis=0)   # (N, T)


# ─────────────────────────────────────────────────────────────────────
# 打印对比表格
# ─────────────────────────────────────────────────────────────────────

def print_table(uniad_l2, dit_l2, out_csv=None):
    """
    nuScenes 规划频率 2Hz（每步 0.5s），6步=3秒。
    对比 1s / 2s / 3s 的 L2 误差均值（越小越好）。
    """
    # 时间步索引（0-indexed）
    time_steps = [
        ('1s', 1),   # step index 1 → 0.5×2 = 1.0s
        ('2s', 3),   # step index 3 → 0.5×4 = 2.0s
        ('3s', 5),   # step index 5 → 0.5×6 = 3.0s
    ]

    sep    = '─' * 70
    header = f"{'指标':<20} {'UniAD原版':>10} {'DiT版本':>10} {'变化':>10} {'优化幅度':>12}"

    print(f"\n{sep}")
    print(" 📊 规划指标对比：UniAD 原版 vs DiffusionPlanningHead（DiT）")
    print(sep)
    print(header)
    print(sep)
    print(" L2 误差（单位：米，越小越好）")

    rows = []

    for label, idx in time_steps:
        if idx >= uniad_l2.shape[1] or idx >= dit_l2.shape[1]:
            continue
        u_val  = float(uniad_l2[:, idx].mean())
        d_val  = float(dit_l2[:, idx].mean())
        delta  = d_val - u_val
        pct    = delta / (u_val + 1e-8) * 100
        sign   = '▼' if delta < 0 else '▲'
        flag   = '✅' if delta < 0 else '❌'
        print(f"  L2_{label:<16} {u_val:>10.4f} {d_val:>10.4f} {delta:>+10.4f}  {sign}{abs(pct):>6.1f}%  {flag}")
        rows.append({
            '指标': f'L2_{label}',
            'UniAD': round(u_val, 4),
            'DiT':   round(d_val, 4),
            '变化':  round(delta, 4),
            '幅度%': round(pct, 1),
        })

    # 平均
    valid_idx = [idx for _, idx in time_steps
                 if idx < uniad_l2.shape[1] and idx < dit_l2.shape[1]]
    u_avg  = float(uniad_l2[:, valid_idx].mean())
    d_avg  = float(dit_l2[:,   valid_idx].mean())
    delta  = d_avg - u_avg
    pct    = delta / (u_avg + 1e-8) * 100
    sign   = '▼' if delta < 0 else '▲'
    flag   = '✅' if delta < 0 else '❌'

    print(sep)
    print(f"  {'L2_avg (1s/2s/3s)':<18} {u_avg:>10.4f} {d_avg:>10.4f} {delta:>+10.4f}  {sign}{abs(pct):>6.1f}%  {flag}")
    rows.append({
        '指标': 'L2_avg',
        'UniAD': round(u_avg, 4),
        'DiT':   round(d_avg, 4),
        '变化':  round(delta, 4),
        '幅度%': round(pct, 1),
    })

    print(sep)
    print()
    print("  说明：")
    print("    ▼ ✅ = DiT 版本 L2 误差更低（更优）")
    print("    ▲ ❌ = DiT 版本 L2 误差更高（更差）")
    print("    nuScenes 规划 2Hz，共 6 步 × 0.5s = 3 秒预测窗口")
    print()
    print(f"  UniAD 原版有效样本数 : {uniad_l2.shape[0]}")
    print(f"  DiT  版本有效样本数  : {dit_l2.shape[0]}")
    print(sep + "\n")

    # 保存 CSV
    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['指标', 'UniAD', 'DiT', '变化', '幅度%'])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  CSV 已保存至: {out_csv}\n")


# ─────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='对比 UniAD vs DiT 规划 L2 指标')
    parser.add_argument('--uniad_pkl', required=True,
                        help='UniAD 原版 test.py 生成的 pkl 路径')
    parser.add_argument('--dit_pkl',   required=True,
                        help='DiT 版本 test.py 生成的 pkl 路径')
    parser.add_argument('--out_csv',   default=None,
                        help='（可选）对比结果 CSV 保存路径')
    parser.add_argument('--n_future',  type=int, default=6,
                        help='规划步数（默认 6）')
    args = parser.parse_args()

    for p in [args.uniad_pkl, args.dit_pkl]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"找不到文件: {p}")

    uniad_l2 = compute_metrics_from_pkl(args.uniad_pkl, n_future=args.n_future)
    dit_l2   = compute_metrics_from_pkl(args.dit_pkl,   n_future=args.n_future)

    print_table(uniad_l2, dit_l2, out_csv=args.out_csv)


if __name__ == '__main__':
    main()