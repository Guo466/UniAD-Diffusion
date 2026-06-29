"""
compare_metrics.py
==================
用途：对比 UniAD 原版 vs DiffusionPlanningHead 的规划指标。

使用方法：
  python tools/compare_metrics.py \
      --uniad_pkl  output/uniad_results.pkl \
      --dit_pkl    output/dit_results.pkl \
      [--out_csv   output/compare.csv]

输入：两个由 test.py 生成的 .pkl 文件，每个文件包含：
  - bbox_results: list of dict，每个 dict 含：
      planning_traj     (n_future, 2) 或 (1, n_future, 2)  预测轨迹
      planning_traj_gt  (1, n_future, 3)                  GT 轨迹
      command           int 或 list                        驾驶命令

输出：
  - 终端打印对比表格
  - （可选）保存为 CSV 文件
"""

import argparse
import pickle
import numpy as np
import os


# ─────────────────────────────────────────────────────
# 指标计算（与 planning_metrics.py 逻辑对齐）
# ─────────────────────────────────────────────────────

def compute_l2(pred_traj, gt_traj):
    """
    计算每个时间步的 L2 误差。
    pred_traj : (T, 2)  预测轨迹（x, y）
    gt_traj   : (T, 2)  GT 轨迹（x, y）
    返回      : (T,)    每步的欧氏距离
    """
    return np.sqrt(((pred_traj[:, :2] - gt_traj[:, :2]) ** 2).sum(axis=-1))


def to_numpy(x):
    """安全地将 tensor/array/list 转为 numpy，兼容 GPU tensor。"""
    if x is None:
        return None
    if hasattr(x, 'cpu'):          # torch.Tensor（包括 cuda tensor）
        x = x.detach().cpu()
    if hasattr(x, 'numpy'):        # torch.Tensor on CPU
        x = x.numpy()
    return np.array(x)


def parse_single_result(result):
    """
    从单个推理结果 dict 中提取：
      pred_traj : (T, 2)
      gt_traj   : (T, 2)
    兼容：
      - GPU tensor (cuda:0)：自动 .cpu()
      - shape (1,T,2) / (T,2) / (1,1,T,2)：自动去掉多余维度
      - planning_traj_gt 可能是 (1,T,3)，只取前 2 列 (x,y)
    """
    pred = to_numpy(result.get('planning_traj', None))
    gt   = to_numpy(result.get('planning_traj_gt', None))

    if pred is None or gt is None:
        return None, None

    # 去掉多余的 batch/mode 维度，直到剩 (T, D)
    while pred.ndim > 2:
        pred = pred[0]
    while gt.ndim > 2:
        gt = gt[0]

    # gt 可能是 (T, 3)，包含 heading；只取前 2 列 (x, y)
    gt = gt[:, :2]

    return pred, gt


def compute_metrics_from_pkl(pkl_path, n_future=6):
    """
    读取 pkl 文件，计算所有样本的 L2（每时间步）。
    返回：
      l2_per_step : (N, T) numpy array，N=样本数，T=时间步数
    """
    print(f"\n读取: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # 兼容三种格式：
    #   格式A: dict(bbox_results=[sample_dict, ...])  ← test.py 标准输出
    #   格式B: list([sample_dict, ...])               ← 部分脚本直接输出列表
    #   格式C: dict(token=..., planning_traj=...)     ← 单样本 dict（罕见）
    if isinstance(data, dict) and 'bbox_results' in data:
        results = data['bbox_results']
    elif isinstance(data, list):
        results = data
    elif isinstance(data, dict) and 'planning_traj' in data:
        # 顶层就是单个样本 dict，包装成列表
        results = [data]
    elif isinstance(data, dict):
        # 顶层是 dict 但没有 bbox_results，尝试把所有 value 是 list 的拼起来
        # vis_results.pkl 可能是 {token: sample_dict} 的字典
        candidate = []
        for v in data.values():
            if isinstance(v, list):
                candidate.extend(v)
            elif isinstance(v, dict):
                candidate.append(v)
        if candidate:
            results = candidate
        else:
            raise ValueError(f"无法解析 pkl 格式，顶层 type=dict, keys={list(data.keys())[:8]}")
    else:
        raise ValueError(f"无法解析 pkl 格式，顶层 type={type(data)}")

    print(f"  总样本数: {len(results)}")

    l2_list = []
    skip = 0
    for r in results:
        if not isinstance(r, dict):
            skip += 1
            continue
        pred, gt = parse_single_result(r)
        if pred is None or gt is None:
            skip += 1
            continue
        # 对齐时间步数
        T = min(pred.shape[0], gt.shape[0], n_future)
        l2 = compute_l2(pred[:T], gt[:T])
        # 如果步数不足 n_future，用最后一步的值填充（保守估计）
        if len(l2) < n_future:
            pad = np.full(n_future - len(l2), l2[-1] if len(l2) > 0 else 0.0)
            l2 = np.concatenate([l2, pad])
        l2_list.append(l2[:n_future])

    if skip > 0:
        print(f"  跳过无规划轨迹样本: {skip}")
    if len(l2_list) == 0:
        raise RuntimeError(
            f"pkl 中没有找到有效的 planning_traj 字段！\n"
            f"请检查 test.py 的 custom_single_gpu_test 是否保留了 planning_traj 字段，\n"
            f"以及 pkl 是否由 DiT/UniAD 的 planning 配置生成。"
        )

    print(f"  有效规划样本: {len(l2_list)}")
    return np.stack(l2_list, axis=0)   # (N, T)


def print_table(uniad_l2, dit_l2, out_csv=None):
    """
    打印对比表格，时间步粒度：1s(step2), 2s(step4), 3s(step6)
    nuScenes 规划频率 2Hz，即每步 0.5s
    """
    # 对应时间步索引（0-indexed）：1s=step1(idx1), 2s=step3(idx3), 3s=step5(idx5)
    time_steps = [
        ('1s', 1),   # 第 2 步（0.5×2=1.0s）
        ('2s', 3),   # 第 4 步（0.5×4=2.0s）
        ('3s', 5),   # 第 6 步（0.5×6=3.0s）
    ]

    # ─── 表头 ───
    sep = "─" * 68
    header = f"{'指标':<20} {'UniAD原版':>12} {'DiT版本':>12} {'变化':>12} {'优化幅度':>10}"
    print("\n" + sep)
    print(" 📊 规划指标对比：UniAD 原版 vs DiffusionPlanningHead（DiT）")
    print(sep)
    print(header)
    print(sep)

    rows = []  # 用于保存 CSV

    # ─── L2 各时间步 ───
    print(" L2 误差（米，越小越好）")
    for label, idx in time_steps:
        if idx >= uniad_l2.shape[1] or idx >= dit_l2.shape[1]:
            continue
        u_val = uniad_l2[:, idx].mean()
        d_val = dit_l2[:, idx].mean()
        delta  = d_val - u_val
        pct    = (d_val - u_val) / (u_val + 1e-8) * 100
        sign   = "▼" if delta < 0 else "▲"
        flag   = "✅" if delta < 0 else "❌"
        print(f"  L2_{label:<16} {u_val:>12.4f} {d_val:>12.4f} {delta:>+12.4f} {sign}{pct:>8.1f}% {flag}")
        rows.append({'指标': f'L2_{label}', 'UniAD': u_val, 'DiT': d_val, '变化': delta, '幅度%': pct})

    # ─── L2 平均 ───
    print(sep)
    u_avg = uniad_l2[:, [idx for _, idx in time_steps if idx < uniad_l2.shape[1]]].mean()
    d_avg = dit_l2[:,   [idx for _, idx in time_steps if idx < dit_l2.shape[1]]].mean()
    delta  = d_avg - u_avg
    pct    = (d_avg - u_avg) / (u_avg + 1e-8) * 100
    sign   = "▼" if delta < 0 else "▲"
    flag   = "✅" if delta < 0 else "❌"
    print(f"  {'L2_avg (1/2/3s)':<18} {u_avg:>12.4f} {d_avg:>12.4f} {delta:>+12.4f} {sign}{pct:>8.1f}% {flag}")
    rows.append({'指标': 'L2_avg', 'UniAD': u_avg, 'DiT': d_avg, '变化': delta, '幅度%': pct})

    print(sep)
    print()
    print(" 说明：")
    print("   ▼ = DiT 版本指标更优（L2 下降）")
    print("   ▲ = DiT 版本指标更差（L2 上升）")
    print("   nuScenes 规划频率 2Hz，每步 0.5s，共 6 步 = 3 秒规划窗口")
    print()
    print(f"  UniAD 原版有效样本数: {uniad_l2.shape[0]}")
    print(f"  DiT 版本有效样本数:   {dit_l2.shape[0]}")
    print(sep + "\n")

    # ─── 可选 CSV 输出 ───
    if out_csv:
        import csv
        os.makedirs(os.path.dirname(out_csv) if os.path.dirname(out_csv) else '.', exist_ok=True)
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['指标', 'UniAD', 'DiT', '变化', '幅度%'])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  CSV 已保存至: {out_csv}")


def main():
    parser = argparse.ArgumentParser(description='对比 UniAD vs DiT 规划指标')
    parser.add_argument('--uniad_pkl', required=True,
                        help='UniAD 原版 test.py 生成的 pkl 文件路径')
    parser.add_argument('--dit_pkl',   required=True,
                        help='DiT 版本 test.py 生成的 pkl 文件路径')
    parser.add_argument('--out_csv',   default=None,
                        help='（可选）对比结果保存为 CSV 的路径，如 output/compare.csv')
    parser.add_argument('--n_future',  type=int, default=6,
                        help='规划步数（默认 6，对应 3 秒）')
    args = parser.parse_args()

    # 检查文件存在
    for p in [args.uniad_pkl, args.dit_pkl]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"找不到文件: {p}\n请检查路径或先在笔记本上运行 test.py")

    uniad_l2 = compute_metrics_from_pkl(args.uniad_pkl, n_future=args.n_future)
    dit_l2   = compute_metrics_from_pkl(args.dit_pkl,   n_future=args.n_future)

    print_table(uniad_l2, dit_l2, out_csv=args.out_csv)


if __name__ == '__main__':
    main()