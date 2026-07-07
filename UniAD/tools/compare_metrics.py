"""
compare_metrics.py
==================
用途：对比 UniAD 原版 vs DiffusionPlanningHead（DiT）的规划指标。

对比指标：
  - L2 误差（米，越小越好）：1s / 2s / 3s
  - obj_col     （占用图碰撞率，越小越好）：1s / 2s / 3s
  - obj_box_col （3D 框碰撞率，越小越好）  ：1s / 2s / 3s

数据来源优先级：
  1) 优先读取 pkl 顶层的 `planning_results_computed`（由 tools/test.py 在评测时
     用 PlanningMetric 在线累积算出，包含 L2 + obj_col + obj_box_col，与官方
     UniAD 评测协议完全一致，是最权威的数据来源）。
  2) 若 pkl 中没有该字段（比如很旧版本生成的 pkl），回退为从
     `bbox_results` 里逐样本用 planning_traj / planning_traj_gt 重新计算 L2
     （此时无法计算碰撞率，因为 segmentation 字段在保存 pkl 时已被清理）。

使用方法：
  python tools/compare_metrics.py \
      --uniad_pkl output/uniad_mini_results.pkl \
      --dit_pkl   output/dit_results_steps5.pkl \
      --out_csv   output/compare.csv

后续在服务器上用完整数据集重新训练 UniAD 原版 / DiT 后，
只需把 --uniad_pkl / --dit_pkl 换成新的 pkl 路径，即可复用本脚本得到同样格式的对比表。
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
# 读取 pkl，提取指标
# ─────────────────────────────────────────────────────────────────────

def load_pkl(pkl_path):
    print(f"\n读取: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def get_metrics_from_planning_results_computed(data, n_future=6):
    """
    优先路径：直接读取 tools/test.py 在评测时在线累积好的
    ret['planning_results_computed']，格式为：
        {'obj_col': Tensor(n_future,), 'obj_box_col': Tensor(n_future,), 'L2': Tensor(n_future,)}
    返回 dict(L2=(T,), obj_col=(T,), obj_box_col=(T,)) 的 numpy 数组，或 None（找不到时）。
    """
    if not isinstance(data, dict) or 'planning_results_computed' not in data:
        return None

    pm = data['planning_results_computed']
    L2          = to_numpy(pm['L2'])[:n_future]
    obj_col     = to_numpy(pm['obj_col'])[:n_future]
    obj_box_col = to_numpy(pm['obj_box_col'])[:n_future]

    n_samples = None
    if isinstance(data.get('bbox_results', None), list):
        n_samples = len(data['bbox_results'])

    print(f"  [快速路径] 使用 pkl 中预先算好的 planning_results_computed")
    if n_samples is not None:
        print(f"  总样本数: {n_samples}")

    return dict(L2=L2, obj_col=obj_col, obj_box_col=obj_box_col, n_samples=n_samples)


def compute_l2_from_bbox_results(data, n_future=6):
    """
    回退路径：从 bbox_results 中逐样本用 planning_traj / planning_traj_gt
    重新计算 L2（无法计算碰撞率，因为 segmentation 字段已在保存 pkl 前被清理）。
    返回 (N, T) ndarray。
    """
    if isinstance(data, dict) and 'bbox_results' in data:
        results = data['bbox_results']
    elif isinstance(data, list):
        results = data
    else:
        raise ValueError(f"无法解析 pkl 顶层格式，type={type(data)}, "
                         f"keys={list(data.keys())[:6] if isinstance(data, dict) else 'N/A'}")

    print(f"  [回退路径] 从 bbox_results 逐样本重新计算 L2（无法计算碰撞率）")
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
            f"pkl 中没有找到任何有效的 planning_traj / planning_traj_gt 字段！"
        )

    print(f"  有效规划样本: {len(l2_list)}")
    return np.stack(l2_list, axis=0)   # (N, T)


def compute_metrics_from_pkl(pkl_path, n_future=6):
    """
    统一入口：优先使用 planning_results_computed，否则回退到逐样本重算 L2。
    返回 dict(L2=(T,), obj_col=(T,) or None, obj_box_col=(T,) or None, per_sample_l2=(N,T) or None)
    """
    data = load_pkl(pkl_path)

    fast = get_metrics_from_planning_results_computed(data, n_future=n_future)
    if fast is not None:
        return dict(
            L2=fast['L2'],
            obj_col=fast['obj_col'],
            obj_box_col=fast['obj_box_col'],
            per_sample_l2=None,
            n_samples=fast['n_samples'],
        )

    # 回退路径：只有逐样本 L2，没有碰撞率
    per_sample_l2 = compute_l2_from_bbox_results(data, n_future=n_future)
    return dict(
        L2=per_sample_l2.mean(axis=0),
        obj_col=None,
        obj_box_col=None,
        per_sample_l2=per_sample_l2,
        n_samples=per_sample_l2.shape[0],
    )


# ─────────────────────────────────────────────────────────────────────
# 打印对比表格
# ─────────────────────────────────────────────────────────────────────

def _print_metric_block(title, unit_hint, uniad_arr, dit_arr, time_steps, rows, lower_is_better=True):
    """打印单个指标（L2 / obj_col / obj_box_col）在 1s/2s/3s 的对比，并追加到 rows。"""
    sep = '─' * 70
    print(f" {title}（{unit_hint}）")

    for label, idx in time_steps:
        if idx >= len(uniad_arr) or idx >= len(dit_arr):
            continue
        u_val = float(uniad_arr[idx])
        d_val = float(dit_arr[idx])
        delta = d_val - u_val
        pct = delta / (abs(u_val) + 1e-8) * 100
        is_better = (delta < 0) if lower_is_better else (delta > 0)
        sign = '▼' if delta < 0 else '▲'
        flag = '✅' if is_better else '❌'
        print(f"  {title}_{label:<12} {u_val:>10.4f} {d_val:>10.4f} {delta:>+10.4f}  {sign}{abs(pct):>6.1f}%  {flag}")
        rows.append({
            '指标': f'{title}_{label}',
            'UniAD': round(u_val, 4),
            'DiT': round(d_val, 4),
            '变化': round(delta, 4),
            '幅度%': round(pct, 1),
        })

    valid_idx = [idx for _, idx in time_steps if idx < len(uniad_arr) and idx < len(dit_arr)]
    if valid_idx:
        u_avg = float(np.mean([uniad_arr[i] for i in valid_idx]))
        d_avg = float(np.mean([dit_arr[i] for i in valid_idx]))
        delta = d_avg - u_avg
        pct = delta / (abs(u_avg) + 1e-8) * 100
        is_better = (delta < 0) if lower_is_better else (delta > 0)
        sign = '▼' if delta < 0 else '▲'
        flag = '✅' if is_better else '❌'
        print(f"  {sep}")
        print(f"  {title}_avg{'':<11} {u_avg:>10.4f} {d_avg:>10.4f} {delta:>+10.4f}  {sign}{abs(pct):>6.1f}%  {flag}")
        rows.append({
            '指标': f'{title}_avg',
            'UniAD': round(u_avg, 4),
            'DiT': round(d_avg, 4),
            '变化': round(delta, 4),
            '幅度%': round(pct, 1),
        })
    print()


def print_table(uniad_metrics, dit_metrics, out_csv=None):
    """
    nuScenes 规划频率 2Hz（每步 0.5s），6步=3秒。
    对比 1s / 2s / 3s 的 L2 误差、碰撞率（若可用）。
    """
    time_steps = [
        ('1s', 1),   # step index 1 → 0.5×2 = 1.0s
        ('2s', 3),   # step index 3 → 0.5×4 = 2.0s
        ('3s', 5),   # step index 5 → 0.5×6 = 3.0s
    ]

    sep = '─' * 70
    header = f"{'指标':<16} {'UniAD原版':>10} {'DiT版本':>10} {'变化':>10} {'优化幅度':>12}"

    print(f"\n{sep}")
    print(" 📊 规划指标对比：UniAD 原版 vs DiffusionPlanningHead（DiT）")
    print(sep)
    print(header)
    print(sep)

    rows = []

    # ---- L2 ----
    _print_metric_block('L2', '单位：米，越小越好', uniad_metrics['L2'], dit_metrics['L2'],
                         time_steps, rows, lower_is_better=True)

    # ---- 碰撞率（若两边都有）----
    has_coll = (uniad_metrics['obj_col'] is not None and dit_metrics['obj_col'] is not None)
    if has_coll:
        _print_metric_block('obj_col', '占用图碰撞率，越小越好',
                             uniad_metrics['obj_col'], dit_metrics['obj_col'],
                             time_steps, rows, lower_is_better=True)
        _print_metric_block('obj_box_col', '3D框碰撞率，越小越好',
                             uniad_metrics['obj_box_col'], dit_metrics['obj_box_col'],
                             time_steps, rows, lower_is_better=True)
    else:
        print(" ⚠️  碰撞率（obj_col / obj_box_col）不可用：")
        print("     至少一个 pkl 中没有 planning_results_computed 字段（走的是回退路径），")
        print("     该字段只能在 tools/test.py 评测时在线累积生成，无法事后从 bbox_results 重新计算。")
        print("     如需碰撞率对比，请确认两个 pkl 都是用最新版 tools/test.py 生成的。\n")

    print(sep)
    print("  说明：")
    print("    ▼ ✅ = DiT 版本更优（误差/碰撞率更低）")
    print("    ▲ ❌ = DiT 版本更差（误差/碰撞率更高）")
    print("    nuScenes 规划 2Hz，共 6 步 × 0.5s = 3 秒预测窗口")
    print()
    if uniad_metrics.get('n_samples') is not None:
        print(f"  UniAD 原版有效样本数 : {uniad_metrics['n_samples']}")
    if dit_metrics.get('n_samples') is not None:
        print(f"  DiT  版本有效样本数  : {dit_metrics['n_samples']}")
    print(sep + "\n")

    # 保存 CSV
    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or '.', exist_ok=True)
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['指标', 'UniAD', 'DiT', '变化', '幅度%'])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  CSV 已保存至: {out_csv}\n")


# ─────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='对比 UniAD vs DiT 规划指标（L2 + 碰撞率）')
    parser.add_argument('--uniad_pkl', required=True,
                        help='UniAD 原版 tools/test.py 生成的 pkl 路径')
    parser.add_argument('--dit_pkl',   required=True,
                        help='DiT 版本 tools/test.py 生成的 pkl 路径')
    parser.add_argument('--out_csv',   default=None,
                        help='（可选）对比结果 CSV 保存路径')
    parser.add_argument('--n_future',  type=int, default=6,
                        help='规划步数（默认 6）')
    args = parser.parse_args()

    for p in [args.uniad_pkl, args.dit_pkl]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"找不到文件: {p}")

    uniad_metrics = compute_metrics_from_pkl(args.uniad_pkl, n_future=args.n_future)
    dit_metrics   = compute_metrics_from_pkl(args.dit_pkl,   n_future=args.n_future)

    print_table(uniad_metrics, dit_metrics, out_csv=args.out_csv)


if __name__ == '__main__':
    main()