# -*- coding: utf-8 -*-
"""
NaviLinkActions 第一版离散标签（数据读取 + 最近 MAIN_SIDE / DEDICATED 标量供 path encoder 对比实验）。

规则摘要（每条 LinkAction 的 assist + main）：
- focus: 0=NONE(占位), 1=MAIN_SIDE, 2=DEDICATED_TURN, 4=OTHER
- ms_kind: 仅 focus==1 时为 1=向左道路语义, 2=向右道路语义；否则 0
- ded_kind: 仅 focus==2 时为 1=左转专用道, 2=右转专用道；否则 0

assist:
  9,10,18 -> DEDICATED（10->ded1, 9/18->ded2）
  6 -> OTHER
  8 -> MAIN_SIDE ms_kind=1；7 -> MAIN_SIDE ms_kind=2
  1,15,17 -> MAIN_SIDE ms_kind=1；2,14,16 -> MAIN_SIDE ms_kind=2
  23,24,25,129 -> OTHER（沿主/沿辅/沿当前不再单独建 ms_kind）
  assist==0: main 在「左族」-> MAIN_SIDE ms_kind=1；「右族」-> ms_kind=2；否则 OTHER
    左族 main: TurnLeft(1), SlightLeft(3), TurnHardLeft(5), MergeLeft(9)
    右族 main: TurnRight(2), SlightRight(4), TurnHardRight(6), MergeRight(10)
其余 assist -> OTHER
"""

from __future__ import annotations

import numpy as np

FOCUS_NONE = 0
FOCUS_MAIN_SIDE = 1
FOCUS_DEDICATED = 2
FOCUS_OTHER = 4

_MS_LEFT = 1
_MS_RIGHT = 2

_DED_LEFT = 1
_DED_RIGHT = 2

# main_action_type：与 proto MainAction 数值一致
_MAIN_LEFT = frozenset({1, 3, 5, 9})
_MAIN_RIGHT = frozenset({2, 4, 6, 10})


def compute_navi_link_ms_nearest_numpy(
    enc_focus: np.ndarray,
    enc_ms_kind: np.ndarray,
    dist: np.ndarray,
    valid_mask: np.ndarray,
):
    """
    在 200m 已截断的槽位中，取 focus==MAIN_SIDE 且 ms_kind∈{1,2} 的最近一条，返回单元素向量供 batch 维 (B,1) 使用。
    无匹配时 valid=0, dist=0, kind=0。
    """
    enc_focus = np.asarray(enc_focus, dtype=np.int32).flatten()
    enc_ms_kind = np.asarray(enc_ms_kind, dtype=np.int32).flatten()
    dist = np.asarray(dist, dtype=np.float32).flatten()
    valid_mask = np.asarray(valid_mask, dtype=np.bool_).flatten()
    n = min(enc_focus.size, enc_ms_kind.size, dist.size, valid_mask.size)
    best_i = -1
    best_d = np.inf
    for i in range(n):
        if not valid_mask[i]:
            continue
        if int(enc_focus[i]) != FOCUS_MAIN_SIDE:
            continue
        mk = int(enc_ms_kind[i])
        if mk not in (1, 2):
            continue
        di = float(dist[i])
        if di < best_d:
            best_d = di
            best_i = i
    if best_i < 0:
        return (
            np.array([0.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            np.array([0], dtype=np.int64),
        )
    return (
        np.array([1.0], dtype=np.float32),
        np.array([float(dist[best_i])], dtype=np.float32),
        np.array([int(enc_ms_kind[best_i])], dtype=np.int64),
    )


def compute_navi_link_ded_nearest_numpy(
    enc_focus: np.ndarray,
    enc_ded_kind: np.ndarray,
    dist: np.ndarray,
    valid_mask: np.ndarray,
):
    """
    在 200m 已截断的槽位中，取 focus==DEDICATED 且 ded_kind∈{1,2} 的最近一条，返回单元素向量供 batch 维 (B,1) 使用。
    逻辑与 compute_navi_link_ms_nearest_numpy 对称；kind 为 ded_kind（1=左转专用道，2=右转专用道）。
    无匹配时 valid=0, dist=0, kind=0。
    """
    enc_focus = np.asarray(enc_focus, dtype=np.int32).flatten()
    enc_ded_kind = np.asarray(enc_ded_kind, dtype=np.int32).flatten()
    dist = np.asarray(dist, dtype=np.float32).flatten()
    valid_mask = np.asarray(valid_mask, dtype=np.bool_).flatten()
    n = min(enc_focus.size, enc_ded_kind.size, dist.size, valid_mask.size)
    best_i = -1
    best_d = np.inf
    for i in range(n):
        if not valid_mask[i]:
            continue
        if int(enc_focus[i]) != FOCUS_DEDICATED:
            continue
        dk = int(enc_ded_kind[i])
        if dk not in (1, 2):
            continue
        di = float(dist[i])
        if di < best_d:
            best_d = di
            best_i = i
    if best_i < 0:
        return (
            np.array([0.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            np.array([0], dtype=np.int64),
        )
    return (
        np.array([1.0], dtype=np.float32),
        np.array([float(dist[best_i])], dtype=np.float32),
        np.array([int(enc_ded_kind[best_i])], dtype=np.int64),
    )


def encode_navi_link_action_v1(assist: int, main: int) -> tuple[int, int, int]:
    """返回 (focus, ms_kind, ded_kind)。"""
    a = int(assist)
    m = int(main)

    if a in (9, 10, 18):
        ded = _DED_LEFT if a == 10 else _DED_RIGHT
        return (FOCUS_DEDICATED, 0, ded)

    if a == 6:
        return (FOCUS_OTHER, 0, 0)

    if a == 8:
        return (FOCUS_MAIN_SIDE, _MS_LEFT, 0)
    if a == 7:
        return (FOCUS_MAIN_SIDE, _MS_RIGHT, 0)

    if a in (1, 15, 17):
        return (FOCUS_MAIN_SIDE, _MS_LEFT, 0)
    if a in (2, 14, 16):
        return (FOCUS_MAIN_SIDE, _MS_RIGHT, 0)

    if a in (23, 24, 25, 129):
        return (FOCUS_OTHER, 0, 0)

    if a == 0:
        if m in _MAIN_LEFT:
            return (FOCUS_MAIN_SIDE, _MS_LEFT, 0)
        if m in _MAIN_RIGHT:
            return (FOCUS_MAIN_SIDE, _MS_RIGHT, 0)
        return (FOCUS_OTHER, 0, 0)

    return (FOCUS_OTHER, 0, 0)
