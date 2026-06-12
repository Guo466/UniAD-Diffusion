"""
参考 pl_diffusion_models_nomap 中 recommend_lane_navitopo_aug 的扩增逻辑：
当 navitopo 按 (0,0) 距离分组后组数 == 推荐车道数 且 > 1 时，
保留 1 条推荐车道及对应的 navitopo（用于验证单组输入的模型性能）。
"""
import torch
import math
import random

def _project_point_to_polyline(point_xy, polyline_xy):
    '''
        向量化：在 polyline 上投影点，返回:
        s_proj: 投影点对应弧长
        total_length: polyline 总长度
        cumulative_dist: 顶点累计弧长 [N]
        seg_len: 每段长度 [N-1]
    '''
    if polyline_xy is None or polyline_xy.dim() != 2 or polyline_xy.shape[0] < 2:
        return 0.0, 0.0, None, None

    starts = polyline_xy[:-1]                                    # (S, 2)
    v = polyline_xy[1:] - starts                                 # (S, 2)
    seg_len = torch.norm(v, dim=1)                               # (S,)
    cumulative_dist = torch.cat([
        torch.zeros(1, device=polyline_xy.device, dtype=polyline_xy.dtype),
        torch.cumsum(seg_len, dim=0),
    ], dim=0)
    total_length = float(cumulative_dist[-1].item())
    if total_length <= 1e-6:
        return 0.0, 0.0, cumulative_dist, seg_len

    denom = (v * v).sum(dim=1)                                   # (S,)
    w = point_xy.unsqueeze(0) - starts                           # (S, 2)
    t = ((w * v).sum(dim=1) / denom.clamp(min=1e-8)).clamp(0.0, 1.0)
    t = torch.where(denom <= 1e-8, torch.zeros_like(t), t)
    proj = starts + t.unsqueeze(1) * v                           # (S, 2)
    dist2 = ((point_xy.unsqueeze(0) - proj) ** 2).sum(dim=1)    # (S,)
    best = int(torch.argmin(dist2).item())
    s_proj = float(cumulative_dist[best].item()) + float(t[best].item()) * float(seg_len[best].item())
    return s_proj, total_length, cumulative_dist, seg_len

def _point_at_arc_length(polyline_xy, cumulative_dist, seg_len, s):
    '''
        - polyline_xy : 折线的点序列，形状 (N, 2)
        - cumulative_dist : 顶点累计弧长数组，形状 (N,) （记录从起点到每个顶点的总长度）
        - seg_len : 每段折线的长度数组，形状 (N-1,) （记录相邻顶点之间的距离）
        - s : 目标弧长位置（从折线起点到目标点的总长度）
    '''
    if (
        polyline_xy is None
        or cumulative_dist is None
        or seg_len is None
        or polyline_xy.shape[0] < 2
    ):
        return None
    s = float(max(0.0, min(s, float(cumulative_dist[-1].item()))))
    for i in range(seg_len.shape[0]):
        s0 = float(cumulative_dist[i].item())
        s1 = float(cumulative_dist[i + 1].item())
        if s <= s1 or i == seg_len.shape[0] - 1:
            denom = max(1e-6, s1 - s0)
            t = (s - s0) / denom
            return polyline_xy[i] + t * (polyline_xy[i + 1] - polyline_xy[i])
    return polyline_xy[-1]


def _batch_min_distance_to_segments(query_pts, seg_starts, seg_ends):
    """
    向量化：批量计算 Q 个查询点到 M 条线段的最小距离。
    query_pts: (Q, 2), seg_starts: (M, 2), seg_ends: (M, 2)
    返回: (Q,) 每个查询点到最近线段的距离
    """
    v = seg_ends - seg_starts                                          # (M, 2)
    denom = (v * v).sum(dim=1)                                         # (M,)
    w = query_pts.unsqueeze(1) - seg_starts.unsqueeze(0)               # (Q, M, 2)
    t = ((w * v.unsqueeze(0)).sum(dim=2) / denom.unsqueeze(0).clamp(min=1e-8)).clamp(0.0, 1.0)
    t = torch.where(denom.unsqueeze(0) <= 1e-8, torch.zeros_like(t), t)  # (Q, M)
    proj = seg_starts.unsqueeze(0) + t.unsqueeze(2) * v.unsqueeze(0)   # (Q, M, 2)
    dists = torch.norm(query_pts.unsqueeze(1) - proj, dim=2)           # (Q, M)
    return dists.min(dim=1).values                                     # (Q,)


def _points_at_arc_lengths(polyline_xy, cumulative_dist, seg_len, s_values):
    """
    向量化：批量计算 polyline 上多个弧长位置的坐标。
    s_values: (Q,) tensor
    返回: (Q, 2) tensor
    """
    if polyline_xy is None or cumulative_dist is None or seg_len is None or polyline_xy.shape[0] < 2:
        return None
    total = cumulative_dist[-1]
    s_values = s_values.clamp(0.0, float(total.item()))
    idx = torch.searchsorted(cumulative_dist[1:], s_values).clamp(0, seg_len.shape[0] - 1)
    s0 = cumulative_dist[idx]
    s1 = cumulative_dist[idx + 1]
    t = (s_values - s0) / (s1 - s0).clamp(min=1e-6)
    return polyline_xy[idx] + t.unsqueeze(1) * (polyline_xy[idx + 1] - polyline_xy[idx])


def _prepare_route_polyline(route_pts):
    if route_pts is None or not isinstance(route_pts, torch.Tensor):
        return None
    if route_pts.dim() == 3 and route_pts.shape[0] == 1:
        route_pts = route_pts[0]
    if route_pts.dim() != 2 or route_pts.shape[-1] < 2:
        return None
    route_xy = route_pts[:, :2]
    valid = torch.any(route_xy.abs() > 1e-6, dim=1)
    route_xy = route_xy[valid]
    if route_xy.shape[0] < 2:
        return None
    return route_xy


def _find_intersection_ref_point(route_pts, laneline_pts, laneline_mask, lookahead_m=150.0, lane_clearance_m=5.0, backoff_m=10.0, sample_step_m=5.0):
    """
    向量化路口检测：
      - 沿 route 前方 lookahead_m 每隔 sample_step_m 采样；
      - 若某点周围 lane_clearance_m 内没有有效车道线，认为进入路口；
      - 取该位置向后 backoff_m 的 route 点作为 navitopo 分组参考点。
    返回: (has_intersection, ref_point_xy_tensor_or_none)
    """
    route_xy = _prepare_route_polyline(route_pts)
    if route_xy is None:
        return False, None
    if laneline_pts is None or laneline_mask is None:
        return False, None
    if not isinstance(laneline_pts, torch.Tensor) or not isinstance(laneline_mask, torch.Tensor):
        return False, None

    dev = route_xy.device
    dt = route_xy.dtype
    point_xy = torch.zeros(2, device=dev, dtype=dt)
    s0, total_length, cumulative_dist, seg_len = _project_point_to_polyline(point_xy, route_xy)
    if total_length <= 1e-6:
        return False, None
    s_end = min(total_length, s0 + float(lookahead_m))
    if s_end <= s0 + 1e-6:
        return False, None

    valid_mask = (laneline_mask == 0)
    if valid_mask.dim() > 1:
        valid_mask = valid_mask.squeeze()
    valid_indices = valid_mask.nonzero(as_tuple=True)[0]
    if valid_indices.numel() == 0:
        return True, _point_at_arc_length(route_xy, cumulative_dist, seg_len, max(s0, s0 - float(backoff_m)))

    all_starts, all_ends = [], []
    for idx_val in valid_indices.tolist():
        line = laneline_pts[idx_val]
        if line.dim() != 2 or line.shape[-1] < 2:
            continue
        line_xy = line[:, :2]
        non_zero = torch.any(line_xy.abs() > 1e-6, dim=1)
        line_xy = line_xy[non_zero]
        if line_xy.shape[0] >= 2:
            all_starts.append(line_xy[:-1])
            all_ends.append(line_xy[1:])

    if len(all_starts) == 0:
        return True, _point_at_arc_length(route_xy, cumulative_dist, seg_len, max(s0, s0 - float(backoff_m)))

    seg_starts = torch.cat(all_starts, dim=0)   # (M, 2)
    seg_ends = torch.cat(all_ends, dim=0)       # (M, 2)

    s_values = torch.arange(s0, s_end + 1e-6, float(sample_step_m), device=dev, dtype=dt)
    if s_values.numel() == 0:
        return False, None

    sample_pts = _points_at_arc_lengths(route_xy, cumulative_dist, seg_len, s_values)
    if sample_pts is None:
        return False, None

    min_dists = _batch_min_distance_to_segments(sample_pts, seg_starts, seg_ends)
    exceed_mask = min_dists > float(lane_clearance_m)
    if not exceed_mask.any():
        return False, None

    first_idx = int(exceed_mask.nonzero(as_tuple=True)[0][0].item())
    s_intersection = float(s_values[first_idx].item())

    s_ref = max(s0, s_intersection - float(backoff_m))
    ref_point = _point_at_arc_length(route_xy, cumulative_dist, seg_len, s_ref)

    '''
        仅在距离路口30m～100m场景使用
    '''
    # if ref_point is not None:
    #     distance_to_ref = s_ref - s0
    #     if not (30.0 <= distance_to_ref <= 100.0):
    #         return False, None
        
    return ref_point is not None, ref_point




def _group_navitopo_by_proximal(navitopo_pts, navitopo_mask, tol=0.5,reference_point=None):
    """
    根据 (0,0) 到每条 navitopo 的最近点进行分组。
    navitopo_pts: [N, K, 2], navitopo_mask: [N] 或标量, 0=valid, 1=invalid
    返回: (num_groups, groups)
        - num_groups: 组数
        - groups: List[List[int]]，每组为该组内 navitopo 的原始索引，按左→右排序
    """
    if isinstance(navitopo_mask, torch.Tensor):
        valid_mask = (navitopo_mask == 0)
    else:
        valid_mask = torch.ones(navitopo_pts.shape[0], dtype=torch.bool, device=navitopo_pts.device)
    if valid_mask.dim() > 1:
        valid_mask = valid_mask.squeeze()
    if valid_mask.sum() == 0:
        return 0, []
    valid_indices = valid_mask.nonzero(as_tuple=True)[0].cpu().tolist()
    pts = navitopo_pts[valid_indices]
    if pts.dim() == 2:
        pts = pts.unsqueeze(0)
    '''改动'''
    # dists = torch.norm(pts, dim=2)
    if reference_point is None:
        reference_point = torch.zeros(2, device=pts.device, dtype=pts.dtype)
    reference_point = reference_point.to(device=pts.device, dtype=pts.dtype).view(1, 1, 2)
    dists = torch.norm(pts - reference_point, dim=2)


    closest_idx_per_line = torch.argmin(dists, dim=1)
    # 确保 closest_idx_per_line 为 1D，避免单行时变成 0-d 导致索引形状异常
    if closest_idx_per_line.dim() == 0:
        closest_idx_per_line = closest_idx_per_line.unsqueeze(0)
    closest_pts = pts[torch.arange(pts.shape[0], device=pts.device), closest_idx_per_line]
    # 确保 closest_pts 为 (M, 2)，单点时可能是 (2,) 需 unsqueeze
    if closest_pts.dim() == 1:
        closest_pts = closest_pts.unsqueeze(0)
    n_pts = closest_pts.shape[0]
    groups = []
    for i in range(n_pts):
        cp = closest_pts[i]
        orig_idx = valid_indices[i]
        merged = False
        for g in groups:
            rep_i = g[0]
            rep_pos = valid_indices.index(rep_i) if rep_i in valid_indices else 0
            rep_pos = min(rep_pos, n_pts - 1)  # 防止 rep_pos 越界
            rep_cp = closest_pts[rep_pos]
            if torch.norm(cp - rep_cp) < tol:
                g.append(orig_idx)
                merged = True
                break
        if not merged:
            groups.append([orig_idx])
    # 按组代表点的 y 坐标排序（左→右，自车坐标系中 y 为横向）
    if len(groups) > 1:
        group_rep_pts = []
        for g in groups:
            rep_i = g[0]
            rep_pos = valid_indices.index(rep_i) if rep_i in valid_indices else 0
            rep_pos = min(rep_pos, n_pts - 1)  # 防止 rep_pos 越界
            group_rep_pts.append(closest_pts[rep_pos])
        group_rep_pts = torch.stack(group_rep_pts)
        sort_idx = torch.argsort(-group_rep_pts[:, 1])  # y 越大越左，降序得到 左→右
        groups = [groups[i] for i in sort_idx.cpu().tolist()]
    return len(groups), groups


def retain_single_navitopo_recommend_lane(res, tol=0.5, apply_probability=1.0,debug=False):
    """
    当 navitopo 组数 == 推荐车道数 且 > 1 时，仅保留 1 条推荐车道及对应 navitopo。
    用于验证「单组 navitopo+推荐车道」输入下的模型性能。

    res: model_input 字典，需包含 navitopo_pts, navitopo_mask, route_lane_recommend_mask
    tol: 分组距离阈值
    apply_probability: 应用概率，1.0 表示满足条件时必应用
    debug: 是否写入调试字段到 res
    """
    res["flag_single_navi_aug_applied"] = torch.tensor(False)
    res["recommend_lane_count_before_aug"] = torch.tensor(0, dtype=torch.long)
    res["navi_aug_has_intersection"] = torch.tensor(False)
    res["navi_aug_ref_point"] = torch.zeros(2, dtype=torch.float32)
    res["navi_aug_group_count"] = torch.tensor(0, dtype=torch.long)
    res["navi_aug_chosen_group_pos"] = torch.tensor(-1, dtype=torch.long)
    res["navi_aug_chosen_navi_idx"] = torch.tensor(-1, dtype=torch.long)
    res["navi_aug_chosen_lane_slot"] = torch.tensor(-1, dtype=torch.long)
    res["navi_aug_group_pick_nearest"] = torch.tensor(False)

    if "navitopo_pts" not in res or "route_lane_recommend_mask" not in res:
        return res

    navitopo_pts = res["navitopo_pts"]
    navitopo_mask = res.get("navitopo_mask", None)
    route_lane_recommend_mask = res["route_lane_recommend_mask"]
    route_lane_attrs = res.get("route_lane_attrs", None)
    route_pts = res.get("route_pts", None)
    laneline_pts = res.get("laneline_pts", None)
    laneline_mask = res.get("laneline_mask", None)

    device = navitopo_pts.device
    # 若 navitopo_pts 为 (N, K, 2) 无 batch 维，则内部补上 (1, N, K, 2) 便于处理，写回时需去掉
    added_batch = False
    if navitopo_pts.dim() == 3:
        added_batch = True
        navitopo_pts = navitopo_pts.unsqueeze(0)
        if route_lane_recommend_mask.dim() == 1:
            route_lane_recommend_mask = route_lane_recommend_mask.unsqueeze(0)

    bs = navitopo_pts.shape[0]
    if navitopo_mask is None:
        navitopo_mask = torch.zeros(navitopo_pts.shape[1], dtype=torch.float32, device=device)
    if navitopo_mask.dim() == 1:
        navitopo_mask = navitopo_mask.unsqueeze(0).expand(bs, -1)

    navitopo_pts = navitopo_pts.clone()
    navitopo_mask_out = navitopo_mask.clone()
    route_lane_recommend_mask = route_lane_recommend_mask.clone()
    if route_lane_attrs is not None:
        route_lane_attrs = route_lane_attrs.clone()


    '''
        改动
    '''
    def _get_batch_item(x, batch_idx):
        if x is None or not isinstance(x, torch.Tensor):
            return None
        if x.dim() >= 1 and x.shape[0] == bs:
            return x[batch_idx]
        if bs == 1:
            return x
        return None

    def _choose_group_item(pts_i, group_indices, pick_nearest):
        if len(group_indices) == 0:
            return None
        if pick_nearest:
            group_pts = pts_i[group_indices]
            d = torch.norm(group_pts, dim=2).min(dim=1).values
            local_idx = int(torch.argmin(d).item())
            return group_indices[local_idx]
        return random.choice(group_indices)


    for i in range(bs):
        pts_i = navitopo_pts[i]  # (N, K, 2)
        mask_i = navitopo_mask_out[i]
        # num_groups, groups = _group_navitopo_by_proximal(pts_i, mask_i, tol)
        route_i = _get_batch_item(route_pts, i)
        laneline_pts_i = _get_batch_item(laneline_pts, i)
        laneline_mask_i = _get_batch_item(laneline_mask, i)
        has_intersection, ref_point = _find_intersection_ref_point(
            route_i, laneline_pts_i, laneline_mask_i,
            lookahead_m=150.0, lane_clearance_m=5.0, backoff_m=10.0, sample_step_m=5.0
        )
        num_groups, groups = _group_navitopo_by_proximal(
            pts_i, mask_i, tol, reference_point=ref_point if has_intersection else None
        )
        if debug and i == 0:
            res["navi_aug_has_intersection"] = torch.tensor(bool(has_intersection))
            if has_intersection and ref_point is not None:
                res["navi_aug_ref_point"] = ref_point.detach().to(dtype=torch.float32).cpu()
            else:
                res["navi_aug_ref_point"] = torch.zeros(2, dtype=torch.float32)
            res["navi_aug_group_count"] = torch.tensor(int(num_groups), dtype=torch.long)
            res["navi_aug_group_pick_nearest"] = torch.tensor(bool(has_intersection))


        rec_mask_i = route_lane_recommend_mask[i]
        if rec_mask_i.dim() > 1 and rec_mask_i.shape[0] == 1:
            rec_mask_i = rec_mask_i.squeeze(0)

        num_recommend = int((rec_mask_i > 0).sum().item())

        if i == 0:
            res["recommend_lane_count_before_aug"] = torch.tensor(num_recommend, dtype=torch.long)
        if num_groups != num_recommend or num_recommend <= 1:
            continue
        if random.random() > apply_probability:
            continue

        recommend_indices = (rec_mask_i > 0).nonzero(as_tuple=True)[0].cpu().tolist()
        # 保持推荐车道与 groups 的左->右对应关系；先随机选中某一组/推荐车道槽位
        chosen_rec_pos = random.randint(0, num_recommend - 1)
        chosen_lane_slot = recommend_indices[chosen_rec_pos]

        navi_group = groups[chosen_rec_pos]
        # chosen_navi_idx = navi_group[0]  # 同一组内取第一个
 # 有路口：组内按到自车距离最近；无路口：组内随机
        chosen_navi_idx = _choose_group_item(pts_i, navi_group, pick_nearest=has_intersection)
        if chosen_navi_idx is None:
            continue
        if debug and i == 0:
            res["navi_aug_chosen_group_pos"] = torch.tensor(int(chosen_rec_pos), dtype=torch.long)
            res["navi_aug_chosen_navi_idx"] = torch.tensor(int(chosen_navi_idx), dtype=torch.long)
            res["navi_aug_chosen_lane_slot"] = torch.tensor(int(chosen_lane_slot), dtype=torch.long)

        chosen_rec_value = rec_mask_i[chosen_lane_slot].clone()
        # route_lane_recommend_mask[i, :] = 0
        # route_lane_recommend_mask[i, chosen_lane_slot] = chosen_rec_value
        rec_slot_tensor = route_lane_recommend_mask[i]
        if rec_slot_tensor.dim() > 1 and rec_slot_tensor.shape[0] == 1:
            rec_slot_tensor = rec_slot_tensor.squeeze(0)
        rec_slot_tensor[:] = 0
        rec_slot_tensor[chosen_lane_slot] = chosen_rec_value

        if route_lane_attrs is not None:
            attrs_i = route_lane_attrs[i]
            if attrs_i.dim() > 2 and attrs_i.shape[0] == 1:
                attrs_i = attrs_i.squeeze(0)

            for j in recommend_indices:
                if j != chosen_lane_slot:
                    attrs_i[j, 0] = 1  # valid=0 标记为无效
                    attrs_i[j, 1] = 0  # recommended=0

        navitopo_mask_out[i, :] = 1
        navitopo_mask_out[i, chosen_navi_idx] = 0
        res["flag_single_navi_aug_applied"] = torch.tensor(True)

    # 若原始无 batch 维，写回时去掉，避免 collate stack 后多出一维导致 map_encoder 形状错误
    # 仅 squeeze navitopo；route_lane_* 保持与 dataset 原始格式一致，避免 map_encoder_route 维度错误
    if added_batch:
        res["navitopo_pts"] = navitopo_pts.squeeze(0)
        res["navitopo_mask"] = navitopo_mask_out.squeeze(0)
        # route_lane_recommend_mask 原始为 (1,8)，collate 后需 (1,1,8)，故不 squeeze
        res["route_lane_recommend_mask"] = route_lane_recommend_mask
        if route_lane_attrs is not None:
            res["route_lane_attrs"] = route_lane_attrs
    else:
        res["navitopo_pts"] = navitopo_pts
        res["navitopo_mask"] = navitopo_mask_out
        res["route_lane_recommend_mask"] = route_lane_recommend_mask
        if route_lane_attrs is not None:
            res["route_lane_attrs"] = route_lane_attrs

    return res


class naviAugmentation():
    def dropNavi(self, res):
        # 随机dropnavi，并在navi有效时，去掉route的输入
        navi_pts = res['navitopo_pts']
        route_pts = res['route_pts']
        trainFlag = res['trainFlag']
        navitopo_mask = res['navitopo_mask']
        navitopo_attrs = res['navitopo_attrs']
        route_lane_attrs = res.get('route_lane_attrs', None)
        route_lane_recommend_mask = res.get('route_lane_recommend_mask', None)
        # 如果dropnavi，就保留route。如果有navi，就去掉route
        if torch.sum(navi_pts) != 0 and torch.sum(route_pts) != 0:
            if random.random() < 0.25:  # 25%的概率dropout掉navi
                navitopo_mask = torch.ones_like(navitopo_mask)
                navi_pts = torch.zeros_like(navi_pts)
                navitopo_attrs = torch.zeros_like(navitopo_attrs)
            else:
                pass
        else:
            pass
        if torch.sum(navi_pts) != 0:
            route_pts = torch.zeros_like(route_pts)
            if route_lane_attrs is not None:
                route_lane_attrs = torch.zeros_like(route_lane_attrs)
            if route_lane_recommend_mask is not None:
                # 与数据构造语义保持一致: invalid lane 使用 -1
                route_lane_recommend_mask = torch.full_like(route_lane_recommend_mask, -1.0)
            
        res['navitopo_pts'] = navi_pts
        res['navitopo_attrs'] = navitopo_attrs
        res['navitopo_mask'] = navitopo_mask
        res['route_pts'] = route_pts
        if route_lane_attrs is not None:
            res['route_lane_attrs'] = route_lane_attrs
        if route_lane_recommend_mask is not None:
            res['route_lane_recommend_mask'] = route_lane_recommend_mask

        return res