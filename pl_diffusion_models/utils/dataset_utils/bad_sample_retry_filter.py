import math
from typing import Any, Dict, List, Optional, Tuple


def _dedup_adjacent(points: List[Tuple[float, float]], eps: float = 1e-6) -> List[Tuple[float, float]]:
    if not points:
        return points
    out = [points[0]]
    for p in points[1:]:
        if math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > eps:
            out.append(p)
    return out


def _compute_cumulative_dist(points: List[Tuple[float, float]]) -> Tuple[List[float], List[float]]:
    seg_len: List[float] = []
    cumulative: List[float] = [0.0]
    for i in range(1, len(points)):
        d = math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
        seg_len.append(d)
        cumulative.append(cumulative[-1] + d)
    return seg_len, cumulative


def _interpolate_at_s(
    points: List[Tuple[float, float]],
    cumulative_dist: List[float],
    s: float,
) -> Tuple[float, float]:
    if s <= 0.0:
        return points[0]
    if s >= cumulative_dist[-1]:
        return points[-1]
    for i in range(1, len(cumulative_dist)):
        if cumulative_dist[i] >= s:
            seg = cumulative_dist[i] - cumulative_dist[i - 1]
            if seg <= 1e-6:
                return points[i]
            t = (s - cumulative_dist[i - 1]) / seg
            x = points[i - 1][0] + t * (points[i][0] - points[i - 1][0])
            y = points[i - 1][1] + t * (points[i][1] - points[i - 1][1])
            return (x, y)
    return points[-1]


def _project_origin_onto_polyline(
    points: List[Tuple[float, float]],
    seg_len: List[float],
    cumulative_dist: List[float],
) -> float:
    best_dist2 = float("inf")
    s_proj = 0.0
    for i in range(len(points) - 1):
        p0x, p0y = points[i]
        p1x, p1y = points[i + 1]
        vx, vy = p1x - p0x, p1y - p0y
        denom = vx * vx + vy * vy
        if denom <= 1e-8:
            t = 0.0
            projx, projy = p0x, p0y
        else:
            t = -(p0x * vx + p0y * vy) / denom
            t = max(0.0, min(1.0, t))
            projx, projy = p0x + t * vx, p0y + t * vy
        dist2 = projx * projx + projy * projy
        if dist2 < best_dist2:
            best_dist2 = dist2
            s_proj = cumulative_dist[i] + t * seg_len[i]
    return s_proj


def _point_to_segment_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    vx = x2 - x1
    vy = y2 - y1
    denom = vx * vx + vy * vy
    if denom <= 1e-8:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * vx + (py - y1) * vy) / denom
    t = max(0.0, min(1.0, t))
    projx = x1 + t * vx
    projy = y1 + t * vy
    return math.hypot(px - projx, py - projy)


def _lateral_distance_to_polyline(
    px: float,
    py: float,
    polyline: List[Tuple[float, float]],
) -> Tuple[float, bool]:
    if len(polyline) == 0:
        return float("inf"), False
    if len(polyline) == 1:
        return math.hypot(px - polyline[0][0], py - polyline[0][1]), False

    best_dist = float("inf")
    best_seg_idx = 0
    best_t = 0.0

    for i in range(len(polyline) - 1):
        x1, y1 = polyline[i]
        x2, y2 = polyline[i + 1]
        vx = x2 - x1
        vy = y2 - y1
        denom = vx * vx + vy * vy
        if denom <= 1e-8:
            dist = math.hypot(px - x1, py - y1)
            t = 0.0
        else:
            t_raw = ((px - x1) * vx + (py - y1) * vy) / denom
            t = max(0.0, min(1.0, t_raw))
            projx = x1 + t * vx
            projy = y1 + t * vy
            dist = math.hypot(px - projx, py - projy)
        if dist < best_dist:
            best_dist = dist
            best_seg_idx = i
            best_t = t

    last_idx = len(polyline) - 2
    beyond_end = best_seg_idx == last_idx and best_t >= 1.0 - 1e-6
    beyond_start = best_seg_idx == 0 and best_t <= 1e-6
    within_range = not (beyond_end or beyond_start)
    return best_dist, within_range


def _cross2d(ox: float, oy: float, ax: float, ay: float) -> float:
    return ox * ay - oy * ax


def _segments_intersect(
    ax1: float,
    ay1: float,
    ax2: float,
    ay2: float,
    bx1: float,
    by1: float,
    bx2: float,
    by2: float,
) -> bool:
    dx_a = ax2 - ax1
    dy_a = ay2 - ay1
    dx_b = bx2 - bx1
    dy_b = by2 - by1
    d1 = _cross2d(dx_b, dy_b, ax1 - bx1, ay1 - by1)
    d2 = _cross2d(dx_b, dy_b, ax2 - bx1, ay2 - by1)
    d3 = _cross2d(dx_a, dy_a, bx1 - ax1, by1 - ay1)
    d4 = _cross2d(dx_a, dy_a, bx2 - ax1, by2 - ay1)
    if d1 * d2 < 0 and d3 * d4 < 0:
        return True

    eps = 1e-10

    def _on_segment(px: float, py: float, qx1: float, qy1: float, qx2: float, qy2: float) -> bool:
        return min(qx1, qx2) - eps <= px <= max(qx1, qx2) + eps and min(qy1, qy2) - eps <= py <= max(qy1, qy2) + eps

    if abs(d1) < eps and _on_segment(ax1, ay1, bx1, by1, bx2, by2):
        return True
    if abs(d2) < eps and _on_segment(ax2, ay2, bx1, by1, bx2, by2):
        return True
    if abs(d3) < eps and _on_segment(bx1, by1, ax1, ay1, ax2, ay2):
        return True
    if abs(d4) < eps and _on_segment(bx2, by2, ax1, ay1, ax2, ay2):
        return True
    return False


def _line_intersects_polyline(
    lx1: float,
    ly1: float,
    lx2: float,
    ly2: float,
    polyline: List[Tuple[float, float]],
) -> bool:
    for i in range(len(polyline) - 1):
        if _segments_intersect(
            lx1,
            ly1,
            lx2,
            ly2,
            polyline[i][0],
            polyline[i][1],
            polyline[i + 1][0],
            polyline[i + 1][1],
        ):
            return True
    return False


def _extract_route_points(sample: Any) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for p in sample.route.ego_routing.points:
        cur = (float(p.x), float(p.y))
        if not pts or cur != pts[-1]:
            pts.append(cur)
    return pts


def _truncate_route_local(
    route_points: List[Tuple[float, float]],
    behind_m: float,
    ahead_m: float,
) -> List[Tuple[float, float]]:
    if len(route_points) < 2:
        return route_points
    seg_len, cumulative_dist = _compute_cumulative_dist(route_points)
    total_length = cumulative_dist[-1]
    if total_length <= 1e-6:
        return route_points
    s_proj = _project_origin_onto_polyline(route_points, seg_len, cumulative_dist)
    s_start = max(0.0, s_proj - behind_m)
    s_end = min(total_length, s_proj + ahead_m)
    if s_end <= s_start:
        return route_points
    start_pt = _interpolate_at_s(route_points, cumulative_dist, s_start)
    end_pt = _interpolate_at_s(route_points, cumulative_dist, s_end)
    truncated: List[Tuple[float, float]] = [start_pt]
    for i in range(1, len(route_points) - 1):
        if s_start < cumulative_dist[i] < s_end:
            truncated.append(route_points[i])
    truncated.append(end_pt)
    dedup = _dedup_adjacent(truncated)
    return dedup if len(dedup) > 1 else route_points


def _interpolate_route_points(route_points: List[Tuple[float, float]], step_m: float = 1.0) -> List[Tuple[float, float]]:
    if len(route_points) < 2:
        return route_points
    _, cumulative_dist = _compute_cumulative_dist(route_points)
    total_length = cumulative_dist[-1]
    if total_length <= 1e-6:
        return [route_points[0]]
    out: List[Tuple[float, float]] = []
    s = 0.0
    while s < total_length:
        out.append(_interpolate_at_s(route_points, cumulative_dist, s))
        s += max(1e-3, step_m)
    if math.hypot(out[-1][0] - route_points[-1][0], out[-1][1] - route_points[-1][1]) > 0.1:
        out.append(route_points[-1])
    return _dedup_adjacent(out)


def _prepare_local_route(
    sample: Any,
    behind_m: float,
    ahead_m: float,
    first_dist_thr: float,
) -> List[Tuple[float, float]]:
    route_points = _extract_route_points(sample)
    if len(route_points) < 2:
        return route_points
    first_dist = math.hypot(route_points[0][0], route_points[0][1])
    if first_dist > first_dist_thr:
        route_points = _truncate_route_local(route_points, behind_m, ahead_m)
    return _interpolate_route_points(route_points, step_m=1.0)


def _extract_navitopo_real_endpoint(pts_points: Any) -> Optional[Tuple[float, float]]:
    if len(pts_points) == 0:
        return None
    all_pts = [(float(p.x), float(p.y)) for p in pts_points]
    end_idx = len(all_pts) - 1
    while end_idx > 0:
        if math.hypot(all_pts[end_idx][0] - all_pts[end_idx - 1][0], all_pts[end_idx][1] - all_pts[end_idx - 1][1]) > 1e-4:
            break
        end_idx -= 1
    return all_pts[end_idx]


def _navitopo_length(pts_points: Any) -> float:
    if len(pts_points) == 0:
        return 0.0
    all_pts = [(float(p.x), float(p.y)) for p in pts_points]
    end_idx = len(all_pts) - 1
    while end_idx > 0:
        if math.hypot(all_pts[end_idx][0] - all_pts[end_idx - 1][0], all_pts[end_idx][1] - all_pts[end_idx - 1][1]) > 1e-4:
            break
        end_idx -= 1
    total = 0.0
    for i in range(end_idx):
        total += math.hypot(all_pts[i + 1][0] - all_pts[i][0], all_pts[i + 1][1] - all_pts[i][1])
    return total


def _find_nearest_navitopo_point(sample: Any) -> Optional[Tuple[float, float]]:
    best_dist2 = float("inf")
    best_point: Optional[Tuple[float, float]] = None
    for navi in sample.data_laneline_navi_topo:
        for p in navi.pts_fixed_num.points:
            px, py = float(p.x), float(p.y)
            d2 = px * px + py * py
            if d2 < best_dist2:
                best_dist2 = d2
                best_point = (px, py)
    return best_point


def _is_curb(laneline: Any) -> bool:
    return (1 <= int(getattr(laneline, "edge_type", 0)) <= 5) or (10 <= int(getattr(laneline, "laneline_type", 0)) <= 11)


def _is_yellow_solid(laneline: Any) -> bool:
    return int(getattr(laneline, "lane_color", 0)) == 3 and int(getattr(laneline, "laneline_type", 0)) in (2, 5, 7, 9)


def _iter_laneline_points(ll: Any) -> List[Tuple[float, float]]:
    if hasattr(ll, "sampled_points") and hasattr(ll.sampled_points, "points") and len(ll.sampled_points.points) > 0:
        return [(float(p.x), float(p.y)) for p in ll.sampled_points.points]
    if hasattr(ll, "pts_fixed_num"):
        pts_src = ll.pts_fixed_num
        if hasattr(pts_src, "points"):
            pts_src = pts_src.points
        return [(float(p.x), float(p.y)) for p in pts_src]
    return []


def _extract_boundary_polylines(sample: Any) -> List[List[Tuple[float, float]]]:
    polylines: List[List[Tuple[float, float]]] = []
    for ll in sample.lanelines:
        if getattr(ll, "type", "") != "laneline":
            continue
        if not (_is_curb(ll) or _is_yellow_solid(ll)):
            continue
        pts = _dedup_adjacent(_iter_laneline_points(ll))
        if len(pts) >= 2:
            polylines.append(pts)
    return polylines


def _check_navitopo_endpoint_far_from_route(
    sample: Any,
    local_route: List[Tuple[float, float]],
    threshold_m: float,
) -> Tuple[bool, str]:
    if len(local_route) < 2:
        return False, ""
    navis = sample.data_laneline_navi_topo
    if len(navis) == 0:
        return False, ""
    longest_len = -1.0
    longest_idx = -1
    longest_endpoint: Optional[Tuple[float, float]] = None
    for idx, navi in enumerate(navis):
        endpoint = _extract_navitopo_real_endpoint(navi.pts_fixed_num.points)
        if endpoint is None:
            continue
        navi_len = _navitopo_length(navi.pts_fixed_num.points)
        if navi_len > longest_len:
            longest_len = navi_len
            longest_idx = idx
            longest_endpoint = endpoint
    if longest_endpoint is None:
        return False, ""
    lateral_dist, within_range = _lateral_distance_to_polyline(longest_endpoint[0], longest_endpoint[1], local_route)
    if not within_range:
        return False, ""
    if lateral_dist > threshold_m:
        return True, (
            f"rule1:longest_navitopo[{longest_idx}]_len={longest_len:.1f}m"
            f"_lateral_dist={lateral_dist:.2f}m>thr={threshold_m:.1f}m"
        )
    return False, ""


def _check_ego_separated_by_boundary(
    sample: Any,
) -> Tuple[bool, str]:
    nearest_navi_pt = _find_nearest_navitopo_point(sample)
    if nearest_navi_pt is None:
        return False, ""
    boundary_polylines = _extract_boundary_polylines(sample)
    if not boundary_polylines:
        return False, ""
    ego_x, ego_y = 0.0, 0.0
    navi_x, navi_y = nearest_navi_pt
    ego_to_navi_dist = math.hypot(navi_x - ego_x, navi_y - ego_y)
    if ego_to_navi_dist < 1e-3:
        return False, ""
    for poly in boundary_polylines:
        if _line_intersects_polyline(ego_x, ego_y, navi_x, navi_y, poly):
            return True, f"rule2:ego_to_nearest_navi=({navi_x:.1f},{navi_y:.1f})_crosses_boundary"
    return False, ""


def _extract_future_endpoint_from_fixed(
    ego_future_status_fixed: Any,
    ego_future_mask_fixed: Any,
) -> Optional[Tuple[float, float]]:
    if ego_future_status_fixed is None or ego_future_mask_fixed is None:
        return None
    pts = ego_future_status_fixed
    masks = ego_future_mask_fixed
    try:
        if hasattr(pts, "detach"):
            pts = pts.detach().cpu().numpy()
        if hasattr(masks, "detach"):
            masks = masks.detach().cpu().numpy()
    except Exception:
        return None
    if len(pts) == 0 or len(masks) == 0:
        return None
    valid_indices = [i for i, v in enumerate(masks) if bool(v)]
    if not valid_indices:
        return None
    last_i = valid_indices[-1]
    if last_i >= len(pts):
        return None
    return float(pts[last_i][0]), float(pts[last_i][1])


def _check_future_endpoint_far_from_route(
    local_route: List[Tuple[float, float]],
    threshold_m: float,
    ego_future_status_fixed: Any,
    ego_future_mask_fixed: Any,
) -> Tuple[bool, str]:
    if len(local_route) < 2:
        return False, ""
    endpoint = _extract_future_endpoint_from_fixed(ego_future_status_fixed, ego_future_mask_fixed)
    if endpoint is None:
        return False, ""
    px, py = endpoint
    lateral_dist, within_range = _lateral_distance_to_polyline(px, py, local_route)
    if not within_range:
        return False, ""
    if lateral_dist > threshold_m:
        return True, (
            f"rule3:path_gt_endpoint=({px:.1f},{py:.1f})"
            f"_lateral_dist={lateral_dist:.2f}m>thr={threshold_m:.1f}m"
        )
    return False, ""


def check_bad_sample_for_retry(
    sample: Any,
    ego_future_status_fixed: Any,
    ego_future_mask_fixed: Any,
    threshold_m: float = 30.0,
    route_truncate_first_dist_thr_m: float = 50.0,
    route_behind_m: float = 5.0,
    route_ahead_m: float = 195.0,
) -> Tuple[bool, List[str], Dict[str, bool]]:
    local_route = _prepare_local_route(
        sample,
        behind_m=route_behind_m,
        ahead_m=route_ahead_m,
        first_dist_thr=route_truncate_first_dist_thr_m,
    )
    hit1, reason1 = _check_navitopo_endpoint_far_from_route(sample, local_route, threshold_m)
    hit2, reason2 = _check_ego_separated_by_boundary(sample)
    hit3, reason3 = _check_future_endpoint_far_from_route(
        local_route,
        threshold_m,
        ego_future_status_fixed,
        ego_future_mask_fixed,
    )
    reasons = [r for r in (reason1, reason2, reason3) if r]
    detail = {
        "rule1_navitopo_far": bool(hit1),
        "rule2_ego_off_route": bool(hit2),
        "rule3_future_far": bool(hit3),
    }
    return bool(hit1 or hit2 or hit3), reasons, detail
