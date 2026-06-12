import math
import time
from typing import List, Tuple, Optional
from visualization.st_graph.stclass import StPoint2D, st_float_equal


class StPolygon2D:
    """2D多边形类"""
    k_eps = 1e-6
    k_float_eps = 1e-6

    def __init__(self, points = None):
        self.points = points if points is not None else []
        self.min_x = 0.0
        self.min_y = 0.0
        self.max_x = 0.0
        self.max_y = 0.0
        self.min_x_idx = 0
        self.max_x_idx = 0
        self.line_segment_length = []
        self.unit_direction = []
        self.lower_points = []
        self.upper_points = []

        if self.points and len(self.points) > 2:
            self._initialize()

    def _initialize(self):
        """初始化多边形参数"""
        self.min_x = self.points[0].x
        self.max_x = self.points[0].x
        self.min_y = self.points[0].y
        self.max_y = self.points[0].y
        for i, point in enumerate(self.points):
            next_point = self.points[self.next(i)]
            direction = next_point - point
            length = direction.norm()
            self.line_segment_length.append(length)
            
            if length <= self.k_eps:
                unit_dir = StPoint2D(0.0, 0.0)
            else:
                unit_dir = direction / length
            self.unit_direction.append(unit_dir)

            if point.x < self.min_x - self.k_float_eps:
                self.min_x = point.x
                self.min_x_idx = i
            if point.x > self.max_x + self.k_float_eps:
                self.max_x = point.x
                self.max_x_idx = i

            self.min_y = min(self.min_y, point.y)
            self.max_y = max(self.max_y, point.y)
        # self.split_to_upper_and_lower()


    def size(self) -> int:
        return len(self.points)

    def __getitem__(self, idx: int) -> 'StPoint2D':
        return self.points[idx]

    def __len__(self) -> int:
        return len(self.points)

    def is_empty(self) -> bool:
        return len(self.points) == 0

    def intersect(self, polygon: 'StPolygon2D') -> bool:
        """检查两个多边形是否相交（分离轴定理）"""
        if self.is_empty() or polygon.is_empty():
            return False

        # 检查轴对齐包围盒是否相交
        a_min_x, a_max_x = self.min_x, self.max_x
        a_min_y, a_max_y = self.min_y, self.max_y
        b_min_x, b_max_x = polygon.min_x, polygon.max_x
        b_min_y, b_max_y = polygon.min_y, polygon.max_y

        if b_max_x < a_min_x or a_max_x < b_min_x:
            return False
        if b_max_y < a_min_y or a_max_y < b_min_y:
            return False

        # 检查第一个多边形的边
        for i in range(len(self.points)):
            current = self.points[i]
            next_p = self.points[self.next(i)]
            edge = next_p - current
            axis = StPoint2D(-edge.y, edge.x)  # 法向量作为轴

            a_proj = [axis.dot(p) for p in self.points]
            b_proj = [axis.dot(p) for p in polygon.points]

            a_min, a_max = min(a_proj), max(a_proj)
            b_min, b_max = min(b_proj), max(b_proj)

            if a_max < b_min - self.k_eps or a_min > b_max + self.k_eps:
                return False

        # 检查第二个多边形的边
        for i in range(len(polygon.points)):
            current = polygon.points[i]
            next_p = polygon.points[polygon.next(i)]
            edge = next_p - current
            axis = StPoint2D(-edge.y, edge.x)  # 法向量作为轴

            a_proj = [axis.dot(p) for p in self.points]
            b_proj = [axis.dot(p) for p in polygon.points]

            a_min, a_max = min(a_proj), max(a_proj)
            b_min, b_max = min(b_proj), max(b_proj)

            if a_max < b_min - self.k_eps or a_min > b_max + self.k_eps:
                return False

        return True

    def point_in(self, point: 'StPoint2D') -> bool:
        """检查点是否在多边形内（射线法）"""
        j = len(self.points) - 1
        count = 0
        for i in range(len(self.points)):
            if (self.points[i].y > point.y) != (self.points[j].y > point.y):
                side = self.cross_prod_2d(point, self.points[i], self.points[j])
                if (self.points[i].y < self.points[j].y and side > self.k_eps) or \
                   (self.points[i].y > self.points[j].y and side < -self.k_eps):
                    count += 1
            j = i
        return count % 2 == 1

    def distance_to_point(self, point: 'StPoint2D') -> float:
        """计算点到多边形的距离"""
        if self.point_in(point):
            return 0.0

        min_dist = float('inf')
        for i in range(len(self.points)):
            vec = point - self.points[i]
            proj = vec.dot(self.unit_direction[i])
            
            if proj <= self.k_eps:
                dist = vec.norm()
            elif proj >= self.line_segment_length[i] - self.k_eps:
                dist = (point - self.points[self.next(i)]).norm()
            else:
                dist = abs(vec.cross(self.unit_direction[i]))

            if dist < min_dist:
                min_dist = dist

        return min_dist

    def distance_to_polygon(self, polygon: 'StPolygon2D') -> float:
        """计算两个多边形之间的距离"""
        if self.intersect(polygon):
            return 0.0

        min_dist = float('inf')
        for p in polygon.points:
            min_dist = min(min_dist, self.distance_to_point(p))
        for p in self.points:
            min_dist = min(min_dist, polygon.distance_to_point(p))
        return min_dist

    def add_point(self, point: 'StPoint2D'):
        """添加点到多边形"""
        self.points.append(point)
        # 重新初始化（简化处理）
        if len(self.points) > 2:
            self._initialize()

    def next(self, idx: int) -> int:
        """获取下一个点的索引"""
        return 0 if idx >= len(self.points) - 1 else idx + 1

    def before(self, idx: int) -> int:
        """获取上一个点的索引"""
        return len(self.points) - 1 if idx <= 0 else idx - 1

    @staticmethod
    def cross_prod_2d(p1: 'StPoint2D', p2: 'StPoint2D', p3: 'StPoint2D') -> float:
        """计算二维叉积 (p2-p1) × (p3-p2)"""
        return (p2 - p1).cross(p3 - p2)

    def split_to_upper_and_lower(self):
        """将多边形分割为上边界和下边界点"""
        self.upper_points.clear()
        self.lower_points.clear()
        if not self.points:
            return

        # 初始化上下边界点
        upper_tmp = [self.points[self.min_x_idx], self.points[self.max_x_idx]]
        lower_tmp = [self.points[self.min_x_idx], self.points[self.max_x_idx]]

        # 填充中间点
        min_idx = min(self.min_x_idx, self.max_x_idx)
        max_idx = max(self.min_x_idx, self.max_x_idx)
        for i in range(min_idx + 1, max_idx):
            upper_tmp.append(self.points[i % len(self.points)])

        for i in range(max_idx + 1, min_idx + len(self.points)):
            lower_tmp.append(self.points[i % len(self.points)])

        # 排序函数
        def cmp(point: StPoint2D) -> Tuple[float, int]:
            for idx, p in enumerate(self.points):
                if st_float_equal(p.x, point.x) and st_float_equal(p.y, point.y):
                    return (point.x, idx)
            return (point.x, len(self.points))

        upper_tmp.sort(key=cmp)
        lower_tmp.sort(key=cmp)

        # 确定上下边界
        if (lower_tmp[1] - lower_tmp[0]).cross(upper_tmp[1] - upper_tmp[0]) < self.k_float_eps:
            sorted_upper = lower_tmp
            sorted_lower = upper_tmp
        else:
            sorted_upper = upper_tmp
            sorted_lower = lower_tmp

        # 去重
        self.lower_points = self._remove_duplicate_points(sorted_lower, keep_min=True)
        self.upper_points = self._remove_duplicate_points(sorted_upper, keep_min=False)

    @staticmethod
    def _remove_duplicate_points(points: List[StPoint2D], keep_min: bool):
        """移除重复点"""
        if not points:
            return
        out = []
        out.append(points[0])
        for p in points[1:]:
            last = out[-1]
            if abs(p.x - last.x) > StPolygon2D.k_float_eps:
                out.append(p)
            else:
                if keep_min:
                    last.y = min(last.y, p.y)
                else:
                    last.y = max(last.y, p.y)

    def __add__(self, other: 'StPolygon2D') -> 'StPolygon2D':
        if self.is_empty():
            return other
        if other.is_empty():
            return self
        if len(self.points) != len(other.points):
            return StPolygon2D()
        return StPolygon2D([self.points[i] + other.points[i] for i in range(len(self.points))])

    def __sub__(self, other: 'StPolygon2D') -> 'StPolygon2D':
        if len(self.points) != len(other.points):
            return StPolygon2D()
        return StPolygon2D([self.points[i] - other.points[i] for i in range(len(self.points))])

    def __mul__(self, ratio: float) -> 'StPolygon2D':
        return StPolygon2D([p * ratio for p in self.points])

    def __rmul__(self, ratio: float) -> 'StPolygon2D':
        return self.__mul__(ratio)

    def __repr__(self) -> str:
        return f"StPolygon2D({[p.__repr__() for p in self.points]})"