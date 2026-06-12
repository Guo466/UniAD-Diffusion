import math
import numpy as np
from typing import List



class StPoint2D:
    """2D点类"""
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        self.y = y

    def norm(self) -> float:
        """计算模长"""
        return math.sqrt(self.x **2 + self.y** 2)

    def squared_norm(self) -> float:
        """计算模长平方"""
        return self.x **2 + self.y** 2

    def dot(self, p: 'StPoint2D') -> float:
        """点积"""
        return self.x * p.x + self.y * p.y

    def cross(self, p: 'StPoint2D') -> float:
        """叉积"""
        return self.x * p.y - self.y * p.x

    def __repr__(self) -> str:
        return f"({self.x}, {self.y})"

    def __eq__(self, other: 'StPoint2D') -> bool:
        if not isinstance(other, StPoint2D):
            return False
        return st_float_equal(self.x, other.x) and st_float_equal(self.y, other.y)

    def __neg__(self) -> 'StPoint2D':
        return StPoint2D(-self.x, -self.y)

    def __add__(self, other: 'StPoint2D') -> 'StPoint2D':
        return StPoint2D(self.x + other.x, self.y + other.y)

    def __iadd__(self, other: 'StPoint2D') -> 'StPoint2D':
        self.x += other.x
        self.y += other.y
        return self

    def __sub__(self, other: 'StPoint2D') -> 'StPoint2D':
        return StPoint2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> 'StPoint2D':
        return StPoint2D(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> 'StPoint2D':
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> 'StPoint2D':
        return StPoint2D(self.x / scalar, self.y / scalar)

class StUsedPoint:
    def __init__(self, t: float = 0.0, s_max: float = 0.0, s_min: float = 0.0):
        self.t = t
        self.s_max = s_max
        self.s_min = s_min

class StObjectContainer:
    def __init__(self, obs_id: int = -1, obj_sqr: float = 0.0):
        self.obs_id = obs_id
        self.collision_points = []
        self.obj_sqr = obj_sqr

class StTrajectoryPoint:
    def __init__(self, position: StPoint2D = StPoint2D(), direction: StPoint2D = StPoint2D(), velocity: float = 0.0, theta: float = 0.0, sum_distance: float = 0.0, time_difference: float = 0.0):
        self.position = position
        self.direction = direction
        self.velocity = velocity
        self.theta = theta
        self.sum_distance = sum_distance
        self.time_difference = time_difference  # 相对时间，单位：秒



class StObject:
    def __init__(self, id: int = -1, length: float = 0.0, width: float = 0.0):
        self.id = id
        self.length = length
        self.width = width
        self.trajectory = []


def print_point2d_vector(points: List[StPoint2D]) -> str:
    """打印点列表"""
    return " ".join([f"{p.x},{p.y}" for p in points])


def st_float_equal(x, y) -> bool:
    """浮点数比较"""
    max_val = max(1.0, abs(x), abs(y))
    return abs(x - y) <= np.finfo(np.float64).eps * max_val