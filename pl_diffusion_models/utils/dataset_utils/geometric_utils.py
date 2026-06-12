
from dataset_utils.state_perburbation import random_offset_generator
from scipy.interpolate import interp1d
from scipy.spatial import distance
from shapely.geometry import Polygon
import scipy.spatial as spt
import numpy as np
approximate_agent_cls_shape = {
    1 : (0.6,0.6), # pedestrian
    2 : (4.835,1.9), # vehicle
    3 : (1.8,0.7) # bike
}

class CustomPathPoint:
    def __init__(self, x, y, progress, angle):
        self.x = x
        self.y = y
        self.progress = progress
        self.angle = angle
    
    def __repr__(self):
        return f"Point(x={self.x}, y={self.y}, progress={self.progress}, angle={self.angle})"

def random_shape_generator(cls, offset_rate = 0.1):
    length = random_offset_generator(low=[(1-offset_rate)*approximate_agent_cls_shape[cls][0]], high=[(1+offset_rate)*approximate_agent_cls_shape[cls][0]])
    width = random_offset_generator(low=[(1-offset_rate)*approximate_agent_cls_shape[cls][1]], high=[(1+offset_rate)*approximate_agent_cls_shape[cls][1]])
    return (length,width)

def generate_rotated_bbox_points(centers, size, angle):
    """
    根据中心点、尺寸和旋转角度生成旋转后的包围盒
    :param center: 中心点 (x, y) shape n,2
    :param size: 尺寸 (length, width) shape n,2
    :param angle: 旋转角度（弧度）shape n,1
    :return: 旋转后的包围盒的四个顶点坐标
    """
    if centers.shape[0] == 0:
        return None
    half_lengths = size[:, 0] / 2.0  # Half lengths
    half_widths = size[:, 1] / 2.0   # Half widths
    n = centers.shape[0]
    bbox_points = np.empty((n, 4, 2))
    
    # Top-right corner
    bbox_points[:, 0, 0] = centers[:, 0] + half_lengths
    bbox_points[:, 0, 1] = centers[:, 1] + half_widths
    
    # Bottom-right corner
    bbox_points[:, 1, 0] = centers[:, 0] + half_lengths
    bbox_points[:, 1, 1] = centers[:, 1] - half_widths
    
    # Bottom-left corner
    bbox_points[:, 2, 0] = centers[:, 0] - half_lengths
    bbox_points[:, 2, 1] = centers[:, 1] - half_widths
    
    # Top-left corner
    bbox_points[:, 3, 0] = centers[:, 0] - half_lengths
    bbox_points[:, 3, 1] = centers[:, 1] + half_widths

    s = np.sin(angle)
    c = np.cos(angle)

    rotation_matrices = np.empty((n, 2, 2))
    
    rotation_matrices[:, 0, 0] = c
    rotation_matrices[:, 0, 1] = -s
    rotation_matrices[:, 1, 0] = s
    rotation_matrices[:, 1, 1] = c

    bbox_points_center_cosy = bbox_points - centers.numpy()[:, np.newaxis, :]

    rotated_points_center_cosy = np.einsum('nij,nkj->nki', rotation_matrices, bbox_points_center_cosy)
    rotated_points = rotated_points_center_cosy + centers.numpy()[:, np.newaxis, :]

    return rotated_points # size: n,4,2

def cal_convex_iou(points_1, points_2):
    '''
       points sets to get iou
    '''
    hull_1 = spt.ConvexHull(points=points_1)
    hull_2 = spt.ConvexHull(points=points_2)
    convex_1 = points_1[hull_1.vertices]
    convex_2 = points_2[hull_2.vertices]

    poly1 = Polygon(convex_1).convex_hull
    poly2 = Polygon(convex_2).convex_hull

    if not poly1.intersects(poly2):
        inter_area = 0
    else:
        inter_area = poly1.intersection(poly2).area

    return inter_area / (poly1.area + poly2.area - inter_area)