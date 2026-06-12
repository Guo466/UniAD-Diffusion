import numpy as np
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_yaw_rotation_matrix(yaw,dtype=np.ndarray):
    '''
    Args:
        yaw: x axis rotation angle in radian. Any shape except empty.

    Returns:
        yaw rotation matrix with shape [input_shape, 3, 3]
    '''
    if dtype is np.ndarray:
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        ones = np.ones_like(yaw)
        zeros = np.zeros_like(yaw) # [input_shape]

        first_row = np.stack([cos_yaw, -1.0 * sin_yaw, zeros], axis=-1)
        second_row = np.stack([sin_yaw, cos_yaw, zeros], axis=-1)
        third_row = np.stack([zeros, zeros, ones], axis=-1)

        # yaw rotation matrix
        # [cos, -sin, 0]
        # [sin,  cos, 0]
        # [ 0 ,   0 , 1]
        return np.stack([first_row, second_row, third_row], axis=-2)
    else:
        import torch
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        ones = torch.ones_like(yaw)
        zeros = torch.zeros_like(yaw) # [input_shape]

        first_row = torch.stack([cos_yaw, -1.0 * sin_yaw, zeros], axis=-1)
        second_row = torch.stack([sin_yaw, cos_yaw, zeros], axis=-1)
        third_row = torch.stack([zeros, zeros, ones], axis=-1)

        # yaw rotation matrix
        # [cos, -sin, 0]
        # [sin,  cos, 0]
        # [ 0 ,   0 , 1]
        return torch.stack([first_row, second_row, third_row], axis=-2)

def transform_component_coordinates(coordinates, yaw_rotation_matrix, translation=None):
    """
    rotation after translation
    """
    translated_coordinates = coordinates + translation[:, np.newaxis] if translation is not None else coordinates
    return np.einsum('nij, nkj->nki', yaw_rotation_matrix, translated_coordinates)

def transform_back_component_coordinates(coordinates, yaw_rotation_matrix, translation=None):
    """
    rotation before translation for evaluation
    """
    rotated_coodinates = np.einsum('...ij, ...kj->...ki', yaw_rotation_matrix[:, np.newaxis], coordinates)
    return rotated_coodinates + translation[:, np.newaxis, np.newaxis] if translation is not None else rotated_coodinates

def wrap_angle(angle):
    '''Wrap angles in the range [-pi, pi]'''
    return (angle + np.pi) % (2 * np.pi) - np.pi