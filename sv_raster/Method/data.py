import os
import torch
import trimesh
import numpy as np
from typing import Optional, Union, Tuple


def loadPointCloud(
    pcd_file_path: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    加载点云文件

    Args:
        pcd_file_path: 点云文件路径

    Returns:
        (points, colors): 点坐标和颜色（如果有），如果失败则返回 (None, None)
    """
    if not os.path.exists(pcd_file_path):
        print('[ERROR][data::loadPointCloud]')
        print('\t pcd file not exist!')
        print('\t pcd_file_path:', pcd_file_path)
        return None, None

    pcd = trimesh.load(pcd_file_path)

    # 获取点坐标
    if hasattr(pcd, 'vertices'):
        points = np.array(pcd.vertices)
    elif hasattr(pcd, 'points'):
        points = np.array(pcd.points)
    else:
        print('[ERROR][data::loadPointCloud]')
        print('\t Cannot extract points from file')
        return None, None

    # 获取颜色（如果有）
    colors = None
    if hasattr(pcd, 'colors') and pcd.colors is not None:
        colors = np.array(pcd.colors)[:, :3]
        if colors.max() > 1.0:
            colors = colors / 255.0
    elif hasattr(pcd, 'visual') and hasattr(pcd.visual, 'vertex_colors'):
        colors = np.array(pcd.visual.vertex_colors)[:, :3] / 255.0

    return points, colors


def sampleMeshSurface(
    mesh: trimesh.Trimesh,
    n_samples: Optional[int] = None,
    sample_density: float = 100.0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    从mesh表面采样点

    Args:
        mesh: trimesh.Trimesh对象
        n_samples: 采样点数，如果为None则根据sample_density计算
        sample_density: 每单位面积的采样点数

    Returns:
        (points, colors): 采样点坐标和颜色（如果有）
    """
    if n_samples is None:
        area = mesh.area
        n_samples = max(10000, int(area * sample_density))

    points, face_indices = trimesh.sample.sample_surface(mesh, n_samples)

    # 获取颜色（如果有）
    colors = None
    if mesh.visual.vertex_colors is not None:
        vertex_colors = mesh.visual.vertex_colors[:, :3] / 255.0
        face_vertices = mesh.faces[face_indices]
        colors = vertex_colors[face_vertices].mean(axis=1)

    return points, colors


def computeBoundingBox(
    points: Union[torch.Tensor, np.ndarray],
    padding_ratio: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算点云的边界框

    Args:
        points: 点坐标 (N, 3)
        padding_ratio: 边界框的padding比例

    Returns:
        (min_bound, max_bound): 边界框的最小和最大坐标
    """
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)

    extent = max_bound - min_bound
    padding = extent * padding_ratio
    min_bound = min_bound - padding
    max_bound = max_bound + padding

    return min_bound, max_bound


def normalizePoints(
    points: Union[torch.Tensor, np.ndarray],
    center: Optional[Union[torch.Tensor, np.ndarray]] = None,
    scale: Optional[float] = None,
) -> Tuple[Union[torch.Tensor, np.ndarray], np.ndarray, float]:
    """
    归一化点云到单位立方体

    Args:
        points: 点坐标 (N, 3)
        center: 中心点，如果为None则使用点云中心
        scale: 缩放因子，如果为None则自动计算

    Returns:
        (normalized_points, center, scale): 归一化后的点、中心和缩放因子
    """
    is_tensor = isinstance(points, torch.Tensor)
    if is_tensor:
        device = points.device
        dtype = points.dtype
        points_np = points.cpu().numpy()
    else:
        points_np = points

    if center is None:
        center = (points_np.max(axis=0) + points_np.min(axis=0)) / 2
    elif isinstance(center, torch.Tensor):
        center = center.cpu().numpy()

    if scale is None:
        scale = (points_np.max(axis=0) - points_np.min(axis=0)).max()

    normalized = (points_np - center) / scale

    if is_tensor:
        normalized = torch.tensor(normalized, dtype=dtype, device=device)

    return normalized, center, scale
