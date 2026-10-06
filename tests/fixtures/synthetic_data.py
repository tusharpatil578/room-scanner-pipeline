"""Synthetic test data for unit testing."""

import numpy as np
from pathlib import Path


def create_synthetic_depth_map(height=480, width=640, noise_level=0.1) -> np.ndarray:
    """Create synthetic depth map for testing."""
    # Create synthetic depth with gradient
    depth = np.ones((height, width), dtype=np.float32) * 2.0
    
    # Add some structure
    depth[100:200, 100:300] = 3.0
    depth[250:400, 400:600] = 1.5
    
    # Add noise
    noise = np.random.normal(0, noise_level, depth.shape)
    depth = depth + noise
    depth = np.clip(depth, 0.1, 10.0)
    
    return depth


def create_synthetic_camera_matrix() -> np.ndarray:
    """Create synthetic camera intrinsics matrix."""
    focal_length = 500.0
    center_x = 320.0
    center_y = 240.0
    
    K = np.array([
        [focal_length, 0, center_x],
        [0, focal_length, center_y],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return K


def create_synthetic_point_cloud(n_points=1000) -> np.ndarray:
    """Create synthetic 3D point cloud."""
    # Generate random points in a room-like space
    x = np.random.uniform(-5, 5, n_points)
    y = np.random.uniform(-5, 5, n_points)
    z = np.random.uniform(0.1, 3, n_points)
    
    point_cloud = np.column_stack([x, y, z]).astype(np.float32)
    
    return point_cloud


def create_synthetic_image(height=480, width=640, channels=3) -> np.ndarray:
    """Create synthetic image for testing."""
    image = np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)
    return image


def create_test_fixtures() -> dict:
    """Create all test fixtures."""
    fixtures = {
        'depth_map': create_synthetic_depth_map(),
        'camera_matrix': create_synthetic_camera_matrix(),
        'point_cloud': create_synthetic_point_cloud(),
        'image': create_synthetic_image(),
    }
    return fixtures