"""LiDAR depth processing and point cloud generation."""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path
import pandas as pd


@dataclass
class LiDARProcessor:
    """Process LiDAR depth maps to 3D point clouds."""
    
    def __init__(self, config=None):
        """Initialize LiDAR processor."""
        self.config = config
        self.camera_matrix = None
        self.camera_matrix_inv = None
    
    def load_camera_matrix(self, camera_matrix_path: Path) -> np.ndarray:
        """Load camera matrix from Excel file."""
        try:
            df = pd.read_excel(camera_matrix_path, header=None)
            self.camera_matrix = df.values.astype(np.float32)
            self.camera_matrix_inv = np.linalg.inv(self.camera_matrix)
            return self.camera_matrix
        except Exception as e:
            print(f"Error loading camera matrix: {e}")
            return None
    
    def backproject_depth(self, depth_map: np.ndarray, camera_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Backproject depth map to 3D point cloud using camera matrix.
        
        Formula: P = K^-1 @ [u, v, 1]^T * depth
        
        Args:
            depth_map: (H, W) depth values in meters
            camera_matrix: (3, 3) intrinsic camera matrix K
        
        Returns:
            (N, 3) point cloud in 3D space
        """
        if camera_matrix is not None:
            self.camera_matrix_inv = np.linalg.inv(camera_matrix)
        
        if self.camera_matrix_inv is None:
            raise ValueError("Camera matrix not initialized")
        
        H, W = depth_map.shape
        
        # Create pixel coordinate grid
        u = np.arange(W, dtype=np.float32)
        v = np.arange(H, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        
        # Stack into homogeneous coordinates [u, v, 1]
        ones = np.ones_like(uu)
        uv_homogeneous = np.stack([uu, vv, ones], axis=-1)  # (H, W, 3)
        
        # Apply inverse camera matrix
        # K^-1 @ [u, v, 1]^T gives direction vector
        directions = np.dot(uv_homogeneous, self.camera_matrix_inv.T)  # (H, W, 3)
        
        # Scale by depth
        depth_expanded = depth_map[:, :, np.newaxis]
        point_cloud = directions * depth_expanded  # (H, W, 3)
        
        # Reshape to (N, 3)
        point_cloud = point_cloud.reshape(-1, 3)
        
        return point_cloud.astype(np.float32)
    
    def remove_outliers(self, point_cloud: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """
        Remove outliers using 3-sigma rule.
        
        Args:
            point_cloud: (N, 3) point cloud
            threshold: standard deviations for outlier detection
        
        Returns:
            (M, 3) filtered point cloud
        """
        mean = np.mean(point_cloud, axis=0)
        std = np.std(point_cloud, axis=0)
        
        # Keep points within threshold * std
        mask = np.all(np.abs(point_cloud - mean) <= threshold * std, axis=1)
        filtered = point_cloud[mask]
        
        return filtered.astype(np.float32)
    
    def transform_to_world(self, point_cloud: np.ndarray, transform_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Transform point cloud to world frame.
        
        Args:
            point_cloud: (N, 3) in camera frame
            transform_matrix: (4, 4) transformation matrix
        
        Returns:
            (N, 3) in world frame
        """
        if transform_matrix is None:
            return point_cloud
        
        # Add homogeneous coordinate
        ones = np.ones((point_cloud.shape[0], 1))
        pc_homogeneous = np.hstack([point_cloud, ones])  # (N, 4)
        
        # Apply transformation
        pc_world = np.dot(pc_homogeneous, transform_matrix.T)  # (N, 4)
        
        # Remove homogeneous coordinate
        return pc_world[:, :3].astype(np.float32)
    
    def process(self, depth_map: np.ndarray, camera_matrix: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full LiDAR processing pipeline.
        
        Args:
            depth_map: (H, W) depth map
            camera_matrix: (3, 3) camera matrix
        
        Returns:
            (point_cloud, confidence) both (N,) or (N, 3)
        """
        # Backproject depth
        point_cloud = self.backproject_depth(depth_map, camera_matrix)
        
        # Remove outliers
        point_cloud = self.remove_outliers(point_cloud)
        
        # Compute confidence based on point density
        # Higher density = higher confidence
        density = len(point_cloud) / (depth_map.shape[0] * depth_map.shape[1])
        reference_density = 100  # points per meter^2
        confidence = np.minimum(1.0, density / reference_density)
        
        return point_cloud, np.full(len(point_cloud), confidence, dtype=np.float32)