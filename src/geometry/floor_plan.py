"""Floor plan extraction from point clouds."""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
from scipy.ndimage import label


@dataclass
class FloorPlanExtractor:
    """Extract 2D floor plan from 3D point cloud."""
    
    def __init__(self, voxel_size: float = 0.01):
        """
        Initialize extractor.
        
        Args:
            voxel_size: Size of voxel grid in meters (1cm default)
        """
        self.voxel_size = voxel_size
    
    def voxelize(self, point_cloud: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Voxelize point cloud.
        
        Args:
            point_cloud: (N, 3) point cloud
        
        Returns:
            (voxel_grid, voxel_info)
        """
        # Compute grid bounds
        min_coord = np.floor(np.min(point_cloud, axis=0) / self.voxel_size)
        max_coord = np.ceil(np.max(point_cloud, axis=0) / self.voxel_size)
        
        grid_size = (max_coord - min_coord).astype(int) + 1
        
        # Create voxel grid
        voxel_grid = np.zeros(grid_size, dtype=np.uint8)
        
        # Assign points to voxels
        indices = np.floor((point_cloud / self.voxel_size) - min_coord).astype(int)
        
        for idx in indices:
            if np.all(idx >= 0) and np.all(idx < grid_size):
                voxel_grid[tuple(idx)] = 1
        
        voxel_info = {
            'grid_size': grid_size,
            'min_coord': min_coord,
            'voxel_size': self.voxel_size,
        }
        
        return voxel_grid, voxel_info
    
    def detect_floor(self, point_cloud: np.ndarray, height_tolerance: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect floor plane using RANSAC.
        
        Args:
            point_cloud: (N, 3) point cloud
            height_tolerance: Tolerance for floor detection (meters)
        
        Returns:
            (floor_plane_normal, floor_plane_offset)
        """
        # Sort by z-coordinate and take bottom 30%
        z_sorted_indices = np.argsort(point_cloud[:, 2])
        bottom_indices = z_sorted_indices[:len(point_cloud) // 3]
        bottom_points = point_cloud[bottom_indices]
        
        # Fit plane using least squares
        # Ax + By + Cz = D
        A = np.column_stack([bottom_points[:, 0], bottom_points[:, 1], np.ones(len(bottom_points))])
        b = bottom_points[:, 2]
        
        try:
            coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
            normal = np.array([coeffs[0], coeffs[1], -1])
            normal = normal / np.linalg.norm(normal)
            offset = coeffs[2]
        except:
            # Fallback: assume horizontal floor
            normal = np.array([0, 0, 1])
            offset = np.median(point_cloud[:, 2])
        
        return normal, offset
    
    def extract_walls(self, point_cloud: np.ndarray, floor_normal: np.ndarray, 
                     floor_offset: float) -> List[np.ndarray]:
        """
        Extract vertical wall points.
        
        Args:
            point_cloud: (N, 3) point cloud
            floor_normal: (3,) floor plane normal
            floor_offset: Floor plane offset
        
        Returns:
            List of wall point clouds
        """
        # Remove floor points
        distances_to_floor = np.abs(np.dot(point_cloud, floor_normal) - floor_offset)
        non_floor_mask = distances_to_floor > 0.05
        
        wall_points = point_cloud[non_floor_mask]
        
        # Cluster into walls
        if len(wall_points) == 0:
            return []
        
        # Simple clustering: group by proximity
        walls = []
        remaining = list(range(len(wall_points)))
        
        while remaining:
            seed = wall_points[remaining[0]]
            cluster = [seed]
            
            for idx in remaining[1:]:
                if np.linalg.norm(wall_points[idx] - seed) < 1.0:
                    cluster.append(wall_points[idx])
            
            walls.append(np.array(cluster))
            remaining = [i for i in remaining if wall_points[i] not in cluster]
        
        return walls
    
    def detect_openings(self, walls: List[np.ndarray], min_gap: float = 0.2) -> List[dict]:
        """
        Detect openings (doors, windows) in walls.
        
        Args:
            walls: List of wall point clouds
            min_gap: Minimum gap size (meters)
        
        Returns:
            List of opening info dicts
        """
        openings = []
        
        for wall in walls:
            if len(wall) < 10:
                continue
            
            # Find gaps in wall
            x_sorted = wall[np.argsort(wall[:, 0])]
            x_diffs = np.diff(x_sorted[:, 0])
            
            for i, diff in enumerate(x_diffs):
                if diff > min_gap:
                    opening_info = {
                        'position': (x_sorted[i] + x_sorted[i+1]) / 2,
                        'width': diff,
                        'type': 'opening'
                    }
                    openings.append(opening_info)
        
        return openings
    
    def extract_floor_plan(self, point_cloud: np.ndarray) -> dict:
        """
        Extract complete floor plan.
        
        Args:
            point_cloud: (N, 3) point cloud
        
        Returns:
            Floor plan dict with walls, openings, etc.
        """
        # Detect floor
        floor_normal, floor_offset = self.detect_floor(point_cloud)
        
        # Extract walls
        walls = self.extract_walls(point_cloud, floor_normal, floor_offset)
        
        # Detect openings
        openings = self.detect_openings(walls)
        
        floor_plan = {
            'floor_normal': floor_normal,
            'floor_offset': floor_offset,
            'walls': walls,
            'openings': openings,
            'num_walls': len(walls),
            'num_openings': len(openings),
        }
        
        return floor_plan