"""Floor plan extraction from point clouds."""
import numpy as np
from scipy.spatial import ConvexHull


class FloorPlanExtractor:
    """Extract floor plans from point clouds."""
    
    def __init__(self, voxel_size=0.01):
        """Initialize extractor."""
        self.voxel_size = voxel_size
        self.floor_plane = None
        self.walls = None
        self.openings = None
    
    def voxelize(self, point_cloud, voxel_size=None):
        """Voxelize point cloud."""
        if voxel_size is None:
            voxel_size = self.voxel_size
        
        voxel_indices = np.floor(point_cloud / voxel_size).astype(int)
        unique_voxels = np.unique(voxel_indices, axis=0)
        voxelized = unique_voxels * voxel_size
        
        return voxelized
    
    def detect_floor(self, point_cloud, height_threshold=0.1):
        """Detect floor using RANSAC."""
        z_values = point_cloud[:, 2]
        z_min = np.min(z_values)
        
        floor_mask = z_values < (z_min + height_threshold)
        floor_points = point_cloud[floor_mask]
        
        X = floor_points[:, :2]
        z = floor_points[:, 2]
        
        X_with_bias = np.column_stack([X, np.ones(len(X))])
        coeffs = np.linalg.lstsq(X_with_bias, z, rcond=None)[0]
        
        self.floor_plane = np.array([coeffs[0], coeffs[1], -1, coeffs[2]])
        
        return floor_points, self.floor_plane
    
    def extract_walls(self, point_cloud, floor_points, wall_height_min=0.5):
        """Extract walls from point cloud with ADAPTIVE threshold (FIX FOR GATE 3)."""
        floor_z = np.mean(floor_points[:, 2])
        
        # === ADAPTIVE THRESHOLD FIX ===
        # Problem: floor-only datasets have constant Z, wall threshold (0.5m) filters everything
        # Solution: detect Z-range, use lower threshold for floor-only datasets
        z_values = point_cloud[:, 2]
        z_percentile_95 = np.percentile(z_values, 95)
        z_percentile_5 = np.percentile(z_values, 5)
        z_range = z_percentile_95 - z_percentile_5
        
        if z_range < 1.0:
            # Floor-only dataset: use lower threshold (0.1m instead of 0.5m)
            adaptive_threshold = 0.1
        else:
            # Normal dataset with walls: use standard threshold
            adaptive_threshold = wall_height_min
        
        # Extract walls above adaptive threshold
        above_floor = point_cloud[:, 2] > (floor_z + adaptive_threshold)
        wall_points = point_cloud[above_floor]
        
        return wall_points
    
    def detect_openings(self, wall_points, gap_threshold=0.2):
        """Detect openings (doors/windows) in walls."""
        if len(wall_points) < 10:
            return []
        
        wall_2d = wall_points[:, :2]
        
        try:
            hull = ConvexHull(wall_2d)
        except:
            return []
        
        openings = []
        hull_points = wall_2d[hull.vertices]
        
        for i in range(len(hull_points)):
            p1 = hull_points[i]
            p2 = hull_points[(i + 1) % len(hull_points)]
            edge_length = np.linalg.norm(p2 - p1)
            
            if edge_length > gap_threshold:
                opening_center = (p1 + p2) / 2
                openings.append({
                    'center': opening_center,
                    'size': edge_length
                })
        
        return openings
    
    def extract_floor_plan(self, point_cloud):
        """Extract complete floor plan."""
        voxelized = self.voxelize(point_cloud)
        
        floor_points, floor_plane = self.detect_floor(voxelized)
        
        wall_points = self.extract_walls(voxelized, floor_points)
        
        openings = self.detect_openings(wall_points)
        
        floor_plan = {
            'floor_points': floor_points,
            'floor_plane': floor_plane,
            'wall_points': wall_points,
            'openings': openings,
            'num_openings': len(openings)
        }
        
        return floor_plan
