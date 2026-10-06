"""Extract measurements from point clouds."""
import numpy as np
from scipy.spatial import ConvexHull


class MeasurementExtractor:
    """Extract room measurements."""
    
    def compute_wall_length(self, wall_points):
        """Compute wall perimeter using convex hull.
        
        Args:
            wall_points: N x 3 array
            
        Returns:
            Wall length in meters
        """
        if len(wall_points) < 3:
            return 0.0
        
        # Project to 2D
        wall_2d = wall_points[:, :2]
        
        try:
            hull = ConvexHull(wall_2d)
            
            # Compute actual perimeter
            hull_points = wall_2d[hull.vertices]
            perimeter = 0.0
            for i in range(len(hull_points)):
                p1 = hull_points[i]
                p2 = hull_points[(i + 1) % len(hull_points)]
                perimeter += np.linalg.norm(p2 - p1)
            
            return perimeter
        except:
            return 0.0
    
    def compute_room_area(self, floor_points):
        """Compute room area from floor points.
        
        Args:
            floor_points: N x 3 array
            
        Returns:
            Room area in square meters
        """
        if len(floor_points) < 3:
            return 0.0
        
        # Project to 2D
        floor_2d = floor_points[:, :2]
        
        try:
            hull = ConvexHull(floor_2d)
            return hull.volume  # In 2D, volume is area
        except:
            return 0.0
    
    def compute_ceiling_height(self, wall_points, floor_z=None):
        """Compute ceiling height (90th percentile of wall heights).
        
        Args:
            wall_points: N x 3 array
            floor_z: Floor z-level (auto-detected if None)
            
        Returns:
            Ceiling height in meters
        """
        if len(wall_points) < 10:
            return 0.0
        
        if floor_z is None:
            floor_z = np.min(wall_points[:, 2])
        
        # Height above floor
        heights = wall_points[:, 2] - floor_z
        
        # 90th percentile (robust to outliers)
        ceiling_height = np.percentile(heights, 90)
        
        return ceiling_height
    
    def compute_confidence_interval(self, measurements, confidence=0.95):
        """Compute confidence interval for measurements.
        
        Args:
            measurements: Array of measurements
            confidence: Confidence level (0.95 = 95%)
            
        Returns:
            Dict with mean, std, ci_lower, ci_upper
        """
        if len(measurements) < 2:
            return {
                'mean': measurements[0] if len(measurements) > 0 else 0.0,
                'std': 0.0,
                'ci_lower': measurements[0] if len(measurements) > 0 else 0.0,
                'ci_upper': measurements[0] if len(measurements) > 0 else 0.0
            }
        
        mean = np.mean(measurements)
        std = np.std(measurements)
        
        # 95% CI for normal distribution: mean ± 1.96*std/sqrt(n)
        z = 1.96 if confidence == 0.95 else 2.576
        ci_margin = z * std / np.sqrt(len(measurements))
        
        return {
            'mean': mean,
            'std': std,
            'ci_lower': mean - ci_margin,
            'ci_upper': mean + ci_margin,
            'confidence': confidence
        }
    
    def extract_measurements(self, floor_points, wall_points):
        """Extract all measurements.
        
        Args:
            floor_points: Floor point cloud
            wall_points: Wall point cloud
            
        Returns:
            Dict with all measurements
        """
        floor_z = np.min(floor_points[:, 2]) if len(floor_points) > 0 else 0.0
        
        measurements = {
            'wall_length': self.compute_wall_length(wall_points),
            'room_area': self.compute_room_area(floor_points),
            'ceiling_height': self.compute_ceiling_height(wall_points, floor_z),
            'floor_z': floor_z
        }
        
        return measurements
