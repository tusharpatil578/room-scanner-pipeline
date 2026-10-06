"""Measurement extraction from floor plans."""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class MeasurementExtractor:
    """Extract measurements from floor plan."""
    
    @staticmethod
    def compute_wall_length(wall_points: np.ndarray) -> float:
        """Compute wall length."""
        if len(wall_points) < 2:
            return 0.0
        
        # Project to 2D (x, y)
        wall_2d = wall_points[:, :2]
        
        # Compute convex hull perimeter
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(wall_2d)
            length = hull.volume  # In 2D, volume is perimeter
        except:
            length = np.linalg.norm(np.max(wall_2d, axis=0) - np.min(wall_2d, axis=0))
        
        return float(length)
    
    @staticmethod
    def compute_room_area(wall_points: np.ndarray) -> float:
        """Compute room area from walls."""
        if len(wall_points) < 3:
            return 0.0
        
        # Project to 2D
        wall_2d = wall_points[:, :2]
        
        # Compute convex hull area
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(wall_2d)
            area = hull.area
        except:
            x_range = np.max(wall_2d[:, 0]) - np.min(wall_2d[:, 0])
            y_range = np.max(wall_2d[:, 1]) - np.min(wall_2d[:, 1])
            area = x_range * y_range
        
        return float(area)
    
    @staticmethod
    def compute_ceiling_height(wall_points: np.ndarray) -> Tuple[float, float]:
        """
        Compute ceiling height and variance.
        
        Returns:
            (median_height, std_dev)
        """
        if len(wall_points) == 0:
            return 0.0, 0.0
        
        z_values = wall_points[:, 2]
        
        # Use 90th percentile to ignore noise
        height = np.percentile(z_values, 90)
        std_dev = np.std(z_values)
        
        return float(height), float(std_dev)
    
    @staticmethod
    def compute_opening_dimensions(opening: dict) -> dict:
        """Compute opening dimensions."""
        opening_dims = {
            'width': opening.get('width', 0.0),
            'type': opening.get('type', 'unknown'),
            'position': opening.get('position'),
        }
        return opening_dims
    
    @staticmethod
    def extract_measurements(floor_plan: dict, point_cloud: np.ndarray) -> dict:
        """
        Extract all measurements.
        
        Args:
            floor_plan: Floor plan dict
            point_cloud: (N, 3) point cloud
        
        Returns:
            Measurements dict
        """
        measurements = {
            'walls': [],
            'room_area': 0.0,
            'ceiling_height': 0.0,
            'ceiling_height_std': 0.0,
            'openings': [],
        }
        
        # Extract wall measurements
        for wall in floor_plan.get('walls', []):
            length = MeasurementExtractor.compute_wall_length(wall)
            measurements['walls'].append({
                'length': length,
                'num_points': len(wall),
            })
        
        # Extract room area
        if floor_plan.get('walls'):
            all_wall_points = np.vstack(floor_plan['walls'])
            measurements['room_area'] = MeasurementExtractor.compute_room_area(all_wall_points)
            
            # Extract ceiling height
            height, std = MeasurementExtractor.compute_ceiling_height(all_wall_points)
            measurements['ceiling_height'] = height
            measurements['ceiling_height_std'] = std
        
        # Extract opening measurements
        for opening in floor_plan.get('openings', []):
            measurements['openings'].append(
                MeasurementExtractor.compute_opening_dimensions(opening)
            )
        
        return measurements
    
    @staticmethod
    def compute_confidence_interval(measurements: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute measurement confidence interval.
        
        Returns:
            (mean, lower_95%, upper_95%)
        """
        mean = np.mean(measurements)
        std = np.std(measurements)
        margin = 1.96 * std / np.sqrt(len(measurements))
        
        return float(mean), float(mean - margin), float(mean + margin)