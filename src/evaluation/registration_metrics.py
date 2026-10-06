"""Evaluate registration quality."""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class RegistrationMetrics:
    """Compute registration evaluation metrics."""
    
    @staticmethod
    def compute_rmse(source: np.ndarray, target: np.ndarray) -> float:
        """
        Compute RMSE between point clouds.
        
        Args:
            source: (N, 3) source points
            target: (N, 3) target points (corresponding)
        
        Returns:
            RMSE value
        """
        differences = source - target
        mse = np.mean(np.sum(differences ** 2, axis=1))
        return np.sqrt(mse)
    
    @staticmethod
    def compute_drift(poses: np.ndarray) -> float:
        """
        Compute accumulated drift in pose sequence.
        
        Args:
            poses: (N, 3) pose positions
        
        Returns:
            Total drift
        """
        if len(poses) < 2:
            return 0.0
        
        displacements = np.diff(poses, axis=0)
        cumulative_displacement = np.sum(np.linalg.norm(displacements, axis=1))
        
        return cumulative_displacement
    
    @staticmethod
    def compute_loop_closure_error(start_pose: np.ndarray, end_pose: np.ndarray) -> float:
        """
        Compute error when loop closes.
        
        Args:
            start_pose: (3,) starting position
            end_pose: (3,) ending position (should be close to start)
        
        Returns:
            Loop closure error
        """
        return np.linalg.norm(end_pose - start_pose)
    
    @staticmethod
    def evaluate_alignment(source: np.ndarray, target: np.ndarray, 
                          R: np.ndarray, t: np.ndarray) -> dict:
        """
        Evaluate alignment quality.
        
        Args:
            source: (N, 3) source points
            target: (M, 3) target points
            R: (3, 3) rotation
            t: (3,) translation
        
        Returns:
            Dictionary of metrics
        """
        # Transform source
        source_transformed = np.dot(source, R.T) + t
        
        # Find closest points
        from scipy.spatial import cKDTree
        tree = cKDTree(target)
        distances, _ = tree.query(source_transformed)
        
        metrics = {
            'mean_distance': float(np.mean(distances)),
            'median_distance': float(np.median(distances)),
            'std_distance': float(np.std(distances)),
            'max_distance': float(np.max(distances)),
            'rmse': float(RegistrationMetrics.compute_rmse(source_transformed[:len(target)], target[:len(source_transformed)])),
        }
        
        return metrics