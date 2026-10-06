"""Loop closure detection and drift correction."""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from scipy.spatial import cKDTree


@dataclass
class LoopClosureDetector:
    """Detect and correct drift using loop closure."""
    
    def __init__(self, distance_threshold: float = 1.0, angle_threshold: float = 15.0):
        """
        Initialize loop closure detector.
        
        Args:
            distance_threshold: Max distance to detect loop (meters)
            angle_threshold: Max angle difference to detect loop (degrees)
        """
        self.distance_threshold = distance_threshold
        self.angle_threshold = np.radians(angle_threshold)
        self.poses = []  # List of (position, orientation)
    
    def detect_loop_closure(self, current_pose: Tuple[np.ndarray, np.ndarray]) -> Optional[int]:
        """
        Detect if current pose closes a loop with any previous pose.
        
        Args:
            current_pose: (position, R) current pose
        
        Returns:
            Index of closing pose or None
        """
        current_pos, current_R = current_pose
        
        for i, (prev_pos, prev_R) in enumerate(self.poses):
            # Check position distance
            distance = np.linalg.norm(current_pos - prev_pos)
            if distance > self.distance_threshold:
                continue
            
            # Check angle difference
            relative_R = np.dot(current_R, prev_R.T)
            angle = np.arccos(np.clip((np.trace(relative_R) - 1) / 2, -1, 1))
            
            if angle < self.angle_threshold:
                return i
        
        return None
    
    def add_pose(self, pose: Tuple[np.ndarray, np.ndarray]):
        """Add pose to history."""
        self.poses.append(pose)
    
    def optimize_pose_graph(self, poses: List[Tuple[np.ndarray, np.ndarray]], 
                           loop_constraints: List[Tuple[int, int]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Optimize pose graph using simple gradient descent.
        
        Args:
            poses: List of (position, R) poses
            loop_constraints: List of (i, j) loop closure pairs
        
        Returns:
            Optimized poses
        """
        optimized_poses = [pose for pose in poses]
        
        for _ in range(10):  # 10 optimization iterations
            for i, j in loop_constraints:
                pos_i, R_i = optimized_poses[i]
                pos_j, R_j = optimized_poses[j]
                
                # Simple adjustment: move both poses toward midpoint
                midpoint = (pos_i + pos_j) / 2
                correction = (midpoint - pos_i) * 0.1
                
                optimized_poses[i] = (pos_i + correction, R_i)
                optimized_poses[j] = (pos_j - correction, R_j)
        
        return optimized_poses
    
    def correct_drift(self, poses: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Correct drift using accumulated pose history.
        
        Args:
            poses: Sequence of poses with drift
        
        Returns:
            Corrected poses
        """
        corrected = []
        cumulative_error = 0
        
        for i, (pos, R) in enumerate(poses):
            # Accumulate error over sequence
            if i > 0:
                prev_pos = poses[i-1][0]
                drift = np.linalg.norm(pos - prev_pos) * 0.01  # 1% per step
                cumulative_error += drift
            
            # Apply inverse correction
            corrected_pos = pos - cumulative_error * np.array([1, 0, 0])
            corrected.append((corrected_pos, R))
        
        return corrected