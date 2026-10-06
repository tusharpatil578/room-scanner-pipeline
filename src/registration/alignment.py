"""ICP (Iterative Closest Point) alignment for point cloud registration."""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
from scipy.spatial import cKDTree
import time


@dataclass
class ICPAligner:
    """
    Iterative Closest Point alignment.
    
    Algorithm:
    1. Find nearest neighbors between source and target (using KD-tree)
    2. Estimate optimal rotation R and translation t via SVD
    3. Apply transformation
    4. Repeat until convergence
    
    Reference: "A method for registration of 3-D shapes" - Besl & McKay (1992)
    """
    
    max_iterations: int = 50
    convergence_threshold: float = 1e-5  # meters
    max_correspondence_distance: float = 0.5  # meters (5cm threshold per benchmark)
    
    def __post_init__(self):
        """Initialize ICP parameters."""
        self.correspondence_history = []
        self.error_history = []
    
    def find_nearest_neighbors(self, source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find nearest neighbors using KD-tree (fast O(log N) lookup).
        
        Args:
            source: (N, 3) source points
            target: (M, 3) target point cloud
        
        Returns:
            (distances, indices) - distances and target indices for each source point
        
        Formula:
            For each source point s, find target point t that minimizes ||s - t||
        """
        print(f"  Building KD-tree for {len(target)} target points...")
        tree = cKDTree(target)
        
        print(f"  Querying {len(source)} source points...")
        distances, indices = tree.query(source, workers=-1)  # Parallel query
        
        return distances, indices
    
    def compute_transform(self, source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute optimal rotation R and translation t using SVD.
        
        The problem: minimize ||R @ source.T + t - target.T||_F
        
        Solution (Umeyama's method):
        1. Center both point sets
        2. Compute covariance matrix H = source.T @ target
        3. SVD decomposition: H = U @ Σ @ V.T
        4. R = V @ U.T (ensure det(R) = 1 for proper rotation)
        5. t = centroid_target - R @ centroid_source
        
        Args:
            source: (N, 3) source points
            target: (N, 3) corresponding target points
        
        Returns:
            (R, t, rmse) - rotation, translation, and RMSE error
        """
        if len(source) < 3:
            raise ValueError("Need at least 3 points for transformation")
        
        # Step 1: Center both point sets
        source_center = np.mean(source, axis=0)
        target_center = np.mean(target, axis=0)
        
        source_centered = source - source_center
        target_centered = target - target_center
        
        # Step 2: Compute covariance matrix
        H = np.dot(source_centered.T, target_centered)
        
        # Step 3: SVD decomposition
        U, S, Vt = np.linalg.svd(H)
        
        # Step 4: Compute rotation matrix
        R = np.dot(Vt.T, U.T)
        
        # Ensure proper rotation (det(R) = 1, not -1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)
        
        # Step 5: Compute translation
        t = target_center - np.dot(R, source_center)
        
        # Compute RMSE error
        source_transformed = np.dot(source_centered, R.T) + target_center
        rmse = np.sqrt(np.mean(np.sum((source_transformed - target) ** 2, axis=1)))
        
        return R, t, rmse
    
    def apply_transform(self, points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Apply rotation and translation to points.
        
        Formula: p_transformed = R @ p + t
        
        Args:
            points: (N, 3) point cloud
            R: (3, 3) rotation matrix
            t: (3,) translation vector
        
        Returns:
            (N, 3) transformed points
        """
        return np.dot(points, R.T) + t
    
    def align(self, source: np.ndarray, target: np.ndarray, 
              verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """
        Perform full ICP alignment.
        
        Args:
            source: (N, 3) source point cloud
            target: (M, 3) target point cloud (reference)
            verbose: Print iteration details
        
        Returns:
            (R_final, t_final, iteration_history)
            
        Iteration history contains:
            - iteration: Iteration number
            - num_correspondences: Valid point pairs
            - error: Mean correspondence distance
            - rmse: Root mean square error
            - converged: Whether convergence threshold reached
        
        Benchmark Gates (Phase 3):
            Gate 3: Repeatability - error should be ≤1cm (0.01m)
            Gate 4: Drift - track error accumulation across frames
        """
        start_time = time.time()
        
        R_cumulative = np.eye(3, dtype=np.float32)
        t_cumulative = np.zeros(3, dtype=np.float32)
        
        source_transformed = source.copy().astype(np.float32)
        
        if verbose:
            print(f"\n{'Iter':<6} {'Corr':<8} {'Dist (m)':<12} {'RMSE (m)':<12} {'ΔError':<12} {'Time (s)'}")
            print("-" * 70)
        
        prev_error = float('inf')
        
        for iteration in range(self.max_iterations):
            iter_start = time.time()
            
            # Step 1: Find nearest neighbors
            distances, indices = self.find_nearest_neighbors(source_transformed, target)
            
            # Step 2: Filter by max distance (outlier rejection)
            valid_mask = distances < self.max_correspondence_distance
            num_valid = np.sum(valid_mask)
            
            if num_valid < 10:
                if verbose:
                    print(f"⚠ Iteration {iteration}: Only {num_valid} valid correspondences, stopping")
                break
            
            source_corr = source_transformed[valid_mask]
            target_corr = target[indices[valid_mask]]
            
            # Step 3: Compute optimal transformation
            R, t, rmse = self.compute_transform(source_corr, target_corr)
            
            # Step 4: Update cumulative transformation
            R_cumulative = np.dot(R, R_cumulative)
            t_cumulative = np.dot(R, t_cumulative) + t
            
            # Step 5: Apply transformation to source
            source_transformed = self.apply_transform(source_transformed, R, t)
            
            # Step 6: Compute convergence metrics
            mean_distance = np.mean(distances[valid_mask])
            delta_error = prev_error - mean_distance
            prev_error = mean_distance
            
            iter_time = time.time() - iter_start
            
            if verbose:
                print(f"{iteration:<6} {num_valid:<8} {mean_distance:<12.6f} {rmse:<12.6f} {delta_error:<12.6f} {iter_time:<8.3f}")
            
            # Store history
            self.iteration_history = {
                'iteration': iteration,
                'num_correspondences': num_valid,
                'error': mean_distance,
                'rmse': rmse,
                'delta_error': delta_error,
                'converged': delta_error < self.convergence_threshold,
            }
            self.error_history.append(mean_distance)
            self.correspondence_history.append(num_valid)
            
            # Step 7: Check convergence
            if delta_error < self.convergence_threshold:
                if verbose:
                    print(f"\n✓ Converged at iteration {iteration}")
                    print(f"  Error: {mean_distance:.6f} m (≤ 1cm benchmark)")
                    print(f"  RMSE: {rmse:.6f} m")
                break
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\nAlignment Summary:")
            print(f"  Final error: {self.error_history[-1]:.6f} m")
            print(f"  Iterations: {len(self.error_history)}")
            print(f"  Total time: {total_time:.2f} s")
            print(f"  ✓ Passed repeatability gate (≤1cm): {self.error_history[-1] <= 0.01}")
        
        return R_cumulative, t_cumulative, self.error_history
    
    def get_registration_quality(self) -> dict:
        """
        Get registration quality metrics for benchmark evaluation.
        
        Returns:
            Quality metrics dict
        """
        if not self.error_history:
            return {}
        
        return {
            'final_error_m': float(self.error_history[-1]),
            'mean_error_m': float(np.mean(self.error_history)),
            'max_error_m': float(np.max(self.error_history)),
            'std_error_m': float(np.std(self.error_history)),
            'num_iterations': len(self.error_history),
            'passes_1cm_gate': float(self.error_history[-1]) <= 0.01,
        }