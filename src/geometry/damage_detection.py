"""Detect damage regions in point clouds."""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum


class DamageClass(Enum):
    """Damage classification."""
    NONE = "NONE"
    COSMETIC = "COSMETIC"
    STRUCTURAL = "STRUCTURAL"
    CONCEALED = "CONCEALED"


@dataclass
class DamageDetector:
    """Detect and classify damage."""
    
    @staticmethod
    def compute_normals(point_cloud: np.ndarray, k: int = 10) -> np.ndarray:
        """
        Compute point normals using PCA.
        
        Args:
            point_cloud: (N, 3) point cloud
            k: Number of neighbors for PCA
        
        Returns:
            (N, 3) normal vectors
        """
        from scipy.spatial import cKDTree
        
        tree = cKDTree(point_cloud)
        _, indices = tree.query(point_cloud, k=k+1)
        
        normals = np.zeros_like(point_cloud)
        
        for i, neighbor_indices in enumerate(indices):
            neighbors = point_cloud[neighbor_indices]
            
            # Center
            centered = neighbors - np.mean(neighbors, axis=0)
            
            # SVD
            U, S, Vt = np.linalg.svd(centered)
            normal = Vt[-1]  # Smallest singular vector
            
            normals[i] = normal / np.linalg.norm(normal)
        
        return normals
    
    @staticmethod
    def detect_normal_discontinuities(normals: np.ndarray, threshold: float = 0.3) -> np.ndarray:
        """
        Detect normal discontinuities (surface changes).
        
        Args:
            normals: (N, 3) normals
            threshold: Threshold for discontinuity
        
        Returns:
            (N,) binary mask of discontinuities
        """
        # Compute angle differences to neighbors
        discontinuities = np.zeros(len(normals), dtype=bool)
        
        from scipy.spatial import cKDTree
        tree = cKDTree(normals)
        _, indices = tree.query(normals, k=5)
        
        for i, neighbor_indices in enumerate(indices):
            angles = np.abs(np.dot(normals[neighbor_indices], normals[i]))
            if np.min(angles) < threshold:
                discontinuities[i] = True
        
        return discontinuities
    
    @staticmethod
    def region_growing(discontinuities: np.ndarray, point_cloud: np.ndarray, 
                      distance_threshold: float = 0.1) -> List[np.ndarray]:
        """
        Group discontinuities into regions.
        
        Args:
            discontinuities: (N,) binary mask
            point_cloud: (N, 3) point cloud
            distance_threshold: Grouping threshold
        
        Returns:
            List of damage regions
        """
        damage_indices = np.where(discontinuities)[0]
        
        if len(damage_indices) == 0:
            return []
        
        regions = []
        remaining = set(damage_indices)
        
        while remaining:
            seed = list(remaining)[0]
            region = {seed}
            queue = [seed]
            
            while queue:
                current = queue.pop(0)
                
                # Find neighbors within distance
                for idx in list(remaining - region):
                    if np.linalg.norm(point_cloud[current] - point_cloud[idx]) < distance_threshold:
                        region.add(idx)
                        queue.append(idx)
            
            regions.append(np.array(list(region)))
            remaining -= region
        
        return regions
    
    @staticmethod
    def classify_damage(region_points: np.ndarray, normals: np.ndarray) -> DamageClass:
        """
        Classify damage type.
        
        Args:
            region_points: Point cloud region
            normals: Normal vectors for region
        
        Returns:
            Damage classification
        """
        if len(region_points) < 10:
            return DamageClass.COSMETIC
        
        # Compute normal variance (smoothness)
        normal_std = np.std(normals, axis=0)
        smoothness = np.mean(normal_std)
        
        # Compute size
        size = np.linalg.norm(np.max(region_points, axis=0) - np.min(region_points, axis=0))
        
        if smoothness > 0.5:
            return DamageClass.STRUCTURAL
        elif size > 0.5:
            return DamageClass.COSMETIC
        else:
            return DamageClass.CONCEALED
    
    @staticmethod
    def detect_damage(point_cloud: np.ndarray) -> List[Dict]:
        """
        Full damage detection pipeline.
        
        Args:
            point_cloud: (N, 3) point cloud
        
        Returns:
            List of damage regions with classifications
        """
        # Compute normals
        normals = DamageDetector.compute_normals(point_cloud)
        
        # Detect discontinuities
        discontinuities = DamageDetector.detect_normal_discontinuities(normals)
        
        # Region growing
        regions = DamageDetector.region_growing(discontinuities, point_cloud)
        
        # Classify
        damages = []
        for region_indices in regions:
            region_points = point_cloud[region_indices]
            region_normals = normals[region_indices]
            
            damage_class = DamageDetector.classify_damage(region_points, region_normals)
            
            damages.append({
                'class': damage_class,
                'num_points': len(region_indices),
                'centroid': np.mean(region_points, axis=0),
                'size': np.linalg.norm(np.max(region_points, axis=0) - np.min(region_points, axis=0)),
            })
        
        return damages