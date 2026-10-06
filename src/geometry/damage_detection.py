"""Damage detection from point clouds."""
import numpy as np
from enum import Enum


class DamageClass(Enum):
    """Damage classification."""
    NONE = 0
    COSMETIC = 1
    STRUCTURAL = 2
    CONCEALED = 3


class DamageDetector:
    """Detect damage in surfaces."""
    
    def __init__(self, normal_threshold=0.3, cluster_size_min=50):
        """Initialize detector.
        
        Args:
            normal_threshold: Threshold for normal discontinuities
            cluster_size_min: Minimum cluster size for damage region
        """
        self.normal_threshold = normal_threshold
        self.cluster_size_min = cluster_size_min
    
    def compute_normals(self, point_cloud, k=10):
        """Compute surface normals using PCA.
        
        Args:
            point_cloud: N x 3 array
            k: Number of neighbors for PCA
            
        Returns:
            N x 3 array of normals
        """
        normals = np.zeros_like(point_cloud, dtype=np.float64)
        
        if len(point_cloud) < k:
            return normals
        
        # Simple normal computation
        for i in range(len(point_cloud)):
            # Find k nearest neighbors
            dists = np.linalg.norm(point_cloud - point_cloud[i], axis=1)
            neighbors_idx = np.argsort(dists)[:k+1]
            neighbors = point_cloud[neighbors_idx]
            
            # Center points
            centered = neighbors - np.mean(neighbors, axis=0)
            
            # PCA via SVD (more stable than eig)
            try:
                U, S, Vt = np.linalg.svd(centered.T @ centered)
                # Smallest singular vector is the normal
                normal = Vt[-1]
                normal = np.real(normal)  # Ensure real
                normals[i] = normal / (np.linalg.norm(normal) + 1e-6)
            except:
                normals[i] = np.array([0, 0, 1])
        
        return normals
    
    def detect_normal_discontinuities(self, normals, threshold=None):
        """Detect discontinuities in surface normals.
        
        Args:
            normals: N x 3 array
            threshold: Discontinuity threshold
            
        Returns:
            Boolean mask of discontinuities
        """
        if threshold is None:
            threshold = self.normal_threshold
        
        # Compute normal differences
        discontinuities = np.zeros(len(normals), dtype=bool)
        
        for i in range(1, len(normals)):
            dot_prod = np.abs(np.dot(normals[i], normals[i-1]))
            if dot_prod < (1 - threshold):
                discontinuities[i] = True
        
        return discontinuities
    
    def region_growing(self, point_cloud, discontinuities):
        """Grow regions from discontinuity points.
        
        Args:
            point_cloud: N x 3 array
            discontinuities: Boolean mask
            
        Returns:
            List of damage region indices
        """
        damage_regions = []
        visited = np.zeros(len(point_cloud), dtype=bool)
        
        for i in np.where(discontinuities)[0]:
            if visited[i]:
                continue
            
            region = [i]
            visited[i] = True
            queue = [i]
            
            # BFS
            while queue:
                current = queue.pop(0)
                
                # Find nearby discontinuity points
                dists = np.linalg.norm(point_cloud - point_cloud[current], axis=1)
                neighbors = np.where((dists < 0.1) & discontinuities & ~visited)[0]
                
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        region.append(neighbor)
                        visited[neighbor] = True
                        queue.append(neighbor)
            
            if len(region) >= self.cluster_size_min:
                damage_regions.append(region)
        
        return damage_regions
    
    def classify_damage(self, damage_regions, point_cloud):
        """Classify damage severity.
        
        Args:
            damage_regions: List of region indices
            point_cloud: Point cloud
            
        Returns:
            List of (region, classification) tuples
        """
        classifications = []
        
        for region in damage_regions:
            region_points = point_cloud[region]
            
            # Simple heuristic: based on cluster size and spread
            cluster_spread = np.max(np.linalg.norm(
                region_points - np.mean(region_points, axis=0),
                axis=1
            ))
            
            if len(region) < 100 or cluster_spread < 0.05:
                damage_class = DamageClass.COSMETIC
            elif cluster_spread > 0.2:
                damage_class = DamageClass.STRUCTURAL
            else:
                damage_class = DamageClass.CONCEALED
            
            classifications.append({
                'region': region,
                'class': damage_class,
                'severity': damage_class.value,
                'cluster_size': len(region),
                'spread': cluster_spread
            })
        
        return classifications
    
    def detect_damage(self, point_cloud):
        """Detect all damage.
        
        Args:
            point_cloud: N x 3 array
            
        Returns:
            List of damage detections with classifications
        """
        # Compute normals (sample for speed)
        sample_idx = np.random.choice(len(point_cloud), min(1000, len(point_cloud)), replace=False)
        sampled_pc = point_cloud[sample_idx]
        normals = self.compute_normals(sampled_pc, k=5)
        
        # Detect discontinuities
        discontinuities = self.detect_normal_discontinuities(normals)
        
        # Region growing
        damage_regions = self.region_growing(sampled_pc, discontinuities)
        
        # Classify
        classifications = self.classify_damage(damage_regions, sampled_pc)
        
        return {
            'num_damage_regions': len(classifications),
            'damage_detections': classifications,
            'discontinuity_count': np.sum(discontinuities)
        }
