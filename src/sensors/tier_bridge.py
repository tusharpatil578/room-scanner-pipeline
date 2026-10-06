"""Bridge to standardize outputs across all three sensor tiers."""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from src.constants import DataTier


@dataclass
class TierBridge:
    """Standardize outputs from different sensor tiers to common format."""
    
    @staticmethod
    def standardize_lidar(point_cloud: np.ndarray, confidence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Standardize LiDAR output.
        
        Args:
            point_cloud: (N, 3) LiDAR point cloud
            confidence: (N,) confidence scores
        
        Returns:
            (point_cloud, confidence) in standard format
        """
        # LiDAR accuracy: ±1-2cm
        accuracy = np.full(len(point_cloud), 0.015, dtype=np.float32)  # meters
        
        # Ensure point cloud is (N, 3) and float32
        point_cloud = np.asarray(point_cloud, dtype=np.float32).reshape(-1, 3)
        confidence = np.asarray(confidence, dtype=np.float32)
        
        return point_cloud, confidence
    
    @staticmethod
    def standardize_video(point_cloud: np.ndarray, confidence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Standardize Video (SfM) output.
        
        Args:
            point_cloud: (N, 3) video point cloud
            confidence: (N,) confidence scores
        
        Returns:
            (point_cloud, confidence) in standard format
        """
        # Video accuracy: ±3-5cm
        accuracy = np.full(len(point_cloud), 0.04, dtype=np.float32)  # meters
        
        # Ensure point cloud is (N, 3) and float32
        point_cloud = np.asarray(point_cloud, dtype=np.float32).reshape(-1, 3)
        confidence = np.asarray(confidence, dtype=np.float32)
        
        return point_cloud, confidence
    
    @staticmethod
    def standardize_photo(point_cloud: np.ndarray, confidence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Standardize Photo (stitching) output.
        
        Args:
            point_cloud: (N, 3) photo point cloud
            confidence: (N,) confidence scores
        
        Returns:
            (point_cloud, confidence) in standard format
        """
        # Photo accuracy: ±8-10% of dimension
        accuracy = np.full(len(point_cloud), 0.08, dtype=np.float32)  # relative
        
        # Ensure point cloud is (N, 3) and float32
        point_cloud = np.asarray(point_cloud, dtype=np.float32).reshape(-1, 3)
        confidence = np.asarray(confidence, dtype=np.float32)
        
        return point_cloud, confidence
    
    @staticmethod
    def bridge(tier: DataTier, point_cloud: np.ndarray, confidence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply tier-appropriate standardization.
        
        Args:
            tier: DataTier enum (LIDAR, VIDEO, PHOTO)
            point_cloud: (N, 3) point cloud
            confidence: (N,) confidence scores
        
        Returns:
            (point_cloud, confidence) standardized
        """
        if tier == DataTier.LIDAR:
            return TierBridge.standardize_lidar(point_cloud, confidence)
        elif tier == DataTier.VIDEO:
            return TierBridge.standardize_video(point_cloud, confidence)
        elif tier == DataTier.PHOTO:
            return TierBridge.standardize_photo(point_cloud, confidence)
        else:
            raise ValueError(f"Unknown tier: {tier}")
    
    @staticmethod
    def align_tiers(lidar_pc: Optional[np.ndarray], video_pc: Optional[np.ndarray], 
                    photo_pc: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Align point clouds from all three tiers to common coordinate system.
        
        Args:
            lidar_pc: LiDAR point cloud (reference frame)
            video_pc: Video point cloud
            photo_pc: Photo point cloud
        
        Returns:
            (aligned_lidar, aligned_video, aligned_photo)
        """
        # Use LiDAR as reference frame (highest accuracy)
        # In production, implement proper ICP alignment here
        
        return lidar_pc, video_pc, photo_pc