"""Unit tests for sensor processors."""

import numpy as np
import pytest
from pathlib import Path
from src.sensors.lidar import LiDARProcessor
from src.sensors.video import VideoProcessor
from src.sensors.photo import PhotoProcessor
from src.sensors.tier_bridge import TierBridge
from src.constants import DataTier
from tests.fixtures.synthetic_data import create_test_fixtures


class TestLiDARProcessor:
    """Test LiDAR processor."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = LiDARProcessor()
        self.fixtures = create_test_fixtures()
    
    def test_backproject_depth(self):
        """Test depth backprojection."""
        depth_map = self.fixtures['depth_map']
        camera_matrix = self.fixtures['camera_matrix']
        
        self.processor.load_camera_matrix(None)
        self.processor.camera_matrix_inv = np.linalg.inv(camera_matrix)
        
        point_cloud = self.processor.backproject_depth(depth_map, camera_matrix)
        
        assert point_cloud.shape[1] == 3
        assert point_cloud.dtype == np.float32
    
    def test_remove_outliers(self):
        """Test outlier removal."""
        point_cloud = self.fixtures['point_cloud']
        
        filtered = self.processor.remove_outliers(point_cloud, threshold=3.0)
        
        assert filtered.shape[1] == 3
        assert len(filtered) <= len(point_cloud)


class TestVideoProcessor:
    """Test Video processor."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = VideoProcessor()
    
    def test_detect_features(self):
        """Test feature detection."""
        image = create_test_fixtures()['image']
        
        keypoints, descriptors = self.processor.detect_features(image)
        
        if descriptors is not None:
            assert descriptors.dtype == np.uint8


class TestPhotoProcessor:
    """Test Photo processor."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = PhotoProcessor()
    
    def test_detect_features(self):
        """Test SIFT feature detection."""
        image = create_test_fixtures()['image']
        
        keypoints, descriptors = self.processor.detect_features(image)
        
        if descriptors is not None:
            assert descriptors.dtype == np.float32


class TestTierBridge:
    """Test tier bridging."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.fixtures = create_test_fixtures()
    
    def test_standardize_lidar(self):
        """Test LiDAR standardization."""
        pc = self.fixtures['point_cloud']
        conf = np.ones(len(pc))
        
        std_pc, std_conf = TierBridge.standardize_lidar(pc, conf)
        
        assert std_pc.shape == pc.shape
        assert std_pc.dtype == np.float32
    
    def test_bridge_all_tiers(self):
        """Test bridging all tiers."""
        pc = self.fixtures['point_cloud']
        conf = np.ones(len(pc))
        
        for tier in [DataTier.LIDAR, DataTier.VIDEO, DataTier.PHOTO]:
            std_pc, std_conf = TierBridge.bridge(tier, pc, conf)
            
            assert std_pc.dtype == np.float32
            assert len(std_pc) == len(pc)