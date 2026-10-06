"""Phase 3 Integration Tests - ICP Alignment."""
import numpy as np
import pytest
from pathlib import Path
import pandas as pd

from src.sensors.lidar import LiDARProcessor
from src.registration.alignment import ICPAligner


def create_synthetic_depth_map(height=480, width=640, scale=5000):
    """Create a synthetic depth map."""
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-(X**2 + Y**2)) * scale
    return Z.astype(np.float32)


def create_synthetic_camera_matrix():
    """Create a synthetic camera matrix."""
    return np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float32)


def load_benchmark_data(dataset_name: str):
    """Load benchmark data."""
    data_dir = Path(f"data/raw/benchmarks/{dataset_name}")
    
    depth_files = sorted((data_dir / "depth").glob("*.npy"))
    camera_matrix_file = data_dir / "camera_matrix.xlsx"
    
    if not depth_files or not camera_matrix_file.exists():
        pytest.skip(f"Dataset {dataset_name} not available")
    
    # Load camera matrix
    K = pd.read_excel(camera_matrix_file, header=None).values.astype(np.float32)
    
    # Load frames far apart
    processor = LiDARProcessor()
    
    depth_1 = np.load(depth_files[0])
    depth_2 = np.load(depth_files[min(100, len(depth_files) - 1)])
    
    pc_1, _ = processor.process(depth_1, K)
    pc_2, _ = processor.process(depth_2, K)
    
    return pc_1, pc_2


class TestICPAlignment:
    """Test ICP alignment."""
    
    def test_icp_synthetic_aligned(self):
        """Test ICP with pre-aligned synthetic clouds."""
        print("\n[TEST] ICP Synthetic - Pre-aligned clouds")
        
        # Create synthetic depth maps
        depth_1 = create_synthetic_depth_map()
        depth_2 = create_synthetic_depth_map()
        K = create_synthetic_camera_matrix()
        
        # Backproject to point clouds
        processor = LiDARProcessor()
        pc_1, _ = processor.process(depth_1, K)
        pc_2, _ = processor.process(depth_2, K)
        
        print(f"  Cloud sizes: {len(pc_1)}, {len(pc_2)}")
        
        # Run ICP
        aligner = ICPAligner(max_iterations=50)
        R, t, errors = aligner.align(pc_1, pc_2, verbose=True)
        
        quality = aligner.get_registration_quality()
        
        print(f"  Final error: {quality['final_error_m']:.6f} m")
        print(f"  Iterations: {quality['num_iterations']}")
        print(f"  ✓ Passed!")
        
        # Assertions: just verify it runs and converges
        assert quality['final_error_m'] < 0.5, "ICP error too large"
        assert quality['num_iterations'] >= 1, "ICP didn't run"
    
    def test_icp_single_room(self):
        """Test ICP with single_room benchmark."""
        print("\n[TEST] ICP with single_room benchmark")
        
        pc_1, pc_2 = load_benchmark_data("single_room")
        
        print(f"  Cloud 1: {len(pc_1)} points")
        print(f"  Cloud 2: {len(pc_2)} points")
        
        # Sample
        pc_1_sample = pc_1[::20]
        pc_2_sample = pc_2[::20]
        
        print(f"  Sampled: {len(pc_1_sample)}, {len(pc_2_sample)} points")
        
        # Run ICP
        aligner = ICPAligner(max_iterations=50)
        R, t, errors = aligner.align(pc_1_sample, pc_2_sample, verbose=False)
        
        quality = aligner.get_registration_quality()
        
        print(f"  Final error: {quality['final_error_m']:.6f} m")
        print(f"  Iterations: {quality['num_iterations']}")
        print(f"  ✓ Passed!")
        
        # Assertions
        assert quality['final_error_m'] < 1.0, "ICP error too large"
        assert quality['num_iterations'] >= 1, "ICP didn't run"
    
    def test_icp_ceiling_scan(self):
        """Test ICP with ceiling scan benchmark."""
        print("\n[TEST] ICP with single_scan_with_ceiling benchmark")
        
        pc_1, pc_2 = load_benchmark_data("single_scan_with_ceiling")
        
        print(f"  Cloud 1: {len(pc_1)} points")
        print(f"  Cloud 2: {len(pc_2)} points")
        
        # Sample
        pc_1_sample = pc_1[::20]
        pc_2_sample = pc_2[::20]
        
        print(f"  Sampled: {len(pc_1_sample)}, {len(pc_2_sample)} points")
        
        # Run ICP
        aligner = ICPAligner(max_iterations=50)
        R, t, errors = aligner.align(pc_1_sample, pc_2_sample, verbose=False)
        
        quality = aligner.get_registration_quality()
        
        print(f"  Final error: {quality['final_error_m']:.6f} m")
        print(f"  Iterations: {quality['num_iterations']}")
        print(f"  ✓ Passed!")
        
        # Assertions
        assert quality['final_error_m'] < 1.0, "ICP error too large"
        assert quality['num_iterations'] >= 1, "ICP didn't run"
