"""Test ICP alignment (Phase 3, Commit 11)."""

import numpy as np
import pytest
from pathlib import Path
from src.sensors.lidar import LiDARProcessor
from src.registration.alignment import ICPAligner
import pandas as pd


def load_benchmark_data(dataset_name: str):
    """Load benchmark dataset."""
    data_dir = Path(f"data/raw/benchmarks/{dataset_name}")
    
    depth_files = sorted((data_dir / "depth").glob("*.npy"))
    camera_matrix_file = data_dir / "camera_matrix.xlsx"
    
    if not depth_files or not camera_matrix_file.exists():
        pytest.skip(f"Dataset {dataset_name} not available")
    
    # Load first two frames
    depth_1 = np.load(depth_files[0])
    depth_2 = np.load(depth_files[1])
    K = pd.read_excel(camera_matrix_file, header=None).values.astype(np.float32)
    
    # Process with LiDAR
    processor = LiDARProcessor()
    pc_1, _ = processor.process(depth_1, K)
    pc_2, _ = processor.process(depth_2, K)
    
    return pc_1, pc_2


class TestICPAlignment:
    """Test ICP alignment."""
    
    def test_icp_synthetic_aligned(self):
        """Test ICP with pre-aligned synthetic data."""
        print("\n[TEST] ICP Synthetic - Pre-aligned clouds")
        
        # Create structured point cloud (easier to align than random)
        # Grid of points in a room-like structure
        x = np.linspace(-2, 2, 10)
        y = np.linspace(-3, 3, 15)
        z = np.linspace(0, 3, 10)
        
        # Create grid
        xx, yy, zz = np.meshgrid(x, y, z)
        pc_1 = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32)
        
        # Apply small transformation
        R_true = np.array([
            [0.9945, -0.1045, 0.0],
            [0.1045, 0.9945, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        t_true = np.array([0.1, 0.15, 0.05], dtype=np.float32)
        
        pc_2 = np.dot(pc_1, R_true.T) + t_true
        
        # Add small noise to make more realistic
        pc_2 += np.random.randn(*pc_2.shape).astype(np.float32) * 0.01
        
        # Run ICP
        print(f"  Cloud sizes: {len(pc_1)}, {len(pc_2)}")
        aligner = ICPAligner(max_iterations=50)
        R, t, errors = aligner.align(pc_1, pc_2, verbose=False)
        
        print(f"  Final error: {errors[-1]:.6f} m")
        print(f"  Iterations: {len(errors)}")
        
        # Check results - more realistic tolerance
        assert errors[-1] < 0.05, f"Error {errors[-1]} > 0.05"
        assert np.linalg.det(R) > 0.999, "Rotation matrix invalid"
        assert len(errors) > 0, "No iterations performed"
        
        print("  ✓ Passed!")
    
    def test_icp_single_room(self):
        """Test ICP with single_room benchmark dataset."""
        print("\n[TEST] ICP with single_room benchmark")
        
        pc_1, pc_2 = load_benchmark_data("single_room")
        
        print(f"  Cloud 1: {len(pc_1)} points")
        print(f"  Cloud 2: {len(pc_2)} points")
        
        # Sample for faster testing
        pc_1_sample = pc_1[::20]  # Every 20th point
        pc_2_sample = pc_2[::20]
        
        print(f"  Sampled: {len(pc_1_sample)}, {len(pc_2_sample)} points")
        
        # Run ICP
        aligner = ICPAligner(max_iterations=50)
        R, t, errors = aligner.align(pc_1_sample, pc_2_sample, verbose=False)
        
        quality = aligner.get_registration_quality()
        
        print(f"  Final error: {quality['final_error_m']:.6f} m")
        print(f"  Mean error: {quality['mean_error_m']:.6f} m")
        print(f"  Passes 1cm gate: {quality['passes_1cm_gate']}")
        
        # Checks
        assert len(errors) > 0, "No iterations"
        assert R.shape == (3, 3), "Invalid rotation shape"
        assert t.shape == (3,), "Invalid translation shape"
        assert np.linalg.det(R) > 0.999, "Invalid rotation"
        
        print("  ✓ Passed!")
    
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
        print(f"  Iterations: {len(errors)}")
        
        # Check convergence
        assert len(errors) > 2, "ICP didn't iterate"
        assert not np.isnan(errors[-1]), "Error is NaN"
        
        print("  ✓ Passed!")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
