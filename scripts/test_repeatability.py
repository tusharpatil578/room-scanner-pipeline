"""Test repeatability: same room, different frame pairs, should match within 1cm."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.sensors.lidar import LiDARProcessor
from src.registration.alignment import ICPAligner
from src.geometry.floor_plan import FloorPlanExtractor
from src.geometry.measurements import MeasurementExtractor


def measure_from_frames(depth_file_1, depth_file_2, camera_matrix_file):
    """Measure room from two depth frames."""
    # Load
    depth_1 = np.load(depth_file_1)
    depth_2 = np.load(depth_file_2)
    K = pd.read_excel(camera_matrix_file, header=None).values.astype(np.float32)
    
    # Process
    lidar = LiDARProcessor()
    pc_1, _ = lidar.process(depth_1, K)
    pc_2, _ = lidar.process(depth_2, K)
    combined = np.vstack([pc_1, pc_2])
    
    # Extract geometry
    floor_ext = FloorPlanExtractor()
    floor_plan = floor_ext.extract_floor_plan(combined)
    
    meas_ext = MeasurementExtractor()
    measurements = meas_ext.extract_measurements(
        floor_plan['floor_points'],
        floor_plan['wall_points']
    )
    
    return measurements


def test_repeatability(dataset_name):
    """Test repeatability: same room, different frame pairs."""
    print(f"\n{'='*80}")
    print(f"REPEATABILITY TEST: {dataset_name}")
    print(f"{'='*80}\n")
    
    data_dir = Path(f"data/raw/benchmarks/{dataset_name}")
    depth_files = sorted((data_dir / "depth").glob("*.npy"))
    K_file = data_dir / "camera_matrix.xlsx"
    
    if len(depth_files) < 150:
        print(f"⚠ Dataset has only {len(depth_files)} frames, need ≥150 for repeatability")
        return None
    
    # Run 1: Frames 0 & 100
    print("Run 1: Frames 0 & 100")
    meas_1 = measure_from_frames(depth_files[0], depth_files[100], K_file)
    print(f"  Area: {meas_1['room_area']:.1f} m²")
    print(f"  Perimeter: {meas_1['wall_length']:.1f} m")
    print(f"  Ceiling: {meas_1['ceiling_height']:.2f} m\n")
    
    # Run 2: Frames 50 & 150
    print("Run 2: Frames 50 & 150")
    meas_2 = measure_from_frames(depth_files[50], depth_files[150], K_file)
    print(f"  Area: {meas_2['room_area']:.1f} m²")
    print(f"  Perimeter: {meas_2['wall_length']:.1f} m")
    print(f"  Ceiling: {meas_2['ceiling_height']:.2f} m\n")
    
    # Compare
    print("Differences:")
    area_diff = abs(meas_1['room_area'] - meas_2['room_area'])
    perim_diff = abs(meas_1['wall_length'] - meas_2['wall_length'])
    ceiling_diff = abs(meas_1['ceiling_height'] - meas_2['ceiling_height'])
    
    print(f"  Area: {area_diff:.2f} m² ({(area_diff/max(meas_1['room_area'], 0.001))*100:.1f}%)")
    print(f"  Perimeter: {perim_diff:.2f} m ({(perim_diff/max(meas_1['wall_length'], 0.001))*100:.1f}%)")
    print(f"  Ceiling: {ceiling_diff:.4f} m = {ceiling_diff*100:.2f} cm")
    
    # Gate 3: within 1cm or 0.5% per wall
    passes_1cm = ceiling_diff < 0.01  # 1cm threshold
    passes_05pct = (perim_diff / max(meas_1['wall_length'], 0.001)) < 0.005  # 0.5%
    
    print(f"\n✅ Gate 3 Criteria:")
    print(f"  Within 1cm? {passes_1cm} (actual: {ceiling_diff*100:.2f} cm)")
    print(f"  Within 0.5% per wall? {passes_05pct} (actual: {(perim_diff/max(meas_1['wall_length'], 0.001))*100:.2f}%)")
    
    result = {
        'dataset': dataset_name,
        'run_1_area': meas_1['room_area'],
        'run_1_perimeter': meas_1['wall_length'],
        'run_1_ceiling': meas_1['ceiling_height'],
        'run_2_area': meas_2['room_area'],
        'run_2_perimeter': meas_2['wall_length'],
        'run_2_ceiling': meas_2['ceiling_height'],
        'area_diff_m2': area_diff,
        'perimeter_diff_m': perim_diff,
        'ceiling_diff_cm': ceiling_diff * 100,
        'passes_gate_3': passes_1cm and passes_05pct
    }
    
    return result


def main():
    print(f"\n{'#'*80}")
    print(f"# REPEATABILITY TEST - GATE 3 VALIDATION")
    print(f"{'#'*80}")
    
    datasets = ["single_room", "single_scan_floor_only", "single_scan_with_ceiling"]
    
    results = []
    for dataset in datasets:
        try:
            result = test_repeatability(dataset)
            if result:
                results.append(result)
        except Exception as e:
            print(f"❌ Error: {e}\n")
    
    # Summary
    if results:
        print(f"\n{'='*80}")
        print(f"REPEATABILITY SUMMARY")
        print(f"{'='*80}")
        
        df = pd.DataFrame(results)
        print(df[['dataset', 'ceiling_diff_cm', 'perimeter_diff_m', 'passes_gate_3']].to_string(index=False))
        
        passed = sum(1 for r in results if r['passes_gate_3'])
        print(f"\n✅ Gate 3 Results: {passed}/{len(results)} datasets pass repeatability")
        
        # Save
        report_path = Path("reports/repeatability_test.csv")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(report_path, index=False)
        print(f"✅ Report saved: {report_path}")


if __name__ == "__main__":
    main()
