"""Validate pipeline on benchmark datasets using simulated ground truth."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime

from src.sensors.lidar import LiDARProcessor
from src.registration.alignment import ICPAligner
from src.geometry.floor_plan import FloorPlanExtractor
from src.geometry.measurements import MeasurementExtractor
from src.geometry.damage_detection import DamageDetector
from src.geometry.scope_items import ScopeItemGenerator
from src.output.gates import BenchmarkGates


def validate_dataset(dataset_name, ground_truth=None):
    """Validate on single dataset."""
    print(f"\n{'='*80}")
    print(f"VALIDATING: {dataset_name}")
    print(f"{'='*80}\n")
    
    data_dir = Path(f"data/raw/benchmarks/{dataset_name}")
    
    if not data_dir.exists():
        print(f"❌ Dataset not found: {data_dir}")
        return None
    
    # Load data
    depth_files = sorted((data_dir / "depth").glob("*.npy"))
    K_file = data_dir / "camera_matrix.xlsx"
    
    if len(depth_files) < 2:
        print(f"❌ Not enough depth frames: {len(depth_files)}")
        return None
    
    print(f"✓ Found {len(depth_files)} depth frames")
    
    # Load camera matrix
    K = pd.read_excel(K_file, header=None).values.astype(np.float32)
    print(f"✓ Loaded camera matrix K: {K.shape}")
    
    # Process LiDAR
    print(f"\n📦 Processing LiDAR...")
    lidar = LiDARProcessor()
    depth_1 = np.load(depth_files[0])
    depth_2 = np.load(depth_files[min(100, len(depth_files) - 1)])
    
    pc_1, _ = lidar.process(depth_1, K)
    pc_2, _ = lidar.process(depth_2, K)
    print(f"✓ Point clouds: {len(pc_1):,} + {len(pc_2):,} points")
    
    # Registration
    print(f"\n🔗 Registration (ICP)...")
    aligner = ICPAligner(max_iterations=50)
    pc_1s = pc_1[::20]
    pc_2s = pc_2[::20]
    R, t, errors = aligner.align(pc_1s, pc_2s, verbose=False)
    quality = aligner.get_registration_quality()
    print(f"✓ ICP error: {quality['final_error_m']:.6f}m ({quality['final_error_m']*100:.4f}cm)")
    print(f"✓ Iterations: {quality['num_iterations']}")
    
    # Geometry
    print(f"\n🏗️  Geometry Extraction...")
    combined = np.vstack([pc_1, pc_2])
    
    floor_ext = FloorPlanExtractor()
    floor_plan = floor_ext.extract_floor_plan(combined)
    print(f"✓ Floor points: {len(floor_plan['floor_points']):,}")
    print(f"✓ Wall points: {len(floor_plan['wall_points']):,}")
    print(f"✓ Openings detected: {floor_plan['num_openings']}")
    
    # Measurements
    meas_ext = MeasurementExtractor()
    measurements = meas_ext.extract_measurements(
        floor_plan['floor_points'],
        floor_plan['wall_points']
    )
    print(f"✓ Room area: {measurements['room_area']:.1f} m²")
    print(f"✓ Wall length: {measurements['wall_length']:.1f} m")
    print(f"✓ Ceiling height: {measurements['ceiling_height']:.2f} m")
    
    # Damage
    damage_det = DamageDetector()
    damage = damage_det.detect_damage(combined)
    print(f"✓ Damage regions: {damage['num_damage_regions']}")
    
    # Scope
    scope_gen = ScopeItemGenerator()
    scope = scope_gen.generate_scope(measurements, damage)
    print(f"✓ Scope cost: ${scope['total_cost']:,.0f}")
    
    # Gates
    print(f"\n✅ BENCHMARK GATE RESULTS")
    print(f"{'='*80}")
    gates = BenchmarkGates.evaluate_all_gates(floor_plan, measurements, damage, quality)
    
    results = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'frames': len(depth_files),
        'registration_error_m': float(quality['final_error_m']),
        'registration_error_cm': float(quality['final_error_m'] * 100),
        'room_area_m2': float(measurements['room_area']),
        'wall_length_m': float(measurements['wall_length']),
        'ceiling_height_m': float(measurements['ceiling_height']),
        'gates_passed': gates['summary']['passed'],
        'gates_total': gates['summary']['total'],
        'gates_score_pct': gates['summary']['score_pct'],
        'damage_regions': damage['num_damage_regions'],
        'scope_cost': float(scope['total_cost'])
    }
    
    for i, gate in enumerate(gates['gates']):
        status = "✅ PASS" if gate['passed'] else "❌ FAIL"
        print(f"Gate {gate['gate']}: {gate['name']:<25} {status}")
    
    print(f"\n📊 Score: {results['gates_passed']}/{results['gates_total']} ({results['gates_score_pct']:.0f}%)")
    print(f"{'='*80}\n")
    
    return results


def main():
    """Validate all benchmark datasets."""
    print(f"\n{'#'*80}")
    print(f"# ROOM SCANNER PIPELINE - BENCHMARK VALIDATION")
    print(f"# Using real data: data/raw/benchmarks/")
    print(f"{'#'*80}\n")
    
    datasets = ["single_room", "single_scan_floor_only", "single_scan_with_ceiling"]
    
    all_results = []
    
    for dataset in datasets:
        result = validate_dataset(dataset)
        if result:
            all_results.append(result)
    
    # Summary report
    if all_results:
        print(f"\n{'='*80}")
        print(f"SUMMARY REPORT")
        print(f"{'='*80}")
        
        df = pd.DataFrame(all_results)
        print(df.to_string(index=False))
        
        # Save report
        report_path = Path("reports/benchmark_validation.csv")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(report_path, index=False)
        print(f"\n✅ Report saved: {report_path}")
        
        # Overall stats
        print(f"\n📈 OVERALL STATISTICS")
        print(f"{'='*80}")
        print(f"Avg registration error: {df['registration_error_cm'].mean():.4f} cm")
        print(f"Avg room area: {df['room_area_m2'].mean():.1f} m²")
        print(f"Avg gates passed: {df['gates_passed'].mean():.1f}/{df['gates_total'].mean():.0f}")
        print(f"Avg score: {df['gates_score_pct'].mean():.1f}%")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
