"""End-to-end pipeline execution."""
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime

from src.sensors.lidar import LiDARProcessor
from src.registration.alignment import ICPAligner
from src.geometry.floor_plan import FloorPlanExtractor
from src.geometry.measurements import MeasurementExtractor
from src.geometry.damage_detection import DamageDetector
from src.geometry.scope_items import ScopeItemGenerator
from src.output.formatters import JSONFormatter
from src.output.gates import BenchmarkGates
from src.output.visualization import FloorPlanVisualizer


def run_pipeline(depth_file_1, depth_file_2, camera_matrix_file, output_dir="results"):
    """Run complete scanning pipeline.
    
    Args:
        depth_file_1: Path to first depth map
        depth_file_2: Path to second depth map
        camera_matrix_file: Path to camera matrix
        output_dir: Output directory
        
    Returns:
        Pipeline results dict
    """
    print("🚀 Starting Room Scanner Pipeline...\n")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    print("📦 Loading sensor data...")
    depth_1 = np.load(depth_file_1)
    depth_2 = np.load(depth_file_2)
    
    import pandas as pd
    K = pd.read_excel(camera_matrix_file, header=None).values.astype(np.float32)
    print(f"  ✓ Loaded 2 depth maps and camera matrix\n")
    
    # 2. Process LiDAR
    print("🔄 Processing LiDAR data...")
    lidar_processor = LiDARProcessor()
    pc_1, _ = lidar_processor.process(depth_1, K)
    pc_2, _ = lidar_processor.process(depth_2, K)
    print(f"  ✓ Generated {len(pc_1):,} + {len(pc_2):,} point clouds\n")
    
    # 3. Registration (ICP)
    print("🔗 Registering point clouds...")
    aligner = ICPAligner(max_iterations=50)
    pc_1_sample = pc_1[::20]
    pc_2_sample = pc_2[::20]
    R, t, errors = aligner.align(pc_1_sample, pc_2_sample, verbose=False)
    registration_quality = aligner.get_registration_quality()
    print(f"  ✓ ICP: error={registration_quality['final_error_m']:.6f}m, "
          f"iterations={registration_quality['num_iterations']}\n")
    
    # 4. Geometry extraction
    print("🏗️  Extracting geometry...")
    
    # Merge point clouds
    combined_pc = np.vstack([pc_1, pc_2])
    
    # Floor plan
    floor_extractor = FloorPlanExtractor()
    floor_plan = floor_extractor.extract_floor_plan(combined_pc)
    print(f"  ✓ Detected floor and {floor_plan['num_openings']} openings")
    
    # Measurements
    measurement_extractor = MeasurementExtractor()
    measurements = measurement_extractor.extract_measurements(
        floor_plan['floor_points'],
        floor_plan['wall_points']
    )
    print(f"  ✓ Measurements: Area={measurements['room_area']:.1f}m², "
          f"Height={measurements['ceiling_height']:.1f}m\n")
    
    # 5. Damage detection
    print("🔍 Detecting damage...")
    damage_detector = DamageDetector()
    damage_detections = damage_detector.detect_damage(combined_pc)
    print(f"  ✓ Detected {damage_detections['num_damage_regions']} damage regions\n")
    
    # 6. Scope generation
    print("📋 Generating scope of work...")
    scope_generator = ScopeItemGenerator()
    scope = scope_generator.generate_scope(measurements, damage_detections)
    print(f"  ✓ Scope: {scope['item_count']} items, "
          f"${scope['total_cost']:,.0f} estimated cost\n")
    
    # 7. Format output
    print("💾 Formatting output...")
    scan_id = f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # JSON
    result_dict = JSONFormatter.format_scan_result(
        scan_id, floor_plan, measurements, damage_detections, scope, registration_quality
    )
    json_path = output_dir / f"{scan_id}.json"
    JSONFormatter.save_json(result_dict, json_path)
    print(f"  ✓ JSON output: {json_path}")
    
    # Visualization
    visualizer = FloorPlanVisualizer()
    floor_plan_img = visualizer.visualize_floor_plan(floor_plan, measurements)
    img_path = output_dir / f"{scan_id}_floor_plan.png"
    visualizer.save_floor_plan(floor_plan_img, img_path)
    print(f"  ✓ Floor plan image: {img_path}\n")
    
    # 8. Evaluate benchmark gates
    print("✅ Evaluating benchmark gates...")
    gates_result = BenchmarkGates.evaluate_all_gates(
        floor_plan, measurements, damage_detections, registration_quality
    )
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK GATE RESULTS")
    print(f"{'='*60}")
    for gate in gates_result['gates']:
        status = "✅ PASS" if gate['passed'] else "❌ FAIL"
        print(f"Gate {gate['gate']}: {gate['name']:<20} {status}")
    
    print(f"\nScore: {gates_result['summary']['passed']}/{gates_result['summary']['total']} "
          f"({gates_result['summary']['score_pct']:.0f}%)")
    print(f"{'='*60}\n")
    
    return {
        'scan_id': scan_id,
        'floor_plan': floor_plan,
        'measurements': measurements,
        'damage': damage_detections,
        'scope': scope,
        'registration': registration_quality,
        'gates': gates_result,
        'output_dir': str(output_dir)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Room Scanner Pipeline')
    parser.add_argument('depth_1', help='Path to first depth map (NPY)')
    parser.add_argument('depth_2', help='Path to second depth map (NPY)')
    parser.add_argument('camera_matrix', help='Path to camera matrix (XLSX)')
    parser.add_argument('--output', default='results', help='Output directory')
    
    args = parser.parse_args()
    
    result = run_pipeline(args.depth_1, args.depth_2, args.camera_matrix, args.output)
    print(f"✨ Pipeline complete! Results in {result['output_dir']}")
