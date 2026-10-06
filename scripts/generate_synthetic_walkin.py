"""Generate realistic synthetic depth maps for walk-in test."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from PIL import Image

def generate_room_geometry(room_length=4.0, room_width=3.0, room_height=2.7):
    """Generate 3D points for a realistic rectangular room."""
    points = []
    
    # Floor (XY plane at Z=0)
    floor_x = np.linspace(0, room_length, 100)
    floor_y = np.linspace(0, room_width, 100)
    for x in floor_x:
        for y in floor_y:
            points.append([x, y, 0.0])
    
    # Ceiling (XY plane at Z=room_height)
    for x in floor_x:
        for y in floor_y:
            points.append([x, y, room_height])
    
    # Wall 1 (X-Z plane at Y=0)
    wall_x = np.linspace(0, room_length, 50)
    wall_z = np.linspace(0, room_height, 50)
    for x in wall_x:
        for z in wall_z:
            points.append([x, 0.0, z])
    
    # Wall 2 (X-Z plane at Y=room_width)
    for x in wall_x:
        for z in wall_z:
            points.append([x, room_width, z])
    
    # Wall 3 (Y-Z plane at X=0)
    wall_y = np.linspace(0, room_width, 50)
    for y in wall_y:
        for z in wall_z:
            points.append([y, 0.0, z])
    
    # Wall 4 (Y-Z plane at X=room_length)
    for y in wall_y:
        for z in wall_z:
            points.append([y, room_length, z])
    
    # Door opening (1m x 2.1m) on wall at X=room_length
    door_x = room_length
    door_y_start = 0.5
    door_y_end = 1.5
    door_z_start = 0.6
    door_z_end = 2.7
    
    door_y = np.linspace(door_y_start, door_y_end, 20)
    door_z = np.linspace(door_z_start, door_z_end, 20)
    for y in door_y:
        for z in door_z:
            if not (y <= door_y_end and z >= door_z_start):
                points.append([door_x, y, z])
    
    return np.array(points, dtype=np.float32)

def project_to_depth_map(points_3d, K, image_width=1920, image_height=1080):
    """Project 3D points to depth map using camera intrinsics."""
    depth_map = np.zeros((image_height, image_width), dtype=np.uint16)
    
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    for point in points_3d:
        x, y, z = point
        
        if x <= 0:
            continue
        
        u = int(fx * y / x + cx)
        v = int(fy * z / x + cy)
        
        if 0 <= u < image_width and 0 <= v < image_height:
            depth = int(x * 1000)
            if depth < 65535:
                depth_map[v, u] = max(depth_map[v, u], depth)
    
    noise = np.random.normal(0, 20, depth_map.shape)
    depth_map = np.maximum(depth_map + noise, 0).astype(np.uint16)
    
    return depth_map

def main():
    print("\n" + "="*80)
    print("GENERATING SYNTHETIC WALK-IN TEST DATA")
    print("="*80 + "\n")
    
    data_dir = Path("data/raw/captures/synthetic_walkin/depth")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("1️⃣  Generating room geometry (4m x 3m x 2.7m)...")
    points_3d = generate_room_geometry(room_length=4.0, room_width=3.0, room_height=2.7)
    print(f"   Generated {len(points_3d):,} 3D points")
    
    print("\n2️⃣  Setting camera intrinsics (iPhone 15 Pro)...")
    K = np.array([
        [1200, 0, 960],
        [0, 1200, 540],
        [0, 0, 1]
    ], dtype=np.float32)
    
    pd.DataFrame(K).to_excel(
        Path("data/raw/captures/synthetic_walkin/camera_matrix.xlsx"),
        index=False, header=False
    )
    print(f"   K matrix: {K[0, 0]:.0f}, {K[1, 1]:.0f}, ({K[0, 2]:.0f}, {K[1, 2]:.0f})")
    
    print("\n3️⃣  Generating depth maps from 3 camera viewpoints...")
    
    viewpoints = [
        {"name": "center_view", "offset": [2.0, 1.5, 1.3]},
        {"name": "corner_view", "offset": [1.0, 0.5, 1.3]},
        {"name": "door_view", "offset": [3.5, 1.5, 1.3]},
    ]
    
    for i, vp in enumerate(viewpoints, 1):
        cam_offset = np.array(vp["offset"])
        points_cam = points_3d - cam_offset
        
        valid = points_cam[:, 0] > 0.1
        points_cam = points_cam[valid]
        
        depth_map = project_to_depth_map(points_cam, K)
        
        np.save(data_dir / f"depth_{i:06d}.npy", depth_map)
        
        depth_png = (depth_map / depth_map.max() * 255).astype(np.uint8)
        Image.fromarray(depth_png).save(data_dir / f"depth_{i:06d}.png")
        
        print(f"   View {i} ({vp['name']}): depth range {depth_map.min()}-{depth_map.max()} mm")
    
    print("\n4️⃣  Creating ground truth measurements...")
    ground_truth = pd.DataFrame({
        'measurement': [
            'room_area',
            'wall_length_1',
            'wall_length_2',
            'wall_length_3',
            'wall_length_4',
            'ceiling_height',
            'door_width',
            'door_height'
        ],
        'value_m': [
            4.0 * 3.0,
            4.0,
            4.0,
            3.0,
            3.0,
            2.7,
            1.0,
            2.1
        ],
        'method': [
            'synthetic',
            'synthetic',
            'synthetic',
            'synthetic',
            'synthetic',
            'synthetic',
            'synthetic',
            'synthetic'
        ]
    })
    
    gt_file = Path("data/raw/captures/synthetic_walkin/ground_truth.csv")
    ground_truth.to_csv(gt_file, index=False)
    print(f"   Saved ground truth to {gt_file}")
    print(f"   Room: 4.0m x 3.0m x 2.7m, Area: 12.0 m²")
    print(f"   Door: 1.0m wide x 2.1m tall")
    
    print("\n" + "="*80)
    print("✅ SYNTHETIC WALK-IN TEST DATA READY")
    print("="*80)
    print(f"\nFiles created:")
    print(f"  data/raw/captures/synthetic_walkin/depth/depth_*.npy (3 frames)")
    print(f"  data/raw/captures/synthetic_walkin/camera_matrix.xlsx")
    print(f"  data/raw/captures/synthetic_walkin/ground_truth.csv\n")

if __name__ == "__main__":
    main()
