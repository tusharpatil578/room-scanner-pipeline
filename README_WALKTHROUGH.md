# Room Scanner - Walk-In Test (Fresh Capture)

## What is Walk-In Test?
Capture a REAL room with your iPhone, run the pipeline, measure accuracy against ground truth (tape measure).

**Time required:** 15 minutes total
**Score value:** 30% of final grade

---

## PART 1: CAPTURE (5 minutes)

### 1.1 Prerequisites
- iPhone Pro (15+) with LiDAR scanner
- iPhone app: "LiDAR Scanner" (free, App Store)
- Tape measure (for ground truth)
- Choose a small room (bedroom, bathroom, hallway)

### 1.2 Capture Steps

1. **Open LiDAR Scanner app**
2. **Position**: Stand in room center
3. **Scan**: Slowly pan 360° around room
   - Cover all 4 walls
   - Scan floor and ceiling
   - Take 2-3 minutes total
4. **Export**: Save as "RoomName_LiDAR.USDZ"
5. **Extract depth map**: Use app's export → PNG depth
6. **Get intrinsics**: Use default iPhone 15 Pro K matrix

### 1.3 Extract Data
```bash
# Convert exported depth PNG to NPY format
python3 << 'CONVERT'
from PIL import Image
import numpy as np

# Load exported depth map (PNG from app)
depth_png = Image.open("RoomName_LiDAR_depth.png")
depth_array = np.array(depth_png, dtype=np.uint16)

# Save as NPY
np.save("depth_001.npy", depth_array)
np.save("depth_100.npy", depth_array)  # Same frame twice for now

# Create intrinsics (iPhone 15 Pro default)
import pandas as pd
K = np.array([
    [1200, 0, 960],      # fx, 0, cx
    [0, 1200, 540],      # 0, fy, cy
    [0, 0, 1]            # 0, 0, 1
], dtype=np.float32)
pd.DataFrame(K).to_excel("camera_matrix.xlsx", index=False, header=False)
CONVERT
```

---

## PART 2: RUN PIPELINE (2 minutes)

```bash
cd /workspaces/room-scanner-pipeline

# Create capture directory
mkdir -p data/raw/captures/walkin_test/depth

# Copy files
cp depth_001.npy data/raw/captures/walkin_test/depth/
cp depth_100.npy data/raw/captures/walkin_test/depth/
cp camera_matrix.xlsx data/raw/captures/walkin_test/

# Run pipeline
PYTHONPATH=/workspaces/room-scanner-pipeline python scripts/run_pipeline.py \
  data/raw/captures/walkin_test/depth/depth_001.npy \
  data/raw/captures/walkin_test/depth/depth_100.npy \
  data/raw/captures/walkin_test/camera_matrix.xlsx \
  --output results/walkin_test

# View results
cat results/walkin_test/scan_*.json
```

---

## PART 3: GROUND TRUTH (3 minutes)

Manually measure the room:

```bash
# Create ground truth file
cat > data/raw/captures/walkin_test/ground_truth.csv << 'GT'
measurement,value_m,method,notes
room_area,8.5,tape_measure,length x width
wall_length_north,4.2,tape_measure,
wall_length_south,4.2,tape_measure,
wall_length_east,2.5,tape_measure,
wall_length_west,2.5,tape_measure,
ceiling_height,2.65,tape_measure,floor to ceiling
door_width,0.9,tape_measure,main entry
GT

cat data/raw/captures/walkin_test/ground_truth.csv
```

---

## PART 4: MEASURE ACCURACY (5 minutes)

```bash
cat > scripts/validate_walkin.py << 'VAL'
import json
import pandas as pd
import numpy as np

# Load predictions
with open("results/walkin_test/scan_20261006_*.json") as f:
    pred = json.load(f)

# Load ground truth
gt = pd.read_csv("data/raw/captures/walkin_test/ground_truth.csv", index_col="measurement")

# Compare
results = {}
results["area_error_pct"] = abs(pred["room_area_m2"] - gt.loc["room_area", "value_m"]) / gt.loc["room_area", "value_m"] * 100
results["ceiling_error_cm"] = abs(pred["ceiling_height_m"] - gt.loc["ceiling_height", "value_m"]) * 100

print("\n" + "="*80)
print("WALK-IN TEST RESULTS")
print("="*80)
print(f"Room Area Error: {results['area_error_pct']:.1f}%")
print(f"Ceiling Height Error: {results['ceiling_error_cm']:.1f} cm")
print(f"Door Width: {pred['openings'][0]['size']:.2f}m (expected: 0.9m)")
print("="*80)

# Gates
area_pass = results["area_error_pct"] < 5  # Must be within 5%
ceiling_pass = results["ceiling_error_cm"] < 2  # Must be within 2cm
door_pass = abs(pred['openings'][0]['size'] - 0.9) < 0.1  # Within 10cm

print(f"\n✅ GATES:")
print(f"  Area: {'PASS' if area_pass else 'FAIL'}")
print(f"  Ceiling: {'PASS' if ceiling_pass else 'FAIL'}")
print(f"  Door: {'PASS' if door_pass else 'FAIL'}")
print(f"\nOVERALL: {'✅ WALK-IN TEST PASSED' if all([area_pass, ceiling_pass, door_pass]) else '❌ WALK-IN TEST FAILED'}")
VAL

PYTHONPATH=/workspaces/room-scanner-pipeline python scripts/validate_walkin.py
```

---

## WALK-IN TEST CHECKLIST

- [ ] Captured room with LiDAR Scanner app
- [ ] Exported depth map and intrinsics
- [ ] Converted PNG to NPY format
- [ ] Placed files in `data/raw/captures/walkin_test/`
- [ ] Ran pipeline (2 min)
- [ ] Measured ground truth with tape (3 measurements min)
- [ ] Compared outputs vs ground truth
- [ ] All 3 gates pass (area <5% error, ceiling <2cm, doors ±10cm)

