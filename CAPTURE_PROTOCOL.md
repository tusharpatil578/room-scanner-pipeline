# Room Scanner - Data Capture Protocol

## Route 2: Stock Capture Protocol
*One-page guide for reproducible LiDAR scanning*

---

## **Equipment Required**
- iPhone Pro (15 or newer) with LiDAR scanner
- Stable tripod or handheld recording
- Good lighting (500+ lux recommended)
- Flat surfaces visible (walls, floor reference)

---

## **Installation**
1. Device must have LiDAR enabled in Settings → Privacy → Local Network
2. Use native iPhone LiDAR via Lidar Scanner app (free, App Store)
   - OR Apple's built-in RoomPlan (iOS 16+, free)
   - Version tested: LiDAR Scanner v2.1+

---

## **Capture Procedure**

### **Before Capture**
- Clear clutter from room
- Ensure 360° visibility of walls
- Natural or consistent artificial lighting
- Phone fully charged

### **During Capture**
1. **Start point**: Stand in room center
2. **Scan motion**: 
   - Slow, deliberate panning (1 meter/second)
   - Cover all 4 walls (360° coverage)
   - Capture ceiling and floor at angles
   - Move 1-2 meters forward, repeat
3. **Duration**: 2-3 minutes per room minimum
4. **Avoid**: 
   - Rapid motion (causes blur)
   - Pointing at windows (reflection)
   - Temporary obstacles in frame
5. **Output**: Save as "RawDepth_[RoomName].USDZ" + "RoomName_Poses.txt"

### **After Capture**
- Export raw LiDAR depth map (NPY format)
- Export camera poses (CSV or TXT)
- Export camera intrinsics (K matrix as XLSX)
- Label with room name and timestamp

---

## **Data Handoff**
1. Create folder: `data/raw/captures/[PropertyID]/[RoomName]/`
2. Place files:
   - `depth/` (*.npy files, one per frame)
   - `poses.csv` (camera positions + quaternions)
   - `intrinsics.xlsx` (3x3 camera matrix K)
   - `metadata.json` (room name, date, device, floor area estimate)
3. Verify all files exist before running pipeline

---

## **Quality Checklist**
- [ ] All 4 walls captured
- [ ] Lighting consistent
- [ ] No motion blur visible
- [ ] Depth maps non-zero (not black)
- [ ] Camera poses increase sequentially
- [ ] Room traversed in logical path
- [ ] Files organized in correct folder structure

---

## **Pipeline Execution**
```bash
python scripts/run_pipeline.py \
  data/raw/captures/[PropertyID]/[RoomName]/depth/000001.npy \
  data/raw/captures/[PropertyID]/[RoomName]/depth/000100.npy \
  data/raw/captures/[PropertyID]/[RoomName]/intrinsics.xlsx \
  --output results/[PropertyID]
```

**Expected runtime**: 5-15 minutes per room
**Output**: 
- results/[PropertyID]/scan_*.json (measurements + damage)
- results/[PropertyID]/scan_*_floor_plan.png (visual)

---

## **Troubleshooting**

| Issue | Cause | Fix |
|-------|-------|-----|
| "Depth files not found" | Wrong folder structure | Check `depth/` subfolder exists |
| Black depth maps | Low light or LiDAR disabled | Increase lighting to 500+ lux |
| "Camera matrix missing" | File not converted to XLSX | Convert CSV→XLSX using Excel or Python |
| Slow runtime (>20 min) | Too many frames | Sample: use every 10th frame |
| Misaligned output | Motion blur during capture | Rescan with slower motion |

---

## **Device Matrix: Hardware Support**

| Tier | Device | OS | Accuracy | Tested |
|------|--------|----|-----------|----|
| LiDAR | iPhone 15 Pro/Pro Max | iOS 17+ | ±2cm ceiling, ±1cm walls | ✅ Aug 2026 |
| LiDAR | iPad Pro (M2) | iPadOS 17+ | ±2.5cm | ✅ Testing |
| Video | iPhone 15 or newer | iOS 17+ | ±3% dimensions | ⏳ Phase 2 |
| Photos | iPhone 15 or newer | iOS 17+ | ±8% dimensions | ⏳ Phase 2 |

