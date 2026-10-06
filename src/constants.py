"""Constants for room scanner pipeline."""

from enum import Enum
from typing import Dict

# ============================================================================
# Data Tier Definitions
# ============================================================================


class DataTier(str, Enum):
    """Input data tier enumeration."""

    PHOTO = "photo"  # Tier 1: Photos only
    VIDEO = "video"  # Tier 2: Handheld video walkthrough
    LIDAR = "lidar"  # Tier 3: LiDAR depth + poses + intrinsics


# ============================================================================
# Benchmark Gates (from assignment spec)
# ============================================================================


class BenchmarkGate(str, Enum):
    """Benchmark evaluation gates."""

    OPENING_WIDTHS = "opening_widths"
    CEILING_HEIGHT = "ceiling_height"
    REPEATABILITY = "repeatability"
    DRIFT_ACCOUNTABILITY = "drift_accountability"
    PHOTO_TIER_STITCHING = "photo_tier_stitching"


# ============================================================================
# Damage Classification
# ============================================================================


class DamageClass(str, Enum):
    """Damage classification levels."""

    NONE = "none"  # No damage
    COSMETIC = "cosmetic"  # Paint, aesthetic issues
    STRUCTURAL = "structural"  # Cracks, holes, material failure
    CONCEALED = "concealed"  # Inferred from anomalies


# ============================================================================
# Unit System
# ============================================================================

UNITS = {
    "length": "meters",
    "area": "square_meters",
    "volume": "cubic_meters",
    "angle": "radians",
}

# Common conversions
CM_TO_M = 0.01
MM_TO_M = 0.001

# ============================================================================
# Sensor Data File Names (from uploaded images)
# ============================================================================

LIDAR_SENSOR_FILES = {
    "depth": "depth",  # Depth maps (npy or csv)
    "confidence": "confidence",  # Confidence maps
    "camera_matrix": "camera_matrix.xlsx",  # Intrinsic parameters K
    "imu": "imu.xlsx",  # IMU data
    "odometry": "odometry.xlsx",  # Pose data
    "rgb": "rgb",  # RGB images (MP4 or frames)
}

# ============================================================================
# Output JSON Schema Fields
# ============================================================================

OUTPUT_JSON_SCHEMA = {
    "schema_version": "1.0",
    "capture_id": "string",
    "tier": "enum[photo, video, lidar]",
    "timestamp": "ISO8601",
    "per_room_plans": [
        {
            "room_id": "int",
            "room_name": "string",
            "floor_area_m2": "float",
            "ceiling_height_m": "float",
            "wall_lengths": [
                {
                    "wall_id": "int",
                    "length_m": "float",
                    "confidence_interval": {"low": "float", "high": "float"},
                }
            ],
            "openings": [
                {
                    "opening_id": "int",
                    "type": "enum[door, window]",
                    "width_m": "float",
                    "height_m": "float",
                }
            ],
        }
    ],
    "multi_room_plan": {
        "total_area_m2": "float",
        "room_adjacencies": [
            {"room_a": "int", "room_b": "int", "wall_length_m": "float"}
        ],
        "coordinates": "array[N, 2] in meters",
    },
    "damage_regions": [
        {
            "damage_id": "int",
            "class": "enum[none, cosmetic, structural, concealed]",
            "surface": "string",
            "area_m2": "float",
            "bounding_box": {"x": "float", "y": "float", "width": "float", "height": "float"},
        }
    ],
    "scope_items": [
        {"item_id": "int", "description": "string", "area_m2": "float", "surface": "string"}
    ],
    "measurements_confidence": [
        {"measurement": "string", "confidence_percent": "float"}
    ],
    "processing_time_seconds": "float",
    "device_info": {
        "hardware": "string",
        "ios_version": "string",
        "sensor_accuracy": "string",
    },
}

# ============================================================================
# Compliance Deliverables
# ============================================================================

DELIVERABLES = [
    {
        "id": 1,
        "name": "Compliance matrix",
        "description": "requirement → file path → artifact → status",
    },
    {
        "id": 2,
        "name": "Capture route",
        "description": "TestFlight/dev build or 1-page stock-capture protocol + device matrix",
    },
    {
        "id": 3,
        "name": "Repo",
        "description": "README to running on fresh capture in <15 minutes, one command per capture",
    },
    {
        "id": 4,
        "name": "Reproduction bundle",
        "description": "Everything to regenerate every reported number from raw inputs",
    },
    {
        "id": 5,
        "name": "Benchmark report",
        "description": "Gates at all three tiers, repeatability table, head-to-head table, timing",
    },
    {
        "id": 6,
        "name": "Fix loop bundle",
        "description": "Before run, after run, diff, root cause analysis",
    },
    {
        "id": 7,
        "name": "Technical report",
        "description": "Max 6 pages: architecture, tier design, device matrix, drift handling, calibration, fix loop, known failure modes",
    },
    {
        "id": 8,
        "name": "Raw benchmark data",
        "description": "Sensor logs, ground truth, app exports",
    },
]

# ============================================================================
# Scoring Weights (from assignment)
# ============================================================================

SCORING_WEIGHTS = {
    "walk_in_test": 0.30,  # 30%
    "fix_loop_delta": 0.25,  # 25%
    "benchmark_accuracy": 0.15,  # 15%
    "compliance_matrix": 0.10,  # 10%
    "head_to_head": 0.10,  # 10%
    "capture_route": 0.05,  # 5%
    "process_evidence": 0.05,  # 5%
}

# ============================================================================
# Typical Room Dimensions (for sanity checks)
# ============================================================================

TYPICAL_DIMENSIONS = {
    "residential_ceiling_height_m": 2.7,  # 2.7 ± 0.3m
    "commercial_ceiling_height_m": 3.0,  # 3.0 ± 0.5m
    "min_room_area_m2": 5.0,  # 5 m²
    "max_room_area_m2": 100.0,  # 100 m²
    "max_wall_length_m": 20.0,  # 20m
    "typical_door_width_m": 0.9,  # 0.9m
    "typical_window_width_m": 1.2,  # 1.2m
}

# ============================================================================
# Incumbent Apps (for head-to-head comparison)
# ============================================================================

INCUMBENT_APPS = {
    "magicplan": {
        "url": "https://www.magicplan.app/",
        "free_tier_available": True,
    },
    "poly_cam": {
        "url": "https://www.polycam.com/",
        "free_tier_available": True,
    },
}