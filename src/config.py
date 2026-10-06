"""Configuration management for the room scanner pipeline."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SensorConfig:
    """LiDAR, Video, Photo specific parameters."""

    # LiDAR (Tier 3 - highest quality)
    lidar_depth_min: float = 0.1  # meters
    lidar_depth_max: float = 8.0  # meters
    lidar_confidence_threshold: float = 0.5
    lidar_outlier_std: float = 3.0  # std devs for outlier removal

    # Video (Tier 2 - medium quality)
    video_min_features: int = 100
    video_match_ratio: float = 0.7  # Lowe's ratio test
    video_keyframe_interval: int = 5  # frames
    video_scale_uncertainty: float = 0.1  # 10% scale ambiguity

    # Photos (Tier 1 - lowest quality but most accessible)
    photo_overlap_threshold: float = 0.3  # 30% overlap required
    photo_feature_count: int = 500
    photo_scale_hint_uncertainty: float = 0.2  # 20% uncertainty


@dataclass
class ProcessingConfig:
    """Processing parameters."""

    # Preprocessing
    depth_normalization_scale: float = 1000.0  # depth stored as uint16
    camera_intrinsic_uncertainty: float = 0.5  # pixels

    # Registration (ICP)
    icp_max_iterations: int = 50
    icp_distance_threshold: float = 0.05  # meters
    icp_fitness_threshold: float = 0.3

    # Drift correction
    loop_closure_distance_threshold: float = 0.5  # meters
    loop_closure_angle_threshold: float = 15.0  # degrees
    pose_graph_optimization_iterations: int = 20

    # Geometry extraction
    voxel_size: float = 0.01  # 1cm voxels for floor plan
    wall_thickness_tolerance: float = 0.05  # 5cm tolerance
    opening_min_width: float = 0.2  # 20cm minimum door/window
    floor_ceiling_margin: float = 0.05  # 5cm margin


@dataclass
class GateConfig:
    """Benchmark gate thresholds from assignment."""

    # Gate 1: Opening widths
    opening_width_tolerance: float = 0.02  # 2cm
    opening_detection_threshold: float = 0.85  # 85% of openings

    # Gate 2: Ceiling height
    ceiling_height_tolerance_per_room: float = 0.015  # 1.5cm
    ceiling_height_tolerance_repeated: float = 0.01  # 1cm across captures

    # Gate 3: Repeatability
    repeatability_tolerance_cm: float = 0.01  # 1cm
    repeatability_tolerance_percent: float = 0.005  # 0.5% per wall

    # Gate 4: Drift accountability (must be implemented, not just theoretical)
    drift_correction_required: bool = True

    # Gate 5: Photo-tier stitching
    photo_footprint_tolerance: float = 0.08  # ±8%
    photo_wall_length_tolerance: float = 0.08  # ±8% with calibrated intervals
    video_wall_length_tolerance: float = 0.03  # ±3%

    # Damage classification
    damage_class_colors: dict = None

    def __post_init__(self):
        if self.damage_class_colors is None:
            self.damage_class_colors = {
                "none": (0, 255, 0),  # green
                "cosmetic": (255, 165, 0),  # orange
                "structural": (255, 0, 0),  # red
                "concealed": (128, 0, 128),  # purple
            }


@dataclass
class OutputConfig:
    """Output format and schema parameters."""

    # JSON schema version
    schema_version: str = "1.0"

    # Output formats
    include_rendered_plan: bool = True
    include_point_cloud_ply: bool = True
    include_confidence_maps: bool = True

    # Rendering
    plan_dpi: int = 150
    plan_width_px: int = 1200
    point_cloud_sampling: int = 10000  # for visualization


@dataclass
class Config:
    """Master configuration."""

    # Paths
    data_dir: Path
    output_dir: Path
    raw_data_dir: Optional[Path] = None
    processed_data_dir: Optional[Path] = None
    ground_truth_dir: Optional[Path] = None

    # Sub-configs
    sensor: SensorConfig = None
    processing: ProcessingConfig = None
    gates: GateConfig = None
    output: OutputConfig = None

    # Random seed for reproducibility
    random_seed: int = 42

    # Device matrix (which tier runs on which hardware)
    device_matrix: dict = None

    def __post_init__(self):
        """Initialize sub-configs and directories."""
        if self.sensor is None:
            self.sensor = SensorConfig()
        if self.processing is None:
            self.processing = ProcessingConfig()
        if self.gates is None:
            self.gates = GateConfig()
        if self.output is None:
            self.output = OutputConfig()

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.processed_data_dir:
            self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        # Set raw/processed/ground truth paths
        if self.raw_data_dir is None:
            self.raw_data_dir = self.data_dir / "raw"
        if self.processed_data_dir is None:
            self.processed_data_dir = self.data_dir / "processed"
        if self.ground_truth_dir is None:
            self.ground_truth_dir = self.data_dir / "ground_truth"

        # Device matrix: which tier can run on which hardware
        if self.device_matrix is None:
            self.device_matrix = {
                "lidar": {
                    "hardware": "iPhone Pro (12+)",
                    "ios_version": "15+",
                    "accuracy": "±1-2cm",
                },
                "video": {
                    "hardware": "iPhone 15+",
                    "ios_version": "17+",
                    "accuracy": "±3-5cm",
                },
                "photo": {
                    "hardware": "Any iPhone 15+",
                    "ios_version": "17+",
                    "accuracy": "±8-10cm",
                },
            }

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        data_dir = Path(os.getenv("DATA_DIR", "./data"))
        output_dir = Path(os.getenv("OUTPUT_DIR", "./outputs"))
        random_seed = int(os.getenv("RANDOM_SEED", "42"))

        return cls(
            data_dir=data_dir,
            output_dir=output_dir,
            random_seed=random_seed,
        )

    def to_dict(self) -> dict:
        """Serialize config to dictionary."""
        return {
            "sensor": self.sensor.__dict__,
            "processing": self.processing.__dict__,
            "gates": self.gates.__dict__,
            "output": self.output.__dict__,
            "device_matrix": self.device_matrix,
            "random_seed": self.random_seed,
        }