"""Data validation utilities."""

import logging
from typing import Dict, List, Tuple

import numpy as np

from src.data.loaders import SensorData

logger = logging.getLogger(__name__)


class ValidationResult:
    """Container for validation results."""

    def __init__(self):
        self.checks: Dict[str, bool] = {}
        self.warnings: List[str] = []
        self.errors: List[str] = []

    def add_check(self, name: str, passed: bool, message: str = ""):
        """Record a validation check."""
        self.checks[name] = passed
        if not passed:
            self.errors.append(f"{name}: {message}")

    def add_warning(self, message: str):
        """Add a warning."""
        self.warnings.append(message)

    def is_valid(self) -> bool:
        """Check if all validations passed."""
        return all(self.checks.values()) and len(self.errors) == 0

    def __repr__(self) -> str:
        s = "Validation Results:\n"
        s += f"  Passed: {sum(self.checks.values())}/{len(self.checks)}\n"
        if self.warnings:
            s += f"  Warnings: {len(self.warnings)}\n"
        if self.errors:
            s += f"  Errors: {len(self.errors)}\n"
        return s


class DataValidator:
    """Validates sensor data quality and consistency."""

    def __init__(self, config=None):
        """Initialize validator.

        Args:
            config: Config object with validation thresholds
        """
        self.config = config

    def validate(self, data: SensorData) -> ValidationResult:
        """Run comprehensive validation on sensor data.

        Args:
            data: SensorData object to validate

        Returns:
            ValidationResult object with all checks
        """
        result = ValidationResult()

        logger.info(f"Validating {data.tier} data...")

        # Basic presence checks
        result.add_check(
            "rgb_frames_present", data.rgb_frames is not None, "RGB frames missing"
        )
        if data.rgb_frames is not None:
            result.add_check(
                "rgb_frames_shape_valid",
                len(data.rgb_frames.shape) == 4,
                f"Expected (N,H,W,C), got {data.rgb_frames.shape}",
            )

        # Tier-specific checks
        if data.tier.value == "lidar":
            self._validate_lidar(data, result)
        elif data.tier.value == "video":
            self._validate_video(data, result)
        elif data.tier.value == "photo":
            self._validate_photo(data, result)

        # Camera matrix checks
        if data.camera_matrix is not None:
            self._validate_camera_matrix(data.camera_matrix, result)

        logger.info(result)
        return result

    def _validate_lidar(self, data: SensorData, result: ValidationResult):
        """LiDAR-specific validation."""
        # Depth presence and shape
        result.add_check(
            "depth_present", data.depth_maps is not None, "Depth maps missing"
        )
        if data.depth_maps is not None:
            result.add_check(
                "depth_shape_valid",
                len(data.depth_maps.shape) == 3,
                f"Expected (N,H,W), got {data.depth_maps.shape}",
            )

            # Depth value ranges
            depth_min, depth_max = np.nanmin(data.depth_maps), np.nanmax(data.depth_maps)
            result.add_check(
                "depth_min_positive",
                depth_min >= 0.01,
                f"Min depth {depth_min} unrealistic",
            )
            result.add_check(
                "depth_max_reasonable",
                depth_max < 50,
                f"Max depth {depth_max} too large",
            )

        # Confidence presence
        result.add_check(
            "confidence_present", data.confidence is not None, "Confidence maps missing"
        )
        if data.confidence is not None:
            conf_min, conf_max = np.nanmin(data.confidence), np.nanmax(data.confidence)
            result.add_check(
                "confidence_range",
                0 <= conf_min and conf_max <= 1,
                f"Confidence out of [0, 1]: [{conf_min}, {conf_max}]",
            )

        # Poses presence
        result.add_check("poses_present", data.poses is not None, "Poses missing")
        if data.poses is not None:
            result.add_check(
                "poses_shape_valid",
                data.poses.shape[1:] == (4, 4),
                f"Expected (N,4,4), got {data.poses.shape}",
            )

        # Odometry data
        result.add_check(
            "odometry_present", data.odometry_data is not None, "Odometry missing"
        )

        # IMU data
        if data.imu_data is not None:
            self._validate_imu(data.imu_data, result)

    def _validate_video(self, data: SensorData, result: ValidationResult):
        """Video-specific validation."""
        # Must have reasonable frame count
        if data.rgb_frames is not None:
            frame_count = len(data.rgb_frames)
            result.add_check(
                "video_frame_count",
                frame_count >= 10,
                f"Too few frames: {frame_count}",
            )

            if frame_count > 500:
                result.add_warning(f"Large frame count {frame_count}: may need downsampling")

    def _validate_photo(self, data: SensorData, result: ValidationResult):
        """Photo-specific validation."""
        if data.rgb_frames is not None:
            frame_count = len(data.rgb_frames)
            result.add_check(
                "photo_count", frame_count >= 2, f"Too few photos: {frame_count}"
            )

            if "room_assignments" in data.metadata:
                rooms = set(data.metadata["room_assignments"].values())
                if len(rooms) > 1:
                    logger.info(f"Multi-room photo set: {rooms}")

    def _validate_camera_matrix(self, K: np.ndarray, result: ValidationResult):
        """Validate camera intrinsic matrix."""
        result.add_check(
            "camera_matrix_shape",
            K.shape == (3, 3),
            f"Expected (3,3), got {K.shape}",
        )

        if K.shape == (3, 3):
            # Check K[2,2] == 1
            result.add_check(
                "camera_matrix_k33",
                np.isclose(K[2, 2], 1.0),
                f"K[2,2]={K[2,2]} should be 1.0",
            )

            # Check focal length reasonable (pixel units)
            fx, fy = K[0, 0], K[1, 1]
            result.add_check(
                "focal_length_reasonable",
                100 < fx < 10000 and 100 < fy < 10000,
                f"Focal lengths unrealistic: fx={fx}, fy={fy}",
            )

    def _validate_imu(self, imu_df, result: ValidationResult):
        """Validate IMU data."""
        required_cols = ["ax", "ay", "az", "gx", "gy", "gz"]
        missing_cols = [col for col in required_cols if col not in imu_df.columns]

        if missing_cols:
            result.add_warning(f"Missing IMU columns: {missing_cols}")
        else:
            # Check acceleration magnitude near 9.81 m/s²
            accel = np.sqrt(
                imu_df["ax"] ** 2 + imu_df["ay"] ** 2 + imu_df["az"] ** 2
            )
            accel_mean = accel.mean()
            result.add_check(
                "imu_gravity",
                8 < accel_mean < 12,
                f"Mean acceleration {accel_mean} should be ~9.81 m/s²",
            )

    def compute_point_density(self, point_cloud: np.ndarray) -> float:
        """Compute point density for confidence metric.

        Args:
            point_cloud: (N, 3) point array in meters

        Returns:
            Points per cubic meter
        """
        if len(point_cloud) == 0:
            return 0

        # Compute bounding box
        min_pt = point_cloud.min(axis=0)
        max_pt = point_cloud.max(axis=0)
        volume = np.prod(max_pt - min_pt)

        if volume < 1e-6:
            return 0

        return len(point_cloud) / volume

    def compute_measurement_confidence(
        self, point_cloud: np.ndarray, measurement_type: str
    ) -> float:
        """Estimate confidence interval for measurements.

        Args:
            point_cloud: (N, 3) point array
            measurement_type: 'length', 'area', 'volume'

        Returns:
            Confidence score 0-1
        """
        density = self.compute_point_density(point_cloud)

        # Empirical mapping from point density to confidence
        if measurement_type == "length":
            return min(1.0, density / 100)
        elif measurement_type == "area":
            return min(1.0, density / 50)
        elif measurement_type == "volume":
            return min(1.0, density / 25)
        else:
            return 0.5