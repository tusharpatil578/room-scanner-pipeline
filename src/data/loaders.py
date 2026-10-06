"""Data loading utilities for sensor data."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import imageio
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.constants import DataTier

logger = logging.getLogger(__name__)


class SensorData:
    """Container for a single sensor capture."""

    def __init__(
        self,
        tier: DataTier,
        capture_id: str,
        rgb_frames: Optional[np.ndarray] = None,
        depth_maps: Optional[np.ndarray] = None,
        confidence: Optional[np.ndarray] = None,
        camera_matrix: Optional[np.ndarray] = None,
        poses: Optional[np.ndarray] = None,  # (N, 4, 4) transformation matrices
        imu_data: Optional[pd.DataFrame] = None,
        odometry_data: Optional[pd.DataFrame] = None,
        timestamps: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None,
    ):
        """Initialize sensor data container.

        Args:
            tier: DataTier enum (PHOTO, VIDEO, LIDAR)
            capture_id: Unique identifier for this capture
            rgb_frames: (N, H, W, 3) RGB images
            depth_maps: (N, H, W) depth in meters
            confidence: (N, H, W) confidence scores [0, 1]
            camera_matrix: (3, 3) intrinsic matrix K
            poses: (N, 4, 4) SE(3) transformation matrices
            imu_data: DataFrame with imu measurements
            odometry_data: DataFrame with odometry/pose data
            timestamps: (N,) timestamps in seconds
            metadata: Additional metadata dict
        """
        self.tier = tier
        self.capture_id = capture_id
        self.rgb_frames = rgb_frames
        self.depth_maps = depth_maps
        self.confidence = confidence
        self.camera_matrix = camera_matrix
        self.poses = poses
        self.imu_data = imu_data
        self.odometry_data = odometry_data
        self.timestamps = timestamps
        self.metadata = metadata or {}

    def validate(self) -> bool:
        """Check data consistency."""
        if self.rgb_frames is not None and self.depth_maps is not None:
            assert len(self.rgb_frames) == len(self.depth_maps), "Frame count mismatch"

        if self.rgb_frames is not None and self.poses is not None:
            assert len(self.rgb_frames) == len(self.poses), "RGB/pose count mismatch"

        return True

    def __repr__(self) -> str:
        s = f"SensorData(tier={self.tier}, capture_id={self.capture_id}\n"
        if self.rgb_frames is not None:
            s += f"  RGB: {self.rgb_frames.shape}\n"
        if self.depth_maps is not None:
            s += f"  Depth: {self.depth_maps.shape}\n"
        if self.camera_matrix is not None:
            s += f"  Camera matrix: {self.camera_matrix.shape}\n"
        s += ")"
        return s


class DataLoader:
    """Loads sensor data from directory structure."""

    def __init__(self, data_dir: Path, tier: DataTier):
        """Initialize loader for a specific tier.

        Args:
            data_dir: Root directory containing sensor data
            tier: Which tier to load (PHOTO, VIDEO, or LIDAR)
        """
        self.data_dir = Path(data_dir)
        self.tier = tier

    def load(self, capture_id: str) -> SensorData:
        """Load all sensor data for a capture.

        Args:
            capture_id: Identifier for this capture

        Returns:
            SensorData object with all loaded data
        """
        if self.tier == DataTier.LIDAR:
            return self._load_lidar(capture_id)
        elif self.tier == DataTier.VIDEO:
            return self._load_video(capture_id)
        elif self.tier == DataTier.PHOTO:
            return self._load_photo(capture_id)
        else:
            raise ValueError(f"Unknown tier: {self.tier}")

    def _load_lidar(self, capture_id: str) -> SensorData:
        """Load LiDAR tier: depth + poses + intrinsics + RGB."""
        capture_dir = self.data_dir / capture_id
        assert capture_dir.exists(), f"Capture dir not found: {capture_dir}"

        logger.info(f"Loading LiDAR data from {capture_dir}")

        # Load depth maps (stored as npy or csv)
        depth_files = sorted(capture_dir.glob("depth/*"))
        if not depth_files:
            depth_files = sorted(capture_dir.glob("depth.npy")) or sorted(
                capture_dir.glob("depth.csv")
            )

        depth_maps = self._load_depth_sequence(depth_files)

        # Load confidence maps
        confidence = self._load_confidence(capture_dir / "confidence")

        # Load camera intrinsics
        camera_matrix = self._load_camera_matrix(capture_dir / "camera_matrix.xlsx")

        # Load odometry/poses
        odometry_data = self._load_dataframe(capture_dir / "odometry.xlsx")
        poses = self._poses_from_odometry(odometry_data, camera_matrix.shape)

        # Load RGB frames
        rgb_files = sorted(capture_dir.glob("rgb/*")) or sorted(capture_dir.glob("rgb.mp4"))
        rgb_frames = self._load_rgb_sequence(rgb_files)

        # Load IMU
        imu_data = self._load_dataframe(capture_dir / "imu.xlsx")

        data = SensorData(
            tier=DataTier.LIDAR,
            capture_id=capture_id,
            rgb_frames=rgb_frames,
            depth_maps=depth_maps,
            confidence=confidence,
            camera_matrix=camera_matrix,
            poses=poses,
            imu_data=imu_data,
            odometry_data=odometry_data,
            metadata={"source": str(capture_dir)},
        )

        data.validate()
        logger.info(f"Loaded LiDAR: {data}")
        return data

    def _load_video(self, capture_id: str) -> SensorData:
        """Load Video tier: handheld walkthrough MP4."""
        capture_dir = self.data_dir / capture_id
        assert capture_dir.exists(), f"Capture dir not found: {capture_dir}"

        logger.info(f"Loading Video data from {capture_dir}")

        # Find MP4 file
        video_files = sorted(capture_dir.glob("*.mp4"))
        if not video_files:
            video_files = sorted(capture_dir.glob("video/*/*.mp4"))

        assert video_files, f"No MP4 files found in {capture_dir}"

        # Extract frames from first MP4 (or specified one)
        video_path = video_files[0]
        rgb_frames = self._extract_frames_from_video(str(video_path), max_frames=500)

        # Load camera matrix if available (else default)
        camera_matrix = None
        if (capture_dir / "camera_matrix.xlsx").exists():
            camera_matrix = self._load_camera_matrix(capture_dir / "camera_matrix.xlsx")
        else:
            # Default iPhone camera matrix
            camera_matrix = self._default_camera_matrix(rgb_frames[0].shape)

        data = SensorData(
            tier=DataTier.VIDEO,
            capture_id=capture_id,
            rgb_frames=rgb_frames,
            camera_matrix=camera_matrix,
            metadata={"source": str(video_path), "frame_count": len(rgb_frames)},
        )

        data.validate()
        logger.info(f"Loaded Video: {data}")
        return data

    def _load_photo(self, capture_id: str) -> SensorData:
        """Load Photo tier: folder of images per room."""
        capture_dir = self.data_dir / capture_id
        assert capture_dir.exists(), f"Capture dir not found: {capture_dir}"

        logger.info(f"Loading Photo data from {capture_dir}")

        # Find all image files (per room or flat)
        image_files = sorted(capture_dir.glob("**/*.jpg")) + sorted(capture_dir.glob("**/*.png"))

        assert image_files, f"No images found in {capture_dir}"

        # Load images
        rgb_frames = []
        for img_file in tqdm(image_files, desc="Loading images"):
            img = cv2.imread(str(img_file))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb_frames.append(img_rgb)

        rgb_frames = np.array(rgb_frames)

        # Default camera matrix
        camera_matrix = self._default_camera_matrix(rgb_frames[0].shape)

        # Extract room assignments from folder structure if present
        room_assignments = self._extract_room_assignments(image_files)

        data = SensorData(
            tier=DataTier.PHOTO,
            capture_id=capture_id,
            rgb_frames=rgb_frames,
            camera_matrix=camera_matrix,
            metadata={
                "source": str(capture_dir),
                "image_count": len(rgb_frames),
                "room_assignments": room_assignments,
            },
        )

        data.validate()
        logger.info(f"Loaded Photo: {data}")
        return data

    # Helper methods

    @staticmethod
    def _load_depth_sequence(depth_files: List[Path]) -> np.ndarray:
        """Load sequence of depth maps."""
        depth_maps = []
        for depth_file in depth_files:
            if depth_file.suffix == ".npy":
                depth = np.load(depth_file)
            elif depth_file.suffix in [".csv", ".xlsx"]:
                depth = pd.read_csv(depth_file, header=None).values
            else:
                continue
            depth_maps.append(depth)

        return np.array(depth_maps) if depth_maps else None

    @staticmethod
    def _load_confidence(confidence_dir: Path) -> Optional[np.ndarray]:
        """Load confidence maps."""
        if not confidence_dir.exists():
            return None

        confidence_files = sorted(confidence_dir.glob("*"))
        confidence = []
        for conf_file in confidence_files:
            if conf_file.suffix == ".npy":
                conf = np.load(conf_file)
                confidence.append(conf)
        return np.array(confidence) if confidence else None

    @staticmethod
    def _load_camera_matrix(matrix_file: Path) -> np.ndarray:
        """Load 3x3 camera intrinsic matrix K."""
        if not matrix_file.exists():
            return None

        if matrix_file.suffix == ".xlsx":
            df = pd.read_excel(matrix_file, header=None)
            return df.values[:3, :3].astype(np.float32)
        elif matrix_file.suffix == ".csv":
            return pd.read_csv(matrix_file, header=None).values[:3, :3].astype(np.float32)

    @staticmethod
    def _load_dataframe(data_file: Path) -> Optional[pd.DataFrame]:
        """Load CSV or XLSX as DataFrame."""
        if not data_file.exists():
            return None

        if data_file.suffix == ".xlsx":
            return pd.read_excel(data_file)
        elif data_file.suffix == ".csv":
            return pd.read_csv(data_file)

    @staticmethod
    def _poses_from_odometry(odometry_df: Optional[pd.DataFrame], matrix_shape) -> Optional[
        np.ndarray
    ]:
        """Convert odometry DataFrame to (N, 4, 4) SE(3) matrices."""
        if odometry_df is None or len(odometry_df) == 0:
            return None

        # Assume columns: t, x, y, z, qx, qy, qz, qw (TUM format)
        # or similar pose representation
        poses = []
        for idx, row in odometry_df.iterrows():
            # Simplified: create identity matrix with translation
            pose = np.eye(4, dtype=np.float32)
            if "x" in row.index and "y" in row.index and "z" in row.index:
                pose[0, 3] = float(row["x"])
                pose[1, 3] = float(row["y"])
                pose[2, 3] = float(row["z"])
            poses.append(pose)

        return np.array(poses) if poses else None

    @staticmethod
    def _load_rgb_sequence(rgb_files: List[Path]) -> Optional[np.ndarray]:
        """Load sequence of RGB images."""
        if not rgb_files:
            return None

        rgb_frames = []
        for rgb_file in rgb_files:
            if rgb_file.suffix in [".jpg", ".png"]:
                img = cv2.imread(str(rgb_file))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                rgb_frames.append(img_rgb)

        return np.array(rgb_frames) if rgb_frames else None

    @staticmethod
    def _extract_frames_from_video(video_path: str, max_frames: int = 500) -> np.ndarray:
        """Extract evenly-spaced frames from MP4 video."""
        frames = []
        reader = imageio.get_reader(video_path)
        frame_count = len(reader)

        # Sample every N-th frame to stay under max_frames
        sample_rate = max(1, frame_count // max_frames)

        for i, frame in enumerate(reader):
            if i % sample_rate == 0:
                frames.append(frame[:, :, :3])  # RGB only

        return np.array(frames)

    @staticmethod
    def _default_camera_matrix(image_shape: Tuple[int, int, int]) -> np.ndarray:
        """Create default camera matrix for iPhone (assumed 75° FOV)."""
        h, w, _ = image_shape
        # Assume 75° horizontal FOV
        focal_length = w / (2 * np.tan(np.radians(75) / 2))
        cx, cy = w / 2, h / 2

        K = np.array(
            [
                [focal_length, 0, cx],
                [0, focal_length, cy],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        return K

    @staticmethod
    def _extract_room_assignments(image_files: List[Path]) -> Dict[int, str]:
        """Extract room IDs from folder structure."""
        assignments = {}
        for idx, img_file in enumerate(image_files):
            # Try to extract room from path: .../room_X/image.jpg
            try:
                parts = img_file.parent.name
                if parts.startswith("room_"):
                    room_id = int(parts.split("_")[1])
                    assignments[idx] = f"room_{room_id}"
            except (ValueError, IndexError):
                assignments[idx] = "unknown"

        return assignments


def load_benchmark_dataset(
    data_dir: Path, dataset_name: str = "single_scan_with_ceiling"
) -> Dict[str, SensorData]:
    """Load one of the pre-provided benchmark datasets.

    Args:
        data_dir: Root data directory
        dataset_name: Name of dataset (single_room, single_scan_floor_only, single_scan_with_ceiling)

    Returns:
        Dict mapping tier name to SensorData
    """
    benchmark_dir = data_dir / "raw" / "benchmarks" / dataset_name

    data = {}
    for tier in [DataTier.LIDAR, DataTier.VIDEO, DataTier.PHOTO]:
        loader = DataLoader(benchmark_dir, tier)
        try:
            data[tier.value] = loader.load(dataset_name)
        except Exception as e:
            logger.warning(f"Failed to load {tier.value}: {e}")

    return data