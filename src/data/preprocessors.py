"""Data preprocessing and normalization."""

import logging
from typing import Optional

import cv2
import numpy as np

from src.config import Config
from src.data.loaders import SensorData

logger = logging.getLogger(__name__)


class PreprocessorPipeline:
    """Normalize and preprocess sensor data across tiers."""

    def __init__(self, config: Config):
        """Initialize preprocessor.

        Args:
            config: Configuration object
        """
        self.config = config

    def process(self, data: SensorData) -> SensorData:
        """Apply full preprocessing pipeline.

        Args:
            data: Raw SensorData

        Returns:
            Preprocessed SensorData
        """
        logger.info(f"Preprocessing {data.tier} tier data...")

        # Step 1: Normalize RGB frames
        if data.rgb_frames is not None:
            data.rgb_frames = self._normalize_rgb(data.rgb_frames)

        # Step 2: Normalize depth (if present)
        if data.depth_maps is not None:
            data.depth_maps = self._normalize_depth(data.depth_maps)

        # Step 3: Undistort camera frames (if calibration available)
        if data.rgb_frames is not None and data.camera_matrix is not None:
            data = self._undistort_frames(data)

        # Step 4: Remove NaN/inf values
        if data.rgb_frames is not None:
            data.rgb_frames = np.nan_to_num(data.rgb_frames, nan=0)

        if data.depth_maps is not None:
            data.depth_maps = np.nan_to_num(data.depth_maps, nan=0)

        # Step 5: Temporal alignment (if timestamps available)
        if data.timestamps is not None:
            data = self._align_timestamps(data)

        logger.info(f"Preprocessing complete for {data.tier}")
        return data

    def _normalize_rgb(self, rgb_frames: np.ndarray) -> np.ndarray:
        """Normalize RGB to uint8 in [0, 255].

        Args:
            rgb_frames: (N, H, W, 3) array

        Returns:
            Normalized RGB frames
        """
        # Check if already uint8
        if rgb_frames.dtype == np.uint8:
            return rgb_frames

        # Convert from float [0, 1] or [0, 255] to uint8
        if rgb_frames.max() <= 1.0:
            rgb_frames = (rgb_frames * 255).astype(np.uint8)
        else:
            rgb_frames = np.clip(rgb_frames, 0, 255).astype(np.uint8)

        return rgb_frames

    def _normalize_depth(self, depth_maps: np.ndarray) -> np.ndarray:
        """Normalize depth to meters.

        Args:
            depth_maps: (N, H, W) depth array

        Returns:
            Depth in meters
        """
        # If uint16, likely stored in mm (common for depth cameras)
        if depth_maps.dtype == np.uint16:
            depth_maps = depth_maps.astype(np.float32) / 1000.0

        elif depth_maps.max() > 100:
            # If max > 100, assume mm, convert to meters
            depth_maps = depth_maps / 1000.0

        # Clip to reasonable range [0.01m, 50m]
        depth_maps = np.clip(depth_maps, 0.01, 50.0)

        return depth_maps.astype(np.float32)

    def _undistort_frames(self, data: SensorData) -> SensorData:
        """Undistort RGB and depth frames using camera matrix.

        Args:
            data: SensorData with camera_matrix

        Returns:
            SensorData with undistorted frames
        """
        if data.camera_matrix is None or data.rgb_frames is None:
            return data

        logger.info("Undistorting frames...")

        K = data.camera_matrix
        h, w = data.rgb_frames[0].shape[:2]

        # Assume no distortion coefficients (common for iPhone with computational photography)
        D = np.zeros(5, dtype=np.float32)

        undistorted_rgb = []
        for frame in data.rgb_frames:
            # Undistort
            frame_undist = cv2.undistort(frame, K, D)
            undistorted_rgb.append(frame_undist)

        data.rgb_frames = np.array(undistorted_rgb)

        # Same for depth if present
        if data.depth_maps is not None:
            undistorted_depth = []
            for depth in data.depth_maps:
                depth_undist = cv2.undistort(depth, K, D)
                undistorted_depth.append(depth_undist)
            data.depth_maps = np.array(undistorted_depth)

        return data

    def _align_timestamps(self, data: SensorData) -> SensorData:
        """Align temporal data to common timeline.

        Args:
            data: SensorData with timestamps

        Returns:
            SensorData with aligned timing
        """
        if data.timestamps is None or len(data.timestamps) == 0:
            return data

        # Ensure timestamps are monotonically increasing
        if np.any(np.diff(data.timestamps) < 0):
            logger.warning("Timestamps not monotonic, sorting...")
            sort_idx = np.argsort(data.timestamps)
            data.timestamps = data.timestamps[sort_idx]

            if data.rgb_frames is not None:
                data.rgb_frames = data.rgb_frames[sort_idx]
            if data.depth_maps is not None:
                data.depth_maps = data.depth_maps[sort_idx]
            if data.poses is not None:
                data.poses = data.poses[sort_idx]

        return data

    def resize_frames(
        self, rgb_frames: np.ndarray, target_size: Optional[tuple] = None
    ) -> np.ndarray:
        """Resize RGB frames for consistency.

        Args:
            rgb_frames: (N, H, W, C) array
            target_size: (height, width) or None to keep as-is

        Returns:
            Resized frames
        """
        if target_size is None:
            return rgb_frames

        resized = []
        for frame in rgb_frames:
            frame_resized = cv2.resize(frame, (target_size[1], target_size[0]))
            resized.append(frame_resized)

        return np.array(resized)

    def compute_optical_flow(
        self, rgb_frames: np.ndarray
    ) -> np.ndarray:
        """Compute optical flow between consecutive frames.

        Args:
            rgb_frames: (N, H, W, 3) array

        Returns:
            (N-1, H, W, 2) optical flow array
        """
        logger.info("Computing optical flow...")

        flows = []
        gray_prev = cv2.cvtColor(rgb_frames[0], cv2.COLOR_RGB2GRAY)

        for i in range(1, len(rgb_frames)):
            gray_curr = cv2.cvtColor(rgb_frames[i], cv2.COLOR_RGB2GRAY)

            # Lucas-Kanade optical flow
            flow = cv2.calcOpticalFlowFarneback(
                gray_prev, gray_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            flows.append(flow)

            gray_prev = gray_curr

        return np.array(flows)

    def estimate_camera_motion(
        self, rgb_frames: np.ndarray, camera_matrix: np.ndarray
    ) -> np.ndarray:
        """Estimate camera motion from consecutive frames.

        Args:
            rgb_frames: (N, H, W, 3) frames
            camera_matrix: (3, 3) intrinsic matrix

        Returns:
            (N-1, 4, 4) relative pose transformations
        """
        logger.info("Estimating camera motion...")

        poses = []
        gray_prev = cv2.cvtColor(rgb_frames[0], cv2.COLOR_RGB2GRAY)

        for i in range(1, len(rgb_frames)):
            gray_curr = cv2.cvtColor(rgb_frames[i], cv2.COLOR_RGB2GRAY)

            # Feature detection and matching
            orb = cv2.ORB_create(nfeatures=500)
            kp1, des1 = orb.detectAndCompute(gray_prev, None)
            kp2, des2 = orb.detectAndCompute(gray_curr, None)

            if des1 is None or des2 is None:
                # No features found, use identity
                poses.append(np.eye(4, dtype=np.float32))
                gray_prev = gray_curr
                continue

            # Brute force matching
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) < 4:
                poses.append(np.eye(4, dtype=np.float32))
                gray_prev = gray_curr
                continue

            # Extract matched points
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

            # Compute fundamental matrix
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)

            if F is None:
                poses.append(np.eye(4, dtype=np.float32))
                gray_prev = gray_curr
                continue

            # Recover pose (simplified: assume unit translation)
            E = camera_matrix.T @ F @ camera_matrix
            _, R, t, _ = cv2.recoverPose(E, pts1, pts2, camera_matrix)

            # Build SE(3) matrix
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = R
            pose[:3, 3] = t.flatten()

            poses.append(pose)
            gray_prev = gray_curr

        return np.array(poses)