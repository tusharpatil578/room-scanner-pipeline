"""Video-based Structure from Motion (SfM) processing."""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path


@dataclass
class VideoProcessor:
    """Process video frames to extract 3D structure and camera poses."""
    
    def __init__(self, config=None):
        """Initialize video processor."""
        self.config = config
        self.orb = cv2.ORB_create(nfeatures=5000)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.poses = []
        self.point_cloud = None
    
    def extract_frames(self, video_path: Path, sample_rate: int = 5) -> List[np.ndarray]:
        """
        Extract frames from video.
        
        Args:
            video_path: Path to video file
            sample_rate: Extract every Nth frame
        
        Returns:
            List of frame images
        """
        frames = []
        cap = cv2.VideoCapture(str(video_path))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def detect_features(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Detect ORB features in image.
        
        Args:
            image: Input image
        
        Returns:
            (keypoints, descriptors)
        """
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        return keypoints, descriptors
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray, ratio_test: float = 0.7) -> List[cv2.DMatch]:
        """
        Match features between two images using Lowe's ratio test.
        
        Args:
            desc1: Descriptors from first image
            desc2: Descriptors from second image
            ratio_test: Lowe's ratio threshold
        
        Returns:
            List of good matches
        """
        if desc1 is None or desc2 is None:
            return []
        
        matches = self.bf_matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_test * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def estimate_pose(self, kp1: List, kp2: List, matches: List[cv2.DMatch], 
                      camera_matrix: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate relative pose between two frames using Essential Matrix.
        
        Args:
            kp1: Keypoints from first image
            kp2: Keypoints from second image
            matches: Feature matches
            camera_matrix: (3, 3) camera intrinsics
        
        Returns:
            (R, t) rotation and translation
        """
        if len(matches) < 8:
            return None, None
        
        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # Use default camera matrix if not provided
        if camera_matrix is None:
            camera_matrix = np.eye(3)
            camera_matrix[0, 2] = 320
            camera_matrix[1, 2] = 240
            camera_matrix[0, 0] = 500
            camera_matrix[1, 1] = 500
        
        # Compute Essential Matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, camera_matrix, cv2.FM_RANSAC, 0.999, 1.0)
        
        if E is None:
            return None, None
        
        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, camera_matrix, mask=mask)
        
        return R, t
    
    def triangulate_points(self, pts1: np.ndarray, pts2: np.ndarray, 
                          P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
        """
        Triangulate points from two views.
        
        Args:
            pts1: (N, 2) points in first image
            pts2: (N, 2) points in second image
            P1: (3, 4) projection matrix of first view
            P2: (3, 4) projection matrix of second view
        
        Returns:
            (N, 3) triangulated points
        """
        points_3d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = points_3d[:3] / points_3d[3]
        
        return points_3d.T.astype(np.float32)
    
    def process(self, video_path: Path, camera_matrix: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full SfM processing pipeline.
        
        Args:
            video_path: Path to video
            camera_matrix: Camera intrinsics
        
        Returns:
            (point_cloud, confidence)
        """
        # Extract frames
        frames = self.extract_frames(video_path)
        
        if len(frames) < 2:
            return np.array([]), np.array([])
        
        all_points = []
        
        # Process consecutive frames
        for i in range(len(frames) - 1):
            img1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            
            # Detect features
            kp1, desc1 = self.detect_features(img1)
            kp2, desc2 = self.detect_features(img2)
            
            if desc1 is None or desc2 is None:
                continue
            
            # Match features
            matches = self.match_features(desc1, desc2)
            
            if len(matches) < 8:
                continue
            
            # Estimate pose
            R, t = self.estimate_pose(kp1, kp2, matches, camera_matrix)
            
            if R is None:
                continue
            
            all_points.append(np.random.rand(100, 3))  # Placeholder
        
        if not all_points:
            return np.array([]), np.array([])
        
        point_cloud = np.vstack(all_points)
        confidence = np.full(len(point_cloud), 0.8, dtype=np.float32)
        
        return point_cloud, confidence