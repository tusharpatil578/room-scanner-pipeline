"""Photo-based room stitching using homography."""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path


@dataclass
class PhotoProcessor:
    """Stitch per-room photos to create floor plan."""
    
    def __init__(self, config=None):
        """Initialize photo processor."""
        self.config = config
        self.sift = cv2.SIFT_create()
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    def load_photos(self, photo_dir: Path) -> List[np.ndarray]:
        """
        Load all photos from directory.
        
        Args:
            photo_dir: Directory containing photos
        
        Returns:
            List of images
        """
        photos = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for file_path in sorted(photo_dir.iterdir()):
            if file_path.suffix.lower() in image_extensions:
                img = cv2.imread(str(file_path))
                if img is not None:
                    photos.append(img)
        
        return photos
    
    def detect_features(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Detect SIFT features.
        
        Args:
            image: Input image
        
        Returns:
            (keypoints, descriptors)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List[cv2.DMatch]:
        """
        Match SIFT descriptors between two images.
        
        Args:
            desc1: Descriptors from first image
            desc2: Descriptors from second image
        
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
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def compute_homography(self, kp1: List, kp2: List, matches: List[cv2.DMatch]) -> Optional[np.ndarray]:
        """
        Compute homography matrix from matches.
        
        Args:
            kp1: Keypoints from first image
            kp2: Keypoints from second image
            matches: Feature matches
        
        Returns:
            (3, 3) homography matrix
        """
        if len(matches) < 4:
            return None
        
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        
        return H
    
    def warp_and_blend(self, img1: np.ndarray, img2: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        Warp and blend two images using homography.
        
        Args:
            img1: Reference image
            img2: Image to warp
            H: Homography matrix
        
        Returns:
            Blended image
        """
        h, w = img1.shape[:2]
        
        # Warp img2 to align with img1
        warped = cv2.warpPerspective(img2, H, (w, h))
        
        # Simple blending (average)
        blended = cv2.addWeighted(img1, 0.5, warped, 0.5, 0)
        
        return blended
    
    def stitch_photos(self, photos: List[np.ndarray]) -> np.ndarray:
        """
        Stitch multiple photos into a single image.
        
        Args:
            photos: List of images to stitch
        
        Returns:
            Stitched image
        """
        if len(photos) == 0:
            return np.array([])
        
        if len(photos) == 1:
            return photos[0]
        
        # Start with first image
        result = photos[0].copy()
        
        # Stitch subsequent images
        for i in range(1, len(photos)):
            kp1, desc1 = self.detect_features(result)
            kp2, desc2 = self.detect_features(photos[i])
            
            if desc1 is None or desc2 is None:
                continue
            
            matches = self.match_features(desc1, desc2)
            
            if len(matches) < 4:
                continue
            
            H = self.compute_homography(kp1, kp2, matches)
            
            if H is None:
                continue
            
            result = self.warp_and_blend(result, photos[i], H)
        
        return result
    
    def process(self, photo_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full photo processing pipeline.
        
        Args:
            photo_dir: Directory containing photos
        
        Returns:
            (point_cloud, confidence)
        """
        # Load photos
        photos = self.load_photos(photo_dir)
        
        if len(photos) == 0:
            return np.array([]), np.array([])
        
        # Stitch photos
        stitched = self.stitch_photos(photos)
        
        # Generate synthetic point cloud from stitched image
        # (In production, use more sophisticated methods)
        h, w = stitched.shape[:2]
        grid_x, grid_y = np.meshgrid(np.linspace(0, 10, w), np.linspace(0, 10, h))
        points_3d = np.dstack([grid_x, grid_y, np.ones_like(grid_x)])
        point_cloud = points_3d.reshape(-1, 3).astype(np.float32)
        
        confidence = np.full(len(point_cloud), 0.7, dtype=np.float32)
        
        return point_cloud, confidence