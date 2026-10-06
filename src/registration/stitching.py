"""Multi-room global alignment and stitching."""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class MultiRoomStitcher:
    """Stitch multiple rooms into global floor plan."""
    
    def __init__(self):
        """Initialize stitcher."""
        self.room_point_clouds = {}  # room_id -> point_cloud
        self.room_poses = {}  # room_id -> (position, R)
        self.adjacencies = {}  # room_id -> [adjacent_room_ids]
    
    def register_room(self, room_id: str, point_cloud: np.ndarray, 
                     pose: Tuple[np.ndarray, np.ndarray]):
        """
        Register a room's point cloud.
        
        Args:
            room_id: Unique room identifier
            point_cloud: (N, 3) room point cloud
            pose: (position, R) room pose
        """
        self.room_point_clouds[room_id] = point_cloud
        self.room_poses[room_id] = pose
    
    def detect_adjacency(self, room_a_id: str, room_b_id: str, 
                        threshold: float = 0.5) -> bool:
        """
        Detect if two rooms are adjacent.
        
        Args:
            room_a_id: First room ID
            room_b_id: Second room ID
            threshold: Distance threshold for adjacency (meters)
        
        Returns:
            True if adjacent
        """
        pc_a = self.room_point_clouds.get(room_a_id)
        pc_b = self.room_point_clouds.get(room_b_id)
        
        if pc_a is None or pc_b is None:
            return False
        
        # Check if rooms are close
        min_distance = np.min(np.linalg.norm(pc_a[:, np.newaxis, :] - pc_b[np.newaxis, :, :], axis=2))
        
        return min_distance < threshold
    
    def compute_global_poses(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute global poses for all rooms using pose graph.
        
        Returns:
            room_id -> global_pose
        """
        global_poses = {}
        reference_room = list(self.room_point_clouds.keys())[0]
        
        # Place first room at origin
        global_poses[reference_room] = self.room_poses[reference_room]
        
        visited = {reference_room}
        queue = [reference_room]
        
        # BFS to compute all global poses
        while queue:
            current_room = queue.pop(0)
            
            for adjacent_room in self.adjacencies.get(current_room, []):
                if adjacent_room in visited:
                    continue
                
                visited.add(adjacent_room)
                queue.append(adjacent_room)
                
                # Compute relative pose
                local_pose = self.room_poses[adjacent_room]
                global_poses[adjacent_room] = local_pose
        
        return global_poses
    
    def stitch_global(self) -> np.ndarray:
        """
        Stitch all rooms into global point cloud.
        
        Returns:
            (N, 3) global point cloud
        """
        global_poses = self.compute_global_poses()
        
        global_points = []
        
        for room_id, point_cloud in self.room_point_clouds.items():
            if room_id not in global_poses:
                continue
            
            pos, R = global_poses[room_id]
            
            # Transform to global coordinates
            transformed = np.dot(point_cloud, R.T) + pos
            global_points.append(transformed)
        
        if not global_points:
            return np.array([]).reshape(0, 3)
        
        return np.vstack(global_points).astype(np.float32)