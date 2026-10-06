"""Visualization of results."""
import numpy as np
from PIL import Image, ImageDraw


class FloorPlanVisualizer:
    """Visualize floor plans."""
    
    @staticmethod
    def visualize_floor_plan(floor_plan, measurements, output_size=800):
        """Create floor plan PNG visualization.
        
        Args:
            floor_plan: Floor plan dict
            measurements: Measurements dict
            output_size: Output image size in pixels
            
        Returns:
            PIL Image
        """
        # Create blank image
        img = Image.new('RGB', (output_size, output_size), color='white')
        draw = ImageDraw.Draw(img)
        
        # Get floor points
        floor_points = floor_plan.get('floor_points', [])
        
        if len(floor_points) == 0:
            return img
        
        # Normalize to image size
        floor_points_2d = floor_points[:, :2]
        min_pt = np.min(floor_points_2d, axis=0)
        max_pt = np.max(floor_points_2d, axis=0)
        range_pt = max_pt - min_pt + 1e-6
        
        normalized = (floor_points_2d - min_pt) / range_pt * (output_size - 50) + 25
        
        # Draw floor boundary
        if len(normalized) > 2:
            points = [tuple(p) for p in normalized]
            draw.polygon(points, fill='lightblue', outline='blue')
        
        # Draw walls
        wall_points = floor_plan.get('wall_points', [])
        if len(wall_points) > 0:
            wall_points_2d = wall_points[:, :2]
            wall_normalized = (wall_points_2d - min_pt) / range_pt * (output_size - 50) + 25
            
            for point in wall_normalized:
                x, y = point
                draw.ellipse([x-2, y-2, x+2, y+2], fill='gray')
        
        # Draw openings
        openings = floor_plan.get('openings', [])
        for opening in openings:
            center = np.array(opening['center'])
            center_norm = (center - min_pt) / range_pt * (output_size - 50) + 25
            size = opening['size'] / np.max(range_pt) * (output_size - 50)
            
            x, y = center_norm
            r = size / 2
            draw.ellipse([x-r, y-r, x+r, y+r], fill='red', outline='darkred')
        
        # Add text
        wall_length = measurements.get('wall_length', 0)
        room_area = measurements.get('room_area', 0)
        ceiling_height = measurements.get('ceiling_height', 0)
        
        text = f"Area: {room_area:.1f}m²\nPerimeter: {wall_length:.1f}m\nHeight: {ceiling_height:.1f}m"
        draw.text((10, 10), text, fill='black')
        
        return img
    
    @staticmethod
    def save_floor_plan(img, output_path):
        """Save floor plan visualization.
        
        Args:
            img: PIL Image
            output_path: Path to save
        """
        img.save(output_path)
        return output_path
