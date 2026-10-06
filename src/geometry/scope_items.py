"""Generate scope of work items."""
import numpy as np
from src.geometry.damage_detection import DamageClass


class ScopeItemGenerator:
    """Generate scope items from measurements and damage."""
    
    # Cost estimates (USD per unit)
    COST_ESTIMATES = {
        'drywall_repair': 50,  # per sq ft
        'paint': 15,  # per sq ft
        'floor_repair': 75,  # per sq ft
        'ceiling_repair': 60,  # per sq ft
        'trim_replacement': 20,  # per linear ft
        'opening_repair': 200,  # per opening
    }
    
    def generate_scope_item(self, item_type, quantity, unit):
        """Generate a single scope item.
        
        Args:
            item_type: Type of work (drywall_repair, paint, etc.)
            quantity: Quantity of work
            unit: Unit (sq ft, linear ft, count, etc.)
            
        Returns:
            Scope item dict
        """
        cost_per_unit = self.COST_ESTIMATES.get(item_type, 100)
        total_cost = quantity * cost_per_unit
        
        return {
            'type': item_type,
            'quantity': quantity,
            'unit': unit,
            'cost_per_unit': cost_per_unit,
            'total_cost': total_cost
        }
    
    def generate_scope(self, measurements, damage_detections):
        """Generate complete scope of work.
        
        Args:
            measurements: Measurements dict from MeasurementExtractor
            damage_detections: Damage dict from DamageDetector
            
        Returns:
            Scope of work dict
        """
        scope_items = []
        
        # Base items from measurements
        wall_length = measurements.get('wall_length', 0)
        room_area = measurements.get('room_area', 0)
        
        # Wall repair if damage detected
        structural_damage_count = sum(
            1 for d in damage_detections.get('damage_detections', [])
            if d['class'] == DamageClass.STRUCTURAL
        )
        
        if structural_damage_count > 0:
            # Estimate affected area
            affected_area = structural_damage_count * 0.5  # ~0.5 sq ft per damage
            scope_items.append(self.generate_scope_item(
                'drywall_repair', affected_area, 'sq ft'
            ))
        
        # Paint (assuming room needs painting)
        # Convert m2 to sq ft (1 m2 = 10.764 sq ft)
        if room_area > 0:
            room_area_sqft = room_area * 10.764
            scope_items.append(self.generate_scope_item(
                'paint', room_area_sqft, 'sq ft'
            ))
        
        # Opening repairs
        num_openings = len(damage_detections.get('damage_detections', []))
        if num_openings > 0:
            scope_items.append(self.generate_scope_item(
                'opening_repair', num_openings, 'count'
            ))
        
        # Trim replacement if walls damaged
        if wall_length > 0 and structural_damage_count > 0:
            scope_items.append(self.generate_scope_item(
                'trim_replacement', wall_length, 'linear ft'
            ))
        
        # Compute total
        total_cost = sum(item['total_cost'] for item in scope_items)
        
        return {
            'scope_items': scope_items,
            'total_cost': total_cost,
            'item_count': len(scope_items)
        }
