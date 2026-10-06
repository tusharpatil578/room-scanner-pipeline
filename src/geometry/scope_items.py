"""Generate repair scope line items from damage."""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from src.geometry.damage_detection import DamageClass


@dataclass
class ScopeItemGenerator:
    """Generate repair scope line items."""
    
    MATERIAL_COSTS = {
        DamageClass.COSMETIC: 5.0,
        DamageClass.STRUCTURAL: 50.0,
        DamageClass.CONCEALED: 100.0,
    }
    
    LABOR_HOURS = {
        DamageClass.COSMETIC: 0.5,
        DamageClass.STRUCTURAL: 4.0,
        DamageClass.CONCEALED: 8.0,
    }
    
    @staticmethod
    def generate_scope_item(damage: Dict) -> Dict:
        """
        Generate a repair scope line item from damage.
        
        Args:
            damage: Damage dict with classification
        
        Returns:
            Scope line item
        """
        damage_class = damage['class']
        size = damage['size']
        
        # Estimate quantity
        quantity = max(1, int(np.ceil(size / 0.5)))
        
        # Get base costs
        material_cost = ScopeItemGenerator.MATERIAL_COSTS.get(damage_class, 0)
        labor_hours = ScopeItemGenerator.LABOR_HOURS.get(damage_class, 0)
        
        # Scale by size
        material_cost *= quantity
        labor_hours *= quantity
        
        labor_cost = labor_hours * 75.0  # $75/hour
        
        scope_item = {
            'description': f"{damage_class.value} damage - {quantity} units",
            'quantity': quantity,
            'unit': 'area',
            'material_cost': float(material_cost),
            'labor_hours': float(labor_hours),
            'labor_cost': float(labor_cost),
            'total_cost': float(material_cost + labor_cost),
            'damage_class': damage_class.value,
        }
        
        return scope_item
    
    @staticmethod
    def generate_scope(damages: List[Dict]) -> Dict:
        """
        Generate complete scope of work.
        
        Args:
            damages: List of damage dicts
        
        Returns:
            Scope dict with line items and totals
        """
        scope_items = []
        
        for damage in damages:
            item = ScopeItemGenerator.generate_scope_item(damage)
            scope_items.append(item)
        
        # Compute totals
        total_material = sum(item['material_cost'] for item in scope_items)
        total_labor_cost = sum(item['labor_cost'] for item in scope_items)
        total_labor_hours = sum(item['labor_hours'] for item in scope_items)
        
        scope = {
            'line_items': scope_items,
            'summary': {
                'total_items': len(scope_items),
                'total_material_cost': float(total_material),
                'total_labor_hours': float(total_labor_hours),
                'total_labor_cost': float(total_labor_cost),
                'total_cost': float(total_material + total_labor_cost),
            }
        }
        
        return scope