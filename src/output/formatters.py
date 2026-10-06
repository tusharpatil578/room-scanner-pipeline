"""Output formatters for scan results."""
import json
from datetime import datetime
from pathlib import Path


class JSONFormatter:
    """Format scan results as JSON."""
    
    @staticmethod
    def format_scan_result(
        scan_id,
        floor_plan,
        measurements,
        damage_detections,
        scope,
        registration_quality=None
    ):
        """Format complete scan result as JSON.
        
        Args:
            scan_id: Unique scan identifier
            floor_plan: FloorPlanExtractor output
            measurements: MeasurementExtractor output
            damage_detections: DamageDetector output
            scope: ScopeItemGenerator output
            registration_quality: Registration metrics
            
        Returns:
            JSON-serializable dict
        """
        result = {
            'metadata': {
                'scan_id': scan_id,
                'timestamp': datetime.now().isoformat(),
                'version': '1.0'
            },
            'geometry': {
                'floor_plan': {
                    'num_openings': floor_plan.get('num_openings', 0),
                    'opening_details': [
                        {
                            'center': [float(o['center'][0]), float(o['center'][1])],
                            'size': float(o['size'])
                        }
                        for o in floor_plan.get('openings', [])
                    ]
                },
                'measurements': {
                    'wall_length_m': float(measurements.get('wall_length', 0)),
                    'room_area_m2': float(measurements.get('room_area', 0)),
                    'ceiling_height_m': float(measurements.get('ceiling_height', 0)),
                    'floor_z_m': float(measurements.get('floor_z', 0))
                }
            },
            'damage': {
                'num_regions': damage_detections.get('num_damage_regions', 0),
                'discontinuity_count': int(damage_detections.get('discontinuity_count', 0)),
                'detections': [
                    {
                        'class': d['class'].name,
                        'severity': int(d['class'].value),
                        'cluster_size': int(d['cluster_size']),
                        'spread_m': float(d['spread'])
                    }
                    for d in damage_detections.get('damage_detections', [])
                ]
            },
            'scope': {
                'total_cost_usd': float(scope.get('total_cost', 0)),
                'item_count': int(scope.get('item_count', 0)),
                'items': [
                    {
                        'type': item['type'],
                        'quantity': float(item['quantity']),
                        'unit': item['unit'],
                        'cost_per_unit': float(item['cost_per_unit']),
                        'total_cost': float(item['total_cost'])
                    }
                    for item in scope.get('scope_items', [])
                ]
            }
        }
        
        # Add registration quality if available
        if registration_quality:
            result['registration'] = {
                'final_error_m': float(registration_quality.get('final_error_m', 0)),
                'num_iterations': int(registration_quality.get('num_iterations', 0)),
                'passes_1cm_gate': bool(registration_quality.get('passes_1cm_gate', False))
            }
        
        return result
    
    @staticmethod
    def save_json(data, output_path):
        """Save result to JSON file.
        
        Args:
            data: JSON-serializable dict
            output_path: Path to output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return output_path
