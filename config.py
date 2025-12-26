"""
Configuration module for the surgery dataset generator.
"""
from dataclasses import dataclass, field
from typing import Tuple
import yaml
import os


@dataclass
class Config:
    """Configuration for the surgery scene dataset generator."""
    
    # Scene composition
    num_doctors: int = 2
    num_assistants: int = 3
    num_active_doctors: int = 1
    num_active_assistants: int = 2
    
    # Generation parameters
    total_num_frames: int = 500
    handover_rate: float = 0.02  # ~every 50 frames
    seed: int = 42
    
    # Image settings
    img_size: int = 448
    
    # Movement settings
    movement_speed: float = 3.0
    max_transition_pause: int = 100
    
    # Colors (RGB tuples)
    doctor_color: Tuple[int, int, int] = (70, 130, 180)  # Steel blue
    assistant_color: Tuple[int, int, int] = (60, 179, 113)  # Medium sea green
    patient_table_color: Tuple[int, int, int] = (147, 112, 219)  # Medium purple
    preparation_table_color: Tuple[int, int, int] = (255, 182, 193)  # Light pink
    instrument_color: Tuple[int, int, int] = (169, 169, 169)  # Dark gray / silver
    scene_object_color: Tuple[int, int, int] = (128, 128, 128)  # Gray
    background_color: Tuple[int, int, int] = (240, 240, 240)  # Light gray
    
    # Scene objects
    num_scene_objects: int = 8
    num_instruments: int = 5
    
    # Person dimensions
    person_radius: int = 20
    
    # Table dimensions (relative to img_size)
    patient_table_width_ratio: float = 0.3
    patient_table_height_ratio: float = 0.4
    preparation_table_width_ratio: float = 0.25
    preparation_table_height_ratio: float = 0.1
    
    # Instrument dimensions
    instrument_size: int = 12
    
    # Action durations (in frames)
    prepare_duration_avg: int = 20
    work_duration_avg: int = 25
    hold_duration_avg: int = 8
    handover_duration: int = 5
    
    # Output settings
    output_dir: str = "output"
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from a YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Convert color lists to tuples
        for key in ['doctor_color', 'assistant_color', 'patient_table_color', 
                    'preparation_table_color', 'instrument_color', 'scene_object_color',
                    'background_color']:
            if key in data and isinstance(data[key], list):
                data[key] = tuple(data[key])
        
        return cls(**data)
    
    def to_yaml(self, path: str):
        """Save configuration to a YAML file."""
        data = {}
        for key, value in self.__dict__.items():
            if isinstance(value, tuple):
                data[key] = list(value)
            else:
                data[key] = value
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    @property
    def patient_table_rect(self) -> Tuple[int, int, int, int]:
        """Get patient table rectangle (x, y, width, height)."""
        width = int(self.img_size * self.patient_table_width_ratio)
        height = int(self.img_size * self.patient_table_height_ratio)
        x = (self.img_size - width) // 2
        y = (self.img_size - height) // 2
        return (x, y, width, height)
    
    @property
    def preparation_table_rect(self) -> Tuple[int, int, int, int]:
        """Get preparation table rectangle (x, y, width, height)."""
        patient_rect = self.patient_table_rect
        width = int(self.img_size * self.preparation_table_width_ratio)
        height = int(self.img_size * self.preparation_table_height_ratio)
        x = (self.img_size - width) // 2
        y = patient_rect[1] + patient_rect[3]  # Below patient table
        return (x, y, width, height)


def create_default_config(path: str = "config.yaml"):
    """Create a default configuration file."""
    config = Config()
    config.to_yaml(path)
    return config
