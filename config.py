"""
Configuration module for the surgery dataset generator.
"""
from dataclasses import dataclass, field
from typing import Tuple
import yaml
import os
import random

@dataclass
class Config:
    """Configuration for the surgery scene dataset generator."""
    
    # Scene composition
    num_doctors: int = 2
    num_assistants: int = 3
    num_active_doctors: int = 1
    num_active_assistants: int = 2
    
    # Generation parameters
    total_num_frames: int = 1000
    num_seperated_videos: int = 4 
    seed: int = 42
    
    # Image settings
    img_size: int = 224
    
    # Movement settings
    movement_speed: float = 3.0
    max_transition_pause: int = 60
    
    # Colors (RGB tuples)
    doctor_color: Tuple[int, int, int] = (59, 167, 255)
    assistant_color: Tuple[int, int, int] = (59, 255, 115)
    patient_table_color: Tuple[int, int, int] = (91, 91, 99)
    preparation_table_color: Tuple[int, int, int] = (255, 59, 222)
    instrument_color: Tuple[int, int, int] = (134, 134, 145)
    scene_object_color: Tuple[int, int, int] = (169, 169, 184)
    occlusion_object_color: Tuple[int, int, int] = (0, 0, 0) 
    background_color: Tuple[int, int, int] = (242, 238, 230)
    handover_highlight_color: Tuple[int, int, int] = (255, 59, 101)
    
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
    instrument_size: int = 10
    
    # Action durations (in frames)
    prepare_duration_avg: int = 20
    work_duration_avg: int = 25
    hold_duration_avg: int = 8
    handover_duration: int = 5
    allow_handover_overlap: bool = False 
    person_avoidance_radius_multiplier: float = 4.0  # Start avoiding at radius * this
    person_separation_buffer: float = 2.0  # Extra space between persons

    visualize_handover: bool = False

    occlusion_obj_appearance_prob: float = 0.01
    occlusion_obj_max_num: int = 3
    occlusion_obj_max_size: float = 0.5 # As a fraction of img_size

    instrument_hidden_prob: float = 0.01
    
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
                    'background_color', 'handover_highlight_color']:
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
        """
        Get patient table rectangle (x, y, width, height).
        x and y have random offsets
        """
        width = int(self.img_size * self.patient_table_width_ratio)
        height = int(self.img_size * self.patient_table_height_ratio)
        x = 0
        y = 0
        return (x, y, width, height)
    
    @property
    def preparation_table_rect(self) -> Tuple[int, int, int, int]:
        """Get preparation table rectangle (x, y, width, height)."""
        width = int(self.img_size * self.preparation_table_width_ratio)
        height = int(self.img_size * self.preparation_table_height_ratio)
        x = 0
        y = 0
        return (x, y, width, height)


def create_default_config(path: str = "config.yaml"):
    """Create a default configuration file."""
    config = Config()
    config.to_yaml(path)
    return config
