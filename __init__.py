"""
Surgery Dataset Generator

A synthetic dataset generator for training action detection models
on medical instrument handover scenarios.
"""

from .config import Config
from .enums import PersonType, AssistantState, DoctorState, ActionLabel, LABEL_NAMES
from .instrument import Instrument
from .person import Person
from .scene_object import SceneObject, PatientTable, PreparationTable, RandomMedicalObject
from .process_manager import ProcessManager
from .render_manager import RenderManager
from .annotation_manager import AnnotationManager
from .dataset_generator import DatasetGenerator
from .dataset_player import DatasetPlayer
from .pathfinding_utils import PathfindingManager, PathfindingGrid

__version__ = '1.2.0'
__all__ = [
    'Config',
    'PersonType', 'AssistantState', 'DoctorState', 'ActionLabel', 'LABEL_NAMES',
    'Instrument',
    'Person',
    'SceneObject', 'PatientTable', 'PreparationTable', 'RandomMedicalObject',
    'ProcessManager',
    'RenderManager',
    'AnnotationManager',
    'DatasetGenerator',
    'DatasetPlayer',
    'PathfindingManager', 'PathfindingGrid',
]
