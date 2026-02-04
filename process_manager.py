"""
ProcessManager class that orchestrates the surgery scene simulation.
"""
from typing import List, Tuple, Optional
import numpy as np

from config import Config
from person import Person
from instrument import Instrument
from scene_object import PatientTable, PreparationTable, RandomMedicalObject, OcclusionObject
from enums import PersonType, AssistantState, DoctorState
from utils import random_point_outside_rects, distance, random_pos_near_center
from pathfinding_utils import PathfindingManager

class ProcessManager:
    """Manages the scene setup and state transitions."""
    
    # Available edges for table placement
    EDGES = ['top', 'bottom', 'left', 'right']
    
    def __init__(self, config: Config):
        """
        Initialize the process manager.
        
        Args:
            config: Scene configuration
        """
        self.config = config
        if config.seed != -1:
            self.rng = np.random.default_rng(config.seed)
        else:
            self.rng = np.random.default_rng()
        
        # Scene elements
        self.patient_table: Optional[PatientTable] = None
        self.preparation_table: Optional[PreparationTable] = None
        self.scene_objects: List[RandomMedicalObject] = []
        self.persons: List[Person] = []
        self.instruments: List[Instrument] = []
        self.occlusion_objects: List[OcclusionObject] = []
        
        # Track which edges are used by tables
        self.patient_table_edge: Optional[str] = None
        self.preparation_table_edge: Optional[str] = None
        
        # Track active persons
        self.active_doctors: List[Person] = []
        self.active_assistants: List[Person] = []
        
        # Pathfinding manager
        self.pathfinding: Optional[PathfindingManager] = None
        
        # Frame counter
        self.current_frame = 0
    
    def initialize_scene(self):
        """Set up the initial scene with all objects and persons."""
        self._create_tables()
        self._create_scene_objects()
        self._setup_pathfinding()  # Setup pathfinding after obstacles are created
        self._create_instruments()
        self._create_persons()
        self._assign_active_status()
        self._set_person_references()
        self._position_persons_initially()
    
    def _create_tables(self):
        """Create the patient and preparation tables on different edges."""
        # Randomly select edge for patient table
        available_edges = self.EDGES.copy()
        self.patient_table_edge = self.rng.choice(available_edges)
        available_edges.remove(self.patient_table_edge)
        
        # Randomly select different edge for preparation table
        self.preparation_table_edge = self.rng.choice(available_edges)
        
        # Create patient table
        pt_rect = self.config.patient_table_rect
        self.patient_table = PatientTable(
            pt_rect[0], pt_rect[1], pt_rect[2], pt_rect[3],
            self.config.patient_table_color
        )
        # Position on its assigned edge
        patient_pos = self.patient_table.position_on_edge(
            self.patient_table_edge, 
            self.config.img_size,
        )
        self.patient_table.position = patient_pos
        
        # Create preparation table
        prep_rect = self.config.preparation_table_rect
        self.preparation_table = PreparationTable(
            prep_rect[0], prep_rect[1], prep_rect[2], prep_rect[3],
            self.config.preparation_table_color
        )
        # Position on its assigned edge
        prep_pos = self.preparation_table.position_on_edge(
            self.preparation_table_edge,
            self.config.img_size,
        )
        self.preparation_table.position = prep_pos
        
        # Re-initialize instrument positions after positioning the table
        self.preparation_table._init_instrument_positions()
    
    def _create_scene_objects(self):
        """Create random medical objects at scene edges (avoiding table edges)."""
        edge_margin = 5
        
        # Determine which edges are free (not used by tables)
        used_edges = {self.patient_table_edge, self.preparation_table_edge}
        free_edges = [e for e in self.EDGES if e not in used_edges]
        
        # Define edge zones based on free edges
        zones = []
        for edge in free_edges:
            if edge == 'top':
                zones.append((edge_margin, edge_margin, 
                             self.config.img_size - 2 * edge_margin, 40))
            elif edge == 'bottom':
                zones.append((edge_margin, self.config.img_size - 40,
                             self.config.img_size - 2 * edge_margin, 20))
            elif edge == 'left':
                zones.append((edge_margin, 80, 40, self.config.img_size - 100))
            elif edge == 'right':
                zones.append((self.config.img_size - 60, 80, 40, self.config.img_size - 100))
        
        # If no free edges, use corners or minimal zones
        if not zones:
            # Fallback: place objects in corners
            zones = [
                (edge_margin, edge_margin, 30, 30),  # Top-left corner
                (self.config.img_size - 40, edge_margin, 30, 30),  # Top-right corner
                (edge_margin, self.config.img_size - 40, 30, 30),  # Bottom-left corner
                (self.config.img_size - 40, self.config.img_size - 40, 30, 30),  # Bottom-right corner
            ]
        
        for i in range(self.config.num_scene_objects):
            # Pick a random zone
            zone = zones[self.rng.integers(0, len(zones))]
            zx, zy, zw, zh = zone
            
            # Random size
            width = self.rng.integers(15, 40)
            height = self.rng.integers(15, 40)
            
            # Random position within zone
            x = self.rng.integers(zx, max(zx + 1, zx + zw - width))
            y = self.rng.integers(zy, max(zy + 1, zy + zh - height))
            
            obj = RandomMedicalObject(
                x, y, width, height,
                self.config.scene_object_color, i
            )
            self.scene_objects.append(obj)
    
    def _setup_pathfinding(self):
        """Initialize the pathfinding manager with obstacles."""
        self.pathfinding = PathfindingManager(
            self.config.img_size, 
            cell_size=8  # 8 pixel grid cells for good balance of precision and speed
        )
        
        # Collect all obstacles
        all_obstacles = self.scene_objects.copy()
        all_obstacles.append(self.patient_table)
        all_obstacles.append(self.preparation_table)
        
        # Update pathfinding grid with obstacles
        self.pathfinding.update_obstacles(all_obstacles, self.config.person_radius)
    
    def _create_instruments(self):
        """Create instruments on the preparation table."""
        for i in range(self.config.num_instruments):
            pos = self.preparation_table.get_instrument_position(i)
            instrument = Instrument(i, pos, self.config.instrument_size)
            self.instruments.append(instrument)
    
    def _create_persons(self):
        """Create all doctors and assistants."""
        person_id = 0
        
        # Create doctors
        for i in range(self.config.num_doctors):
            person = Person(
                person_id, PersonType.DOCTOR,
                (0, 0),  # Position set later
                self.config, self.rng
            )
            person.set_tables(self.patient_table, self.preparation_table,
                            self.scene_objects, self.pathfinding)
            self.persons.append(person)
            person_id += 1
        
        # Create assistants
        for i in range(self.config.num_assistants):
            person = Person(
                person_id, PersonType.ASSISTANT,
                (0, 0),  # Position set later
                self.config, self.rng
            )
            person.set_tables(self.patient_table, self.preparation_table,
                            self.scene_objects, self.pathfinding)
            self.persons.append(person)
            person_id += 1

    def _set_person_references(self):
        """Give each person a reference to all other persons for collision avoidance."""
        for person in self.persons:
            person.set_all_persons(self.persons)
    
    def _assign_active_status(self):
        """Assign is_active status to doctors and assistants."""
        doctors = [p for p in self.persons if p.person_type == PersonType.DOCTOR]
        assistants = [p for p in self.persons if p.person_type == PersonType.ASSISTANT]
        
        # Randomly select active doctors
        if doctors:
            num_active = min(self.config.num_active_doctors, len(doctors))
            active_indices = self.rng.choice(
                len(doctors), size=num_active, replace=False)
            for idx in active_indices:
                doctors[idx].set_active(True)
                self.active_doctors.append(doctors[idx])
        
        # Randomly select active assistants
        if assistants:
            num_active = min(self.config.num_active_assistants, len(assistants))
            active_indices = self.rng.choice(
                len(assistants), size=num_active, replace=False)
            for idx in active_indices:
                assistants[idx].set_active(True)
                self.active_assistants.append(assistants[idx])
    
    def _position_persons_initially(self):
        """Position all persons at valid starting locations."""
        # Get all obstacle rectangles
        obstacles = [obj.rect for obj in self.scene_objects]
        obstacles.append(self.patient_table.rect)
        obstacles.append(self.preparation_table.rect)
        
        bounds = (0, 0, self.config.img_size, self.config.img_size)
        
        for person in self.persons:
            person_pos = random_pos_near_center(bounds, self.rng)
            if person.is_active:
                if person.person_type == PersonType.DOCTOR:
                    # Position active doctors near patient table
                    person._set_target_near_patient_table()
                    person.position = person_pos
                else:
                    # Position active assistants near preparation table
                    person._set_target_near_prep_table()
                    person.position = person_pos
            else:
                # Position inactive persons randomly
                random_pos = random_point_outside_rects(
                    obstacles, bounds, person.radius, self.rng)
                
                if random_pos:
                    person.position = random_pos
                    person.original_position = random_pos
                else:
                    # Fallback position
                    person.position = (self.config.img_size * 0.2,
                                       self.config.img_size * 0.2)
                    person.original_position = person.position
    
    def update(self):
        """Update the scene for one frame."""
        # Get available instruments (on table)
        available_instruments = [
            inst for inst in self.instruments 
            if inst.holder is None
        ]
        
        # Update all persons
        for person in self.persons:
            person.update(available_instruments, 
                         self.active_doctors, self.active_assistants)
        
        # Update instruments
        for instrument in self.instruments:
            instrument.update()
        
        # Dynamic active status changes (occasionally)
        if self.rng.random() < 0.001:  # Very rare
            self._maybe_swap_active_status()

        # Randomly add occlusion objects
        self.occlusion_objects = []
        occluded_persons = []

        for _ in range(self.config.occlusion_obj_max_num):
            if self.rng.random() < self.config.occlusion_obj_appearance_prob:
                max_size = int(self.config.img_size * self.config.occlusion_obj_max_size)
                width = self.rng.integers(20, max_size)
                height = self.rng.integers(20, max_size)

                person_to_occlude = self.rng.choice(self.persons)
                # if person already occluded, pick another
                while person_to_occlude in occluded_persons and len(occluded_persons) < len(self.persons):
                    person_to_occlude = self.rng.choice(self.persons)

                occluded_persons.append(person_to_occlude)
                person_pos = person_to_occlude.position
                x = person_pos[0] - width // 2
                y = person_pos[1] - height // 2

                occlusion_obj = OcclusionObject(
                    x, y, width, height,
                    self.config.occlusion_object_color,
                    len(self.occlusion_objects)
                )
                self.occlusion_objects.append(occlusion_obj)

        # Randomly hide instruments (occlusion)
        for instrument in self.instruments:
            instrument.opacity = 1.0  # Default visible
            if instrument.holder is not None:
                if self.rng.random() < self.config.instrument_hidden_prob:
                    instrument.opacity = 0.0
        
        self.current_frame += 1
    
    def _maybe_swap_active_status(self):
        """Occasionally swap which persons are active."""
        # Only swap if persons are in idle state
        doctors = [p for p in self.persons if p.person_type == PersonType.DOCTOR]
        assistants = [p for p in self.persons if p.person_type == PersonType.ASSISTANT]
        
        # Check if we can swap a doctor
        idle_active_doctors = [
            d for d in self.active_doctors 
            if d.state == DoctorState.IDLE and d.held_instrument is None
        ]
        inactive_doctors = [d for d in doctors if not d.is_active]
        
        if idle_active_doctors and inactive_doctors:
            if self.rng.random() < 0.3:
                old_active = self.rng.choice(idle_active_doctors)
                new_active = self.rng.choice(inactive_doctors)
                
                old_active.set_active(False)
                new_active.set_active(True)
                
                self.active_doctors.remove(old_active)
                self.active_doctors.append(new_active)
        
        # Similar for assistants
        idle_active_assistants = [
            a for a in self.active_assistants
            if a.state == AssistantState.IDLE and a.held_instrument is None
        ]
        inactive_assistants = [a for a in assistants if not a.is_active]
        
        if idle_active_assistants and inactive_assistants:
            if self.rng.random() < 0.3:
                old_active = self.rng.choice(idle_active_assistants)
                new_active = self.rng.choice(inactive_assistants)
                
                old_active.set_active(False)
                new_active.set_active(True)
                
                self.active_assistants.remove(old_active)
                self.active_assistants.append(new_active)
    
    def get_scene_state(self):
        """Get current state of all scene elements."""
        return {
            'frame': self.current_frame,
            'persons': self.persons,
            'instruments': self.instruments,
            'patient_table': self.patient_table,
            'preparation_table': self.preparation_table,
            'scene_objects': self.scene_objects,
            'occlusion_objects': self.occlusion_objects,
            'patient_table_edge': self.patient_table_edge,
            'preparation_table_edge': self.preparation_table_edge
        }