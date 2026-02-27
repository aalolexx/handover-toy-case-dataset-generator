"""
ProcessManager class that orchestrates the surgery scene simulation.
Supports both grid-based and continuous movement modes.
"""
from typing import List, Tuple, Optional
import numpy as np

from config import Config
from person import Person
from instrument import Instrument
from scene_object import PatientTable, PreparationTable, RandomMedicalObject, OcclusionObject
from enums import PersonType, AssistantState, DoctorState


class ProcessManager:
    """Manages the scene setup and state transitions."""
    
    EDGES = ['top', 'bottom', 'left', 'right']
    
    def __init__(self, config: Config):
        self.config = config
        self.use_grid = config.use_grid_movement
        
        if config.seed != -1:
            self.rng = np.random.default_rng(config.seed)
        else:
            self.rng = np.random.default_rng()
        
        # Initialize movement system based on mode
        self.grid = None
        self.pathfinding = None
        
        if self.use_grid:
            from grid_manager import GridManager
            self.grid = GridManager(config.img_size, config.grid_size)
        else:
            from pathfinding_utils import PathfindingManager
            self.pathfinding = PathfindingManager(config.img_size, cell_size=8)
        
        # Scene elements
        self.patient_table: Optional[PatientTable] = None
        self.preparation_table: Optional[PreparationTable] = None
        self.scene_objects: List[RandomMedicalObject] = []
        self.persons: List[Person] = []
        self.instruments: List[Instrument] = []
        self.occlusion_objects: List[OcclusionObject] = []
        
        self.patient_table_edge: Optional[str] = None
        self.preparation_table_edge: Optional[str] = None
        
        self.active_doctors: List[Person] = []
        self.active_assistants: List[Person] = []
        
        self.current_frame = 0
    
    def initialize_scene(self):
        """Set up the initial scene with all objects and persons."""
        self._create_tables()
        self._create_scene_objects()
        self._register_obstacles()
        self._create_instruments()
        self._create_persons()
        self._assign_active_status()
        self._set_person_references()
        self._position_persons_initially()
    
    def _create_tables(self):
        """Create the patient and preparation tables on different edges."""
        available_edges = self.EDGES.copy()
        self.patient_table_edge = self.rng.choice(available_edges)
        available_edges.remove(self.patient_table_edge)
        self.preparation_table_edge = self.rng.choice(available_edges)
        
        # Create patient table
        pt_rect = self.config.patient_table_rect
        
        if self.use_grid:
            cell_size = self.config.cell_size
            pt_width_cells = max(1, (pt_rect[2] + cell_size - 1) // cell_size)
            pt_height_cells = max(1, (pt_rect[3] + cell_size - 1) // cell_size)
            self.patient_table = PatientTable(
                0, 0,
                pt_width_cells * cell_size,
                pt_height_cells * cell_size,
                self.config.patient_table_color
            )
        else:
            self.patient_table = PatientTable(
                0, 0, pt_rect[2], pt_rect[3],
                self.config.patient_table_color
            )
        
        patient_pos = self.patient_table.position_on_edge(
            self.patient_table_edge, 
            self.config.img_size,
            margin=10.0,
            use_grid=self.use_grid,
            grid_size=self.config.grid_size
        )
        self.patient_table.position = patient_pos
        
        # Create preparation table
        prep_rect = self.config.preparation_table_rect
        
        if self.use_grid:
            prep_width_cells = max(1, (prep_rect[2] + cell_size - 1) // cell_size)
            prep_height_cells = max(1, (prep_rect[3] + cell_size - 1) // cell_size)
            self.preparation_table = PreparationTable(
                0, 0,
                prep_width_cells * cell_size,
                prep_height_cells * cell_size,
                self.config.preparation_table_color
            )
        else:
            self.preparation_table = PreparationTable(
                0, 0, prep_rect[2], prep_rect[3],
                self.config.preparation_table_color
            )
        
        prep_pos = self.preparation_table.position_on_edge(
            self.preparation_table_edge,
            self.config.img_size,
            margin=10.0,
            use_grid=self.use_grid,
            grid_size=self.config.grid_size
        )
        self.preparation_table.position = prep_pos
        
        # Initialize instrument positions
        self.preparation_table._init_instrument_positions(
            use_grid=self.use_grid,
            grid_size=self.config.grid_size,
            img_size=self.config.img_size
        )
    
    def _create_scene_objects(self):
        """Create random medical objects at scene edges."""
        used_edges = {self.patient_table_edge, self.preparation_table_edge}
        free_edges = [e for e in self.EDGES if e not in used_edges]
        
        if self.use_grid:
            self._create_scene_objects_grid(free_edges)
        else:
            self._create_scene_objects_continuous(free_edges)
    
    def _create_scene_objects_grid(self, free_edges: List[str]):
        """Create scene objects with grid alignment."""
        cell_size = self.config.cell_size
        grid_size = self.config.grid_size
        
        zones = []
        for edge in free_edges:
            if edge == 'top':
                zones.append((0, 0, 2, grid_size))
            elif edge == 'bottom':
                zones.append((grid_size - 2, 0, 2, grid_size))
            elif edge == 'left':
                zones.append((2, 0, grid_size - 4, 2))
            elif edge == 'right':
                zones.append((2, grid_size - 2, grid_size - 4, 2))
        
        if not zones:
            zones = [
                (0, 0, 2, 2),
                (0, grid_size - 2, 2, 2),
                (grid_size - 2, 0, 2, 2),
                (grid_size - 2, grid_size - 2, 2, 2),
            ]
        
        for i in range(self.config.num_scene_objects):
            zone = zones[self.rng.integers(0, len(zones))]
            zone_row, zone_col, zone_h, zone_w = zone
            
            width_cells = self.rng.integers(1, 3)
            height_cells = self.rng.integers(1, 3)
            
            max_col = zone_col + zone_w - width_cells
            max_row = zone_row + zone_h - height_cells
            
            col = self.rng.integers(zone_col, max(zone_col + 1, max_col + 1))
            row = self.rng.integers(zone_row, max(zone_row + 1, max_row + 1))
            
            obj = RandomMedicalObject(
                col * cell_size, row * cell_size,
                width_cells * cell_size, height_cells * cell_size,
                self.config.scene_object_color, i
            )
            self.scene_objects.append(obj)
    
    def _create_scene_objects_continuous(self, free_edges: List[str]):
        """Create scene objects with continuous positioning."""
        edge_margin = 5
        
        zones = []
        for edge in free_edges:
            if edge == 'top':
                zones.append((edge_margin, edge_margin, 
                             self.config.img_size - 2 * edge_margin, 40))
            elif edge == 'bottom':
                zones.append((edge_margin, self.config.img_size - 40,
                             self.config.img_size - 2 * edge_margin, 30))
            elif edge == 'left':
                zones.append((edge_margin, 60, 40, self.config.img_size - 80))
            elif edge == 'right':
                zones.append((self.config.img_size - 50, 60, 40, self.config.img_size - 80))
        
        if not zones:
            zones = [
                (edge_margin, edge_margin, 30, 30),
                (self.config.img_size - 40, edge_margin, 30, 30),
                (edge_margin, self.config.img_size - 40, 30, 30),
                (self.config.img_size - 40, self.config.img_size - 40, 30, 30),
            ]
        
        for i in range(self.config.num_scene_objects):
            zone = zones[self.rng.integers(0, len(zones))]
            zx, zy, zw, zh = zone
            
            width = self.rng.integers(15, 40)
            height = self.rng.integers(15, 40)
            
            x = self.rng.integers(zx, max(zx + 1, zx + zw - width))
            y = self.rng.integers(zy, max(zy + 1, zy + zh - height))
            
            obj = RandomMedicalObject(
                x, y, width, height,
                self.config.scene_object_color, i
            )
            self.scene_objects.append(obj)
    
    def _register_obstacles(self):
        """Register obstacles with the movement system."""
        if self.use_grid:
            self.grid.add_table(self.patient_table.rect, is_patient_table=True)
            self.grid.add_table(self.preparation_table.rect, is_prep_table=True)
            for obj in self.scene_objects:
                self.grid.add_obstacle(obj.rect)
        else:
            all_obstacles = self.scene_objects.copy()
            all_obstacles.append(self.patient_table)
            all_obstacles.append(self.preparation_table)
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
        
        for i in range(self.config.num_doctors):
            person = Person(
                person_id, PersonType.DOCTOR,
                (0, 0),
                self.config, self.rng
            )
            if self.use_grid:
                person.set_grid(self.grid)
            else:
                person.set_pathfinding(self.pathfinding)
            person.set_tables(self.patient_table, self.preparation_table, self.scene_objects)
            self.persons.append(person)
            person_id += 1
        
        for i in range(self.config.num_assistants):
            person = Person(
                person_id, PersonType.ASSISTANT,
                (0, 0),
                self.config, self.rng
            )
            if self.use_grid:
                person.set_grid(self.grid)
            else:
                person.set_pathfinding(self.pathfinding)
            person.set_tables(self.patient_table, self.preparation_table, self.scene_objects)
            self.persons.append(person)
            person_id += 1
    
    def _set_person_references(self):
        """Give each person a reference to all other persons."""
        for person in self.persons:
            person.set_all_persons(self.persons)
    
    def _assign_active_status(self):
        """Assign is_active status to doctors and assistants."""
        doctors = [p for p in self.persons if p.person_type == PersonType.DOCTOR]
        assistants = [p for p in self.persons if p.person_type == PersonType.ASSISTANT]
        
        if doctors:
            num_active = min(self.config.num_active_doctors, len(doctors))
            active_indices = self.rng.choice(len(doctors), size=num_active, replace=False)
            for idx in active_indices:
                doctors[idx].set_active(True)
                self.active_doctors.append(doctors[idx])
        
        if assistants:
            num_active = min(self.config.num_active_assistants, len(assistants))
            active_indices = self.rng.choice(len(assistants), size=num_active, replace=False)
            for idx in active_indices:
                assistants[idx].set_active(True)
                self.active_assistants.append(assistants[idx])
    
    def _position_persons_initially(self):
        """Position all persons at valid starting locations."""
        if self.use_grid:
            self._position_persons_grid()
        else:
            self._position_persons_continuous()
    
    def _position_persons_grid(self):
        """Position persons on grid."""
        placed = set()
        
        for person in self.persons:
            if person.is_active:
                if person.person_type == PersonType.DOCTOR:
                    target = self.grid.find_cell_adjacent_to_table(
                        self.grid.patient_table_cells, (8, 8),
                        self.persons, person.id)
                else:
                    target = self.grid.find_cell_adjacent_to_table(
                        self.grid.prep_table_cells, (8, 8),
                        self.persons, person.id)
                
                if target and target not in placed:
                    person.grid_pos = target
                    person.home_pos = target
                    placed.add(target)
                else:
                    pos = self._find_free_grid_cell(placed)
                    if pos:
                        person.grid_pos = pos
                        person.home_pos = pos
                        placed.add(pos)
            else:
                pos = self._find_free_grid_cell(placed)
                if pos:
                    person.grid_pos = pos
                    person.home_pos = pos
                    placed.add(pos)
    
    def _find_free_grid_cell(self, excluded: set) -> Optional[Tuple[int, int]]:
        """Find a free walkable cell."""
        for _ in range(100):
            row = self.rng.integers(0, self.config.grid_size)
            col = self.rng.integers(0, self.config.grid_size)
            pos = (row, col)
            if self.grid.is_walkable(pos) and pos not in excluded:
                return pos
        
        for row in range(self.config.grid_size):
            for col in range(self.config.grid_size):
                pos = (row, col)
                if self.grid.is_walkable(pos) and pos not in excluded:
                    return pos
        
        return None
    
    def _position_persons_continuous(self):
        """Position persons in continuous space."""
        from utils import random_point_outside_rects
        
        obstacles = [obj.rect for obj in self.scene_objects]
        obstacles.append(self.patient_table.rect)
        obstacles.append(self.preparation_table.rect)
        bounds = (0, 0, self.config.img_size, self.config.img_size)
        
        for person in self.persons:
            if person.is_active:
                if person.person_type == PersonType.DOCTOR:
                    # Position near patient table
                    pos = self._find_pos_near_table_continuous(self.patient_table)
                else:
                    # Position near prep table
                    pos = self._find_pos_near_table_continuous(self.preparation_table)
                person.position = pos
                person.original_position = pos
            else:
                pos = random_point_outside_rects(
                    obstacles, bounds, person.radius, self.rng)
                if pos:
                    person.position = pos
                    person.original_position = pos
                else:
                    person.position = (self.config.img_size * 0.2,
                                       self.config.img_size * 0.2)
                    person.original_position = person.position
    
    def _find_pos_near_table_continuous(self, table) -> Tuple[float, float]:
        """Find a position near a table."""
        margin = self.config.person_radius + 10
        sides = []
        
        if table.x > margin + self.config.person_radius:
            sides.append('left')
        if table.x + table.width < self.config.img_size - margin - self.config.person_radius:
            sides.append('right')
        if table.y > margin + self.config.person_radius:
            sides.append('top')
        if table.y + table.height < self.config.img_size - margin - self.config.person_radius:
            sides.append('bottom')
        
        if not sides:
            sides = ['left', 'right', 'top', 'bottom']
        
        side = self.rng.choice(sides)
        
        if side == 'left':
            x = table.x - margin
            y = table.y + table.height / 2
        elif side == 'right':
            x = table.x + table.width + margin
            y = table.y + table.height / 2
        elif side == 'top':
            x = table.x + table.width / 2
            y = table.y - margin
        else:
            x = table.x + table.width / 2
            y = table.y + table.height + margin
        
        # Clamp to bounds
        r = self.config.person_radius
        x = max(r + 5, min(self.config.img_size - r - 5, x))
        y = max(r + 5, min(self.config.img_size - r - 5, y))
        
        return (x, y)
    
    def update(self):
        """Update the scene for one frame."""
        available_instruments = [
            inst for inst in self.instruments 
            if inst.holder is None
        ]
        
        for person in self.persons:
            person.update(available_instruments, 
                         self.active_doctors, self.active_assistants)
        
        for instrument in self.instruments:
            instrument.update()
        
        if self.rng.random() < 0.001:
            self._maybe_swap_active_status()
        
        # Occlusion objects
        self.occlusion_objects = []
        occluded_persons = []
        
        for _ in range(self.config.occlusion_obj_max_num):
            if self.rng.random() < self.config.occlusion_obj_appearance_prob:
                max_size = int(self.config.img_size * self.config.occlusion_obj_max_size)
                width = self.rng.integers(20, max_size)
                height = self.rng.integers(20, max_size)
                
                person_to_occlude = self.rng.choice(self.persons)
                while person_to_occlude in occluded_persons and len(occluded_persons) < len(self.persons):
                    person_to_occlude = self.rng.choice(self.persons)
                
                occluded_persons.append(person_to_occlude)
                person_pos = person_to_occlude.position
                x = person_pos[0] - width // 2
                y = person_pos[1] - height // 2
                
                occlusion_obj = OcclusionObject(
                    int(x), int(y), width, height,
                    self.config.occlusion_object_color,
                    len(self.occlusion_objects)
                )
                self.occlusion_objects.append(occlusion_obj)
        
        for instrument in self.instruments:
            instrument.opacity = 1.0
            if instrument.holder is not None:
                if self.rng.random() < self.config.instrument_hidden_prob:
                    instrument.opacity = 0.0
        
        self.current_frame += 1
    
    def _maybe_swap_active_status(self):
        """Occasionally swap which persons are active."""
        doctors = [p for p in self.persons if p.person_type == PersonType.DOCTOR]
        assistants = [p for p in self.persons if p.person_type == PersonType.ASSISTANT]
        
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
        state = {
            'frame': self.current_frame,
            'persons': self.persons,
            'instruments': self.instruments,
            'patient_table': self.patient_table,
            'preparation_table': self.preparation_table,
            'scene_objects': self.scene_objects,
            'occlusion_objects': self.occlusion_objects,
            'patient_table_edge': self.patient_table_edge,
            'preparation_table_edge': self.preparation_table_edge,
            'use_grid_movement': self.use_grid
        }
        
        if self.use_grid:
            state['grid'] = self.grid
        
        return state