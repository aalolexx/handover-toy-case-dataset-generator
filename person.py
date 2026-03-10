"""
Person class supporting both grid-based and continuous movement.
Movement mode is determined by config.use_grid_movement.
"""
from typing import Tuple, Optional, List, TYPE_CHECKING
import math
import numpy as np

from enums import PersonType, AssistantState, DoctorState, ActionLabel

if TYPE_CHECKING:
    from instrument import Instrument
    from grid_manager import GridManager
    from pathfinding_utils import PathfindingManager
    from config import Config
    from scene_object import SceneObject, PatientTable, PreparationTable


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


class Person:
    """
    Represents a doctor or assistant.
    Supports both grid-based (discrete) and continuous movement.
    """
    
    def __init__(self, person_id: int, person_type: PersonType,
                 initial_pos: Tuple[float, float], config: 'Config',
                 rng: np.random.Generator):
        self.id = person_id
        self.person_type = person_type
        self.config = config
        self.rng = rng
        self.use_grid = config.use_grid_movement
        
        # State management
        if person_type == PersonType.DOCTOR:
            self.state = DoctorState.IDLE
            self.color = config.doctor_color
        else:
            self.state = AssistantState.IDLE
            self.color = config.assistant_color
        
        self.is_active = False
        self.held_instrument: Optional['Instrument'] = None
        
        # Position storage - depends on mode
        if self.use_grid:
            self._grid_pos: Tuple[int, int] = (0, 0)
            self._home_pos: Tuple[int, int] = (0, 0)
            self.path: List[Tuple[int, int]] = []
            self.separation_steps_remaining: int = 0
            self.separation_direction: Tuple[int, int] = (0, 0)
        else:
            self._continuous_pos: Tuple[float, float] = initial_pos
            self._original_pos: Tuple[float, float] = initial_pos
            self.target_position: Optional[Tuple[float, float]] = None
            self.waypoints: List[Tuple[float, float]] = []
            self.velocity: Tuple[float, float] = (0.0, 0.0)
            self.wander_timer: int = 0
            self.idle_movement_range: float = 15.0
            self.separation_frames: int = 0
            self.separation_direction: Tuple[float, float] = (0.0, 0.0)
        
        # State timing
        self.state_timer: int = 0
        self.state_duration: int = 0
        self.move_timeout: int = 0
        self.max_move_timeout: int = 200
        
        # Handover coordination
        self.handover_partner: Optional['Person'] = None
        self.is_fake_handover: bool = False  # True if current handover is between same-color actors
        
        # References (set by ProcessManager)
        self.grid: Optional['GridManager'] = None
        self.pathfinding: Optional['PathfindingManager'] = None
        self.patient_table: Optional['PatientTable'] = None
        self.preparation_table: Optional['PreparationTable'] = None
        self.obstacles: List['SceneObject'] = []
        self.all_persons: List['Person'] = []
    
    # -------------------------------------------------------------------------
    # Position Properties
    # -------------------------------------------------------------------------
    
    @property
    def grid_pos(self) -> Tuple[int, int]:
        """Grid position (row, col). Only valid in grid mode."""
        return self._grid_pos
    
    @grid_pos.setter
    def grid_pos(self, value: Tuple[int, int]):
        self._grid_pos = value
    
    @property
    def home_pos(self) -> Tuple[int, int]:
        """Home position for idle movement (grid mode)."""
        return self._home_pos
    
    @home_pos.setter
    def home_pos(self, value: Tuple[int, int]):
        self._home_pos = value
    
    @property
    def position(self) -> Tuple[float, float]:
        """Get pixel position for rendering."""
        if self.use_grid:
            if self.grid:
                return self.grid.grid_to_pixel(self._grid_pos)
            cell_size = self.config.cell_size
            return (self._grid_pos[1] * cell_size + cell_size / 2,
                    self._grid_pos[0] * cell_size + cell_size / 2)
        else:
            return self._continuous_pos
    
    @position.setter
    def position(self, value: Tuple[float, float]):
        """Set position (continuous mode only)."""
        if not self.use_grid:
            self._continuous_pos = value
    
    @property
    def original_position(self) -> Tuple[float, float]:
        """Original/home position (continuous mode)."""
        if self.use_grid:
            return self.position
        return self._original_pos
    
    @original_position.setter
    def original_position(self, value: Tuple[float, float]):
        if not self.use_grid:
            self._original_pos = value
    
    @property
    def radius(self) -> float:
        """Get visual radius for rendering."""
        if self.use_grid:
            return self.config.cell_size / 2
        return self.config.person_radius
    
    # -------------------------------------------------------------------------
    # Setup Methods
    # -------------------------------------------------------------------------
    
    def set_grid(self, grid: 'GridManager'):
        """Set reference to grid manager (grid mode)."""
        self.grid = grid
    
    def set_pathfinding(self, pathfinding: 'PathfindingManager'):
        """Set reference to pathfinding manager (continuous mode)."""
        self.pathfinding = pathfinding
    
    def set_tables(self, patient_table: 'PatientTable', 
                   preparation_table: 'PreparationTable',
                   obstacles: List['SceneObject']):
        """Set references to tables and obstacles."""
        self.patient_table = patient_table
        self.preparation_table = preparation_table
        self.obstacles = obstacles
    
    def set_all_persons(self, persons: List['Person']):
        """Set reference to all persons."""
        self.all_persons = persons
    
    def set_active(self, active: bool):
        """Set whether this person participates in handovers."""
        self.is_active = active
        if not active:
            if self.person_type == PersonType.DOCTOR:
                self.state = DoctorState.IDLE
            else:
                self.state = AssistantState.IDLE
            if self.use_grid:
                self.path = []
            else:
                self.waypoints = []
    
    # -------------------------------------------------------------------------
    # Main Update Loop
    # -------------------------------------------------------------------------
    
    def update(self, available_instruments: List['Instrument'],
               active_doctors: List['Person'],
               active_assistants: List['Person']):
        """Update person state and position each frame."""
        
        # Handle separation first
        if self._is_separating():
            self._do_separation()
            return
        
        if self.is_active:
            if self.person_type == PersonType.ASSISTANT:
                self._update_assistant_state(available_instruments, active_doctors)
            else:
                self._update_doctor_state(active_assistants)
        else:
            self._do_wander()
    
    def _is_separating(self) -> bool:
        """Check if currently in separation phase."""
        if self.use_grid:
            return self.separation_steps_remaining > 0
        else:
            return self.separation_frames > 0
    
    # -------------------------------------------------------------------------
    # Movement - Mode-Specific Implementations
    # -------------------------------------------------------------------------
    
    def _move_one_step(self) -> bool:
        """Move one step towards current target. Returns True if arrived or moved."""
        if self.use_grid:
            return self._grid_move_one_step()
        else:
            return self._continuous_move_to_target()
    
    def _is_near_prep_table(self) -> bool:
        """Check if near preparation table."""
        if self.use_grid:
            return self.grid.is_adjacent_to_prep_table(self._grid_pos)
        else:
            return self._distance_to_rect(self.preparation_table.rect) < self.radius + 15
    
    def _is_near_patient_table(self) -> bool:
        """Check if near patient table."""
        if self.use_grid:
            return self.grid.is_adjacent_to_patient_table(self._grid_pos)
        else:
            return self._distance_to_rect(self.patient_table.rect) < self.radius + 15
    
    def _is_adjacent_to(self, other: 'Person') -> bool:
        """Check if adjacent to another person."""
        if self.use_grid:
            return self.grid.are_adjacent(self._grid_pos, other._grid_pos)
        else:
            touch_dist = self.radius + other.radius + 5
            return distance(self.position, other.position) <= touch_dist
    
    def _move_towards_person(self, other: 'Person'):
        """Move towards another person."""
        if self.use_grid:
            self._grid_move_adjacent_to(other)
        else:
            self._continuous_move_towards(other.position)
    
    def _set_target_near_prep_table(self):
        """Set target position near prep table."""
        if self.use_grid:
            target = self.grid.find_cell_adjacent_to_table(
                self.grid.prep_table_cells, self._grid_pos,
                self.all_persons, self.id)
            if target:
                self._grid_set_path_to(target)
        else:
            self.target_position = self._find_pos_near_table(self.preparation_table)
            self.waypoints = []
    
    def _set_target_near_patient_table(self):
        """Set target position near patient table."""
        if self.use_grid:
            target = self.grid.find_cell_adjacent_to_table(
                self.grid.patient_table_cells, self._grid_pos,
                self.all_persons, self.id)
            if target:
                self._grid_set_path_to(target)
        else:
            self.target_position = self._find_pos_near_table(self.patient_table)
            self.waypoints = []
    
    # -------------------------------------------------------------------------
    # Grid Movement Implementation
    # -------------------------------------------------------------------------
    
    def _grid_move_one_step(self) -> bool:
        """Move one step along path (grid mode)."""
        if not self.path or not self.grid:
            return False
        
        next_pos = self.path[0]
        
        if self.grid.is_walkable(next_pos) and \
           not self.grid.is_occupied_by_person(next_pos, self.id, self.all_persons):
            self._grid_pos = next_pos
            self.path.pop(0)
            return True
        
        # Blocked - recalculate
        if len(self.path) > 1:
            new_path = self.grid.find_path(self._grid_pos, self.path[-1], 
                                            self.all_persons, self.id)
            if new_path and len(new_path) > 1:
                self.path = new_path[1:]
                return self._grid_move_one_step()
        
        return False
    
    def _grid_set_path_to(self, target: Tuple[int, int]):
        """Set path to target (grid mode)."""
        if not self.grid:
            return
        path = self.grid.find_path(self._grid_pos, target, self.all_persons, self.id)
        self.path = path[1:] if path and len(path) > 1 else []
    
    def _grid_move_adjacent_to(self, other: 'Person'):
        """Move to become adjacent to another person (grid mode)."""
        if self.grid.are_adjacent(self._grid_pos, other._grid_pos):
            self.path = []
            return
        
        row, col = other._grid_pos
        best_cell = None
        best_dist = float('inf')
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (row + dr, col + dc)
            if self.grid.is_walkable(neighbor):
                if not self.grid.is_occupied_by_person(neighbor, self.id, self.all_persons):
                    dist = self.grid.manhattan_distance(self._grid_pos, neighbor)
                    if dist < best_dist:
                        best_dist = dist
                        best_cell = neighbor
        
        if best_cell and self._grid_pos != best_cell:
            if not self.path or self.path[-1] != best_cell:
                self._grid_set_path_to(best_cell)
            self._grid_move_one_step()
    
    def _grid_do_idle_movement(self):
        """Small movements around home position (grid mode)."""
        if self.rng.random() < 0.05:
            adjacent = self.grid.get_adjacent_walkable_cells(
                self._grid_pos, self.all_persons, self.id)
            if adjacent:
                adjacent.sort(key=lambda p: self.grid.manhattan_distance(p, self._home_pos))
                idx = min(self.rng.integers(0, max(1, len(adjacent))), len(adjacent) - 1)
                self._grid_pos = adjacent[idx]
    
    # -------------------------------------------------------------------------
    # Continuous Movement Implementation
    # -------------------------------------------------------------------------
    
    def _continuous_move_to_target(self) -> bool:
        """Move towards target (continuous mode). Returns True if arrived."""
        if self.target_position is None:
            return True
        
        dist = distance(self.position, self.target_position)
        if dist < self.config.movement_speed * 2:
            self._continuous_pos = self.target_position
            self.waypoints = []
            return True
        
        # Get path if needed
        if not self.waypoints and self.pathfinding:
            path = self.pathfinding.get_path(self.position, self.target_position)
            if path and len(path) > 1:
                self.waypoints = path[1:]
        
        # Determine current waypoint
        current_target = self.target_position
        if self.waypoints:
            current_target = self.waypoints[0]
            if distance(self.position, current_target) < self.config.movement_speed * 1.5:
                self.waypoints.pop(0)
                current_target = self.waypoints[0] if self.waypoints else self.target_position
        
        # Move towards target
        self._continuous_move_towards(current_target)
        return False
    
    def _continuous_move_towards(self, target: Tuple[float, float]):
        """Move one step towards target position."""
        dx = target[0] - self.position[0]
        dy = target[1] - self.position[1]
        dist = math.sqrt(dx * dx + dy * dy)
        
        if dist < 0.001:
            return
        
        # Apply avoidance
        avoid_x, avoid_y = self._get_avoidance_force()
        combined_x = dx / dist + avoid_x * 0.3
        combined_y = dy / dist + avoid_y * 0.3
        
        combined_dist = math.sqrt(combined_x**2 + combined_y**2)
        if combined_dist > 0.001:
            combined_x /= combined_dist
            combined_y /= combined_dist
        
        # Apply movement
        new_x = self.position[0] + combined_x * self.config.movement_speed
        new_y = self.position[1] + combined_y * self.config.movement_speed
        new_pos = self._clamp_to_bounds((new_x, new_y))
        
        if not self._would_collide(new_pos):
            self._continuous_pos = new_pos
    
    def _find_pos_near_table(self, table) -> Tuple[float, float]:
        """Find a position near a table (continuous mode)."""
        valid_sides = self._get_valid_sides(table)
        side = self.rng.choice(valid_sides) if valid_sides else 'top'
        
        margin = self.radius + 10
        rect = table.rect
        
        if side == 'left':
            x = rect[0] - margin
            y = rect[1] + rect[3] / 2
        elif side == 'right':
            x = rect[0] + rect[2] + margin
            y = rect[1] + rect[3] / 2
        elif side == 'top':
            x = rect[0] + rect[2] / 2
            y = rect[1] - margin
        else:
            x = rect[0] + rect[2] / 2
            y = rect[1] + rect[3] + margin
        
        return self._clamp_to_bounds((x, y))
    
    def _get_valid_sides(self, table) -> List[str]:
        """Get accessible sides of a table."""
        margin = self.radius + 25
        valid = []
        
        if table.x > margin:
            valid.append('left')
        if table.x + table.width < self.config.img_size - margin:
            valid.append('right')
        if table.y > margin:
            valid.append('top')
        if table.y + table.height < self.config.img_size - margin:
            valid.append('bottom')
        
        return valid if valid else ['left', 'right', 'top', 'bottom']
    
    def _get_avoidance_force(self) -> Tuple[float, float]:
        """Calculate steering force to avoid other persons."""
        force_x, force_y = 0.0, 0.0
        avoidance_radius = self.radius * 4
        
        for other in self.all_persons:
            if other.id == self.id:
                continue
            if self.handover_partner and self.handover_partner.id == other.id:
                continue
            
            dx = self.position[0] - other.position[0]
            dy = self.position[1] - other.position[1]
            dist = math.sqrt(dx * dx + dy * dy)
            
            if 0 < dist < avoidance_radius:
                factor = (avoidance_radius - dist) / avoidance_radius
                force_x += (dx / dist) * factor
                force_y += (dy / dist) * factor
        
        return (force_x, force_y)
    
    def _would_collide(self, pos: Tuple[float, float]) -> bool:
        """Check if position would cause collision."""
        all_obs = self.obstacles.copy()
        if self.patient_table:
            all_obs.append(self.patient_table)
        if self.preparation_table:
            all_obs.append(self.preparation_table)
        
        for obs in all_obs:
            if self._circle_rect_collision(pos, self.radius + 2, obs.rect):
                return True
        
        return False
    
    def _circle_rect_collision(self, center: Tuple[float, float], radius: float,
                                rect: Tuple[int, int, int, int]) -> bool:
        """Check circle-rectangle collision."""
        rx, ry, rw, rh = rect
        closest_x = max(rx, min(center[0], rx + rw))
        closest_y = max(ry, min(center[1], ry + rh))
        dist = math.sqrt((center[0] - closest_x)**2 + (center[1] - closest_y)**2)
        return dist < radius
    
    def _distance_to_rect(self, rect: Tuple[int, int, int, int]) -> float:
        """Calculate distance to nearest point on rectangle."""
        rx, ry, rw, rh = rect
        nearest_x = max(rx, min(self.position[0], rx + rw))
        nearest_y = max(ry, min(self.position[1], ry + rh))
        return distance(self.position, (nearest_x, nearest_y))
    
    def _continuous_do_idle_movement(self):
        """Small random movements (continuous mode)."""
        self.wander_timer -= 1
        if self.wander_timer <= 0:
            offset_x = self.rng.uniform(-self.idle_movement_range, self.idle_movement_range)
            offset_y = self.rng.uniform(-self.idle_movement_range, self.idle_movement_range)
            target = (self._original_pos[0] + offset_x, self._original_pos[1] + offset_y)
            self.target_position = self._clamp_to_bounds(target)
            self.wander_timer = self.rng.integers(20, 60)
        
        self._continuous_move_to_target()
    
    def _clamp_to_bounds(self, pos: Tuple[float, float]) -> Tuple[float, float]:
        """Clamp position to scene bounds."""
        margin = self.radius + 5
        return (
            max(margin, min(self.config.img_size - margin, pos[0])),
            max(margin, min(self.config.img_size - margin, pos[1]))
        )
    
    # -------------------------------------------------------------------------
    # Separation
    # -------------------------------------------------------------------------
    
    def _start_separation(self, from_partner: 'Person'):
        """Start moving away from partner after handover."""
        if self.use_grid:
            self.separation_steps_remaining = 2
            dr = self._grid_pos[0] - from_partner._grid_pos[0]
            dc = self._grid_pos[1] - from_partner._grid_pos[1]
            if dr != 0: dr = dr // abs(dr)
            if dc != 0: dc = dc // abs(dc)
            if dr == 0 and dc == 0:
                dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                dr, dc = dirs[self.rng.integers(4)]
            self.separation_direction = (dr, dc)
        else:
            self.separation_frames = 10
            dx = self.position[0] - from_partner.position[0]
            dy = self.position[1] - from_partner.position[1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 0.001:
                self.separation_direction = (dx/dist, dy/dist)
            else:
                angle = self.rng.uniform(0, 2 * math.pi)
                self.separation_direction = (math.cos(angle), math.sin(angle))
    
    def _do_separation(self):
        """Move away from partner."""
        if self.use_grid:
            self.separation_steps_remaining -= 1
            dr, dc = int(self.separation_direction[0]), int(self.separation_direction[1])
            new_pos = (self._grid_pos[0] + dr, self._grid_pos[1] + dc)
            
            if self.grid.is_walkable(new_pos) and \
               not self.grid.is_occupied_by_person(new_pos, self.id, self.all_persons):
                self._grid_pos = new_pos
            else:
                for alt_dr, alt_dc in [(dc, dr), (-dc, -dr), (-dr, -dc)]:
                    alt_pos = (self._grid_pos[0] + alt_dr, self._grid_pos[1] + alt_dc)
                    if self.grid.is_walkable(alt_pos) and \
                       not self.grid.is_occupied_by_person(alt_pos, self.id, self.all_persons):
                        self._grid_pos = alt_pos
                        break
        else:
            self.separation_frames -= 1
            dx = self.separation_direction[0] * self.config.movement_speed * 0.8
            dy = self.separation_direction[1] * self.config.movement_speed * 0.8
            new_pos = self._clamp_to_bounds((self.position[0] + dx, self.position[1] + dy))
            if not self._would_collide(new_pos):
                self._continuous_pos = new_pos
    
    # -------------------------------------------------------------------------
    # Wandering & Idle
    # -------------------------------------------------------------------------
    
    def _do_wander(self):
        """Random wandering for inactive persons."""
        if self.use_grid:
            if not self.path and self.rng.random() < 0.02:
                target = self.grid.get_random_walkable_cell(self.rng, self.all_persons, self.id)
                if target:
                    self._grid_set_path_to(target)
            self._grid_move_one_step()
        else:
            self.wander_timer -= 1
            if self.wander_timer <= 0 or self.target_position is None:
                margin = self.radius + 10
                for _ in range(20):
                    x = self.rng.uniform(margin, self.config.img_size - margin)
                    y = self.rng.uniform(margin, self.config.img_size - margin)
                    if not self._would_collide((x, y)):
                        self.target_position = (x, y)
                        break
                self.wander_timer = self.rng.integers(60, 180)
            self._continuous_move_to_target()
    
    def _do_idle_movement(self):
        """Small movements when idle."""
        if self.use_grid:
            self._grid_do_idle_movement()
        else:
            self._continuous_do_idle_movement()
    
    # -------------------------------------------------------------------------
    # Assistant State Machine
    # -------------------------------------------------------------------------
    
    def _update_assistant_state(self, available_instruments: List['Instrument'],
                                 active_doctors: List['Person']):
        """Update assistant state machine."""
        state = self.state
        
        if state == AssistantState.IDLE:
            self._do_idle_movement()
            self.state_timer += 1
            
            if self.state_timer > self.rng.integers(5, 15):
                self.state_timer = 0
                
                if self.held_instrument is None and available_instruments:
                    self._set_target_near_prep_table()
                    self.state = AssistantState.MOVING_TO_PREP_TABLE
                    self.move_timeout = 0
                elif self.held_instrument:
                    # Go through HOLDING state (where fake handover logic lives)
                    self.state = AssistantState.HOLDING
                    self.state_timer = 0
        
        elif state == AssistantState.MOVING_TO_PREP_TABLE:
            self.move_timeout += 1
            
            if self._is_near_prep_table():
                self.state = AssistantState.PREPARING
                self.state_timer = 0
                self.state_duration = self.rng.integers(
                    max(1, self.config.prepare_duration_avg - 10),
                    self.config.prepare_duration_avg + 10)
                if self.use_grid:
                    self.path = []
                else:
                    self.waypoints = []
            elif self.move_timeout > self.max_move_timeout:
                self._reset_to_idle()
            else:
                self._move_one_step()
        
        elif state == AssistantState.PREPARING:
            self.state_timer += 1
            
            if self.state_timer >= self.state_duration:
                if self.held_instrument is None:
                    instrument = self._pick_up_instrument(available_instruments)
                    if instrument:
                        # Always go to HOLDING first, then decide next action
                        self.state = AssistantState.HOLDING
                        self.state_timer = 0
                    else:
                        self._reset_to_idle()
                else:
                    self._lay_down_instrument()
                    self._reset_to_idle()
        
        elif state == AssistantState.HOLDING:
            self._do_idle_movement()
            self.state_timer += 1
            
            # Decide: try fake handover OR real handover
            if self.config.enable_fake_handovers and self.held_instrument:
                if self.rng.random() < self.config.fake_handover_prob:
                    # Try fake handover (green-to-green)
                    partner = self._find_same_color_partner_for_fake_handover()
                    if partner:
                        if self._is_adjacent_to(partner):
                            self._start_fake_handover(partner)
                            return
                        else:
                            # Move toward fake partner (reuse MOVING_TO_DOCTOR state)
                            self.handover_partner = partner
                            self.is_fake_handover = True
                            self.state = AssistantState.MOVING_TO_DOCTOR
                            self.state_timer = 0
                            return
            
            # Try real handover (green-to-blue)
            doctor = self._find_available_doctor(active_doctors)
            if doctor:
                self.handover_partner = doctor
                self.is_fake_handover = False
                self.state = AssistantState.MOVING_TO_DOCTOR
                self.state_timer = 0
                return
            
            # If stuck too long, put instrument back
            if self.state_timer > 150:
                self._set_target_near_prep_table()
                self.state = AssistantState.MOVING_TO_PREP_TABLE
                self.state_timer = 0
        
        elif state == AssistantState.MOVING_TO_DOCTOR:
            self.move_timeout += 1
            
            if self.handover_partner:
                if self._is_adjacent_to(self.handover_partner):
                    if self.is_fake_handover:
                        # Fake handover with same-color partner
                        if self._can_start_fake_handover():
                            self._start_fake_handover(self.handover_partner)
                        else:
                            self.state = AssistantState.HOLDING
                            self.handover_partner = None
                            self.is_fake_handover = False
                            self.state_timer = 0
                    else:
                        # Real handover with doctor
                        if self._can_start_handover():
                            self._start_giving_handover()
                        else:
                            self.state = AssistantState.HOLDING
                            self.handover_partner = None
                            self.state_timer = 0
                else:
                    self._move_towards_person(self.handover_partner)
            
            if self.move_timeout > self.max_move_timeout:
                self.state = AssistantState.HOLDING
                self.handover_partner = None
                self.is_fake_handover = False
                self.state_timer = 0
        
        elif state == AssistantState.GIVING:
            self.state_timer += 1
            
            # Transfer instrument after first frame (timer == 1)
            if self.state_timer == 1 and self.held_instrument and self.handover_partner:
                instrument = self.held_instrument
                self.held_instrument = None
                instrument.complete_handover(self.handover_partner)
                self.handover_partner.held_instrument = instrument
            
            # Complete handover after second frame (timer >= 2)
            if self.state_timer >= self.state_duration:
                self._complete_give_to_doctor()
        
        elif state == AssistantState.WAITING_BY_DOCTOR:
            if self.handover_partner:
                doctor = self.handover_partner
                
                if doctor.state == DoctorState.GIVING:
                    self.state = AssistantState.RECEIVING
                    self.state_timer = 0
                    self.state_duration = self.config.handover_duration
                elif doctor.held_instrument is None and doctor.state == DoctorState.IDLE:
                    self._reset_to_idle()
                elif not self._is_adjacent_to(doctor):
                    self._move_towards_person(doctor)
            else:
                self._reset_to_idle()
        
        elif state == AssistantState.RECEIVING:
            self.state_timer += 1
            
            # Transfer instrument after first frame (timer == 1)
            if self.state_timer == 1 and self.handover_partner and self.handover_partner.held_instrument:
                instrument = self.handover_partner.held_instrument
                self.handover_partner.held_instrument = None
                instrument.complete_handover(self)
                self.held_instrument = instrument
            
            # Complete handover after second frame (timer >= 2)
            if self.state_timer >= self.state_duration:
                self._complete_receive_from_doctor()
        
        elif state == AssistantState.MOVING_FROM_DOCTOR:
            self.move_timeout += 1
            
            if self._is_near_prep_table():
                self.state = AssistantState.PREPARING
                self.state_timer = 0
                self.state_duration = self.rng.integers(5, 15)
            elif self.move_timeout > self.max_move_timeout:
                self.state = AssistantState.PREPARING
                self.state_timer = 0
                self.state_duration = self.rng.integers(5, 15)
            else:
                self._move_one_step()
    
    # -------------------------------------------------------------------------
    # Doctor State Machine
    # -------------------------------------------------------------------------
    
    def _update_doctor_state(self, active_assistants: List['Person']):
        """Update doctor state machine."""
        state = self.state
        
        if state == DoctorState.IDLE:
            approaching = self._find_approaching_assistant(active_assistants)
            if approaching:
                if not self._is_adjacent_to(approaching):
                    self._move_towards_person(approaching)
            else:
                self._do_idle_movement()
                if not self._is_near_patient_table():
                    self._set_target_near_patient_table()
                    self._move_one_step()
        
        elif state == DoctorState.RECEIVING:
            pass
        
        elif state == DoctorState.HOLDING:
            self._do_idle_movement()
            self.state_timer += 1
            
            # Check for fake handover opportunity (blue-to-blue)
            if (self.config.enable_fake_handovers and 
                self.held_instrument and
                self.rng.random() < self.config.fake_handover_prob):
                partner = self._find_same_color_partner_for_fake_handover()
                if partner and self._is_adjacent_to(partner):
                    self._start_fake_handover(partner)
                    return
            
            if self.state_timer > self.rng.integers(5, 20):
                self.state = DoctorState.WORKING
                self.state_timer = 0
                self.state_duration = max(15, int(self.rng.normal(
                    self.config.work_duration_avg, 10)))
                if self.held_instrument:
                    self.held_instrument.start_use()
        
        elif state == DoctorState.WORKING:
            self._do_idle_movement()
            self.state_timer += 1
            
            if self.state_timer >= self.state_duration:
                if self.held_instrument:
                    self.held_instrument.stop_use()
                
                assistant = self._find_assistant_for_return(active_assistants)
                if assistant:
                    self._start_giving_to_assistant(assistant)
                else:
                    self.state = DoctorState.HOLDING
                    self.state_timer = 100
        
        elif state == DoctorState.GIVING:
            self.state_timer += 1
            
            # For fake handovers (blue-to-blue), handle transfer here
            if self.is_fake_handover:
                # Transfer instrument after first frame (timer == 1)
                if self.state_timer == 1 and self.held_instrument and self.handover_partner:
                    instrument = self.held_instrument
                    self.held_instrument = None
                    instrument.complete_handover(self.handover_partner)
                    self.handover_partner.held_instrument = instrument
                
                # Complete fake handover after second frame
                if self.state_timer >= self.state_duration:
                    if self.handover_partner:
                        partner = self.handover_partner
                        
                        # Giver goes to IDLE, receiver goes to HOLDING
                        self.state = DoctorState.IDLE
                        partner.state = DoctorState.HOLDING
                        partner.state_timer = 0
                        
                        # Clear handover state
                        self.handover_partner = None
                        partner.handover_partner = None
                        self.is_fake_handover = False
                        partner.is_fake_handover = False
                        
                        # Separation
                        self._start_separation(partner)
                        partner._start_separation(self)
    
    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------
    
    def _find_available_doctor(self, active_doctors: List['Person']) -> Optional['Person']:
        """Find a doctor that can receive an instrument."""
        available = []
        
        for doctor in active_doctors:
            if doctor.held_instrument is not None:
                continue
            if doctor.state not in [DoctorState.IDLE, DoctorState.HOLDING]:
                continue
            if doctor._is_separating():
                continue
            
            already_targeted = False
            for person in self.all_persons:
                if person.id == self.id:
                    continue
                if (person.person_type == PersonType.ASSISTANT and
                    person.state == AssistantState.MOVING_TO_DOCTOR and
                    person.handover_partner and
                    person.handover_partner.id == doctor.id):
                    already_targeted = True
                    break
            
            if not already_targeted:
                available.append(doctor)
        
        if not available:
            return None
        
        if self.use_grid:
            return min(available, 
                       key=lambda d: self.grid.manhattan_distance(self._grid_pos, d._grid_pos))
        else:
            return min(available, key=lambda d: distance(self.position, d.position))
    
    def _find_approaching_assistant(self, active_assistants: List['Person']) -> Optional['Person']:
        """Find assistant approaching this doctor."""
        for assistant in active_assistants:
            if (assistant.state == AssistantState.MOVING_TO_DOCTOR and
                assistant.handover_partner and
                assistant.handover_partner.id == self.id):
                return assistant
        return None
    
    def _find_assistant_for_return(self, active_assistants: List['Person']) -> Optional['Person']:
        """Find assistant to return instrument to."""
        for assistant in active_assistants:
            if (assistant.state == AssistantState.WAITING_BY_DOCTOR and
                assistant.handover_partner and
                assistant.handover_partner.id == self.id and
                assistant.held_instrument is None and
                not assistant._is_separating()):
                return assistant
        
        idle = [a for a in active_assistants
                if a.state == AssistantState.IDLE and
                a.held_instrument is None and
                not a._is_separating()]
        
        if idle:
            if self.use_grid:
                return min(idle, 
                           key=lambda a: self.grid.manhattan_distance(self._grid_pos, a._grid_pos))
            else:
                return min(idle, key=lambda a: distance(self.position, a.position))
        
        return None
    
    def _find_same_color_partner_for_fake_handover(self) -> Optional['Person']:
        """Find a same-color actor available for fake handover."""
        for person in self.all_persons:
            if person.id == self.id:
                continue
            if person.person_type != self.person_type:
                continue
            if person.held_instrument is not None:
                continue
            if person._is_separating():
                continue
            
            # Check if in a compatible state
            if self.person_type == PersonType.ASSISTANT:
                if person.state not in [AssistantState.IDLE, AssistantState.HOLDING]:
                    continue
            else:
                if person.state not in [DoctorState.IDLE, DoctorState.HOLDING]:
                    continue
            
            # Check if already in a handover
            if person.handover_partner is not None:
                continue
            
            return person
        
        return None
    
    def _start_fake_handover(self, partner: 'Person'):
        """Start a fake handover with a same-color actor."""
        self.handover_partner = partner
        partner.handover_partner = self
        self.is_fake_handover = True
        partner.is_fake_handover = True
        
        # Set both to GIVING/RECEIVING states
        if self.person_type == PersonType.ASSISTANT:
            self.state = AssistantState.GIVING
            partner.state = AssistantState.RECEIVING
        else:
            self.state = DoctorState.GIVING
            partner.state = DoctorState.RECEIVING
        
        self.state_timer = 0
        partner.state_timer = 0
        self.state_duration = self.config.handover_duration
        partner.state_duration = self.config.handover_duration
    
    def _can_start_handover(self) -> bool:
        """Check if handover can start (real handover to doctor)."""
        if not self.handover_partner:
            return False
        if self.handover_partner.held_instrument is not None:
            return False
        if self.handover_partner.state not in [DoctorState.IDLE, DoctorState.HOLDING]:
            return False
        if self.handover_partner._is_separating():
            return False
        return True
    
    def _can_start_fake_handover(self) -> bool:
        """Check if fake handover can start (same-color partner)."""
        if not self.handover_partner:
            return False
        if self.handover_partner.held_instrument is not None:
            return False
        if self.handover_partner._is_separating():
            return False
        if self.handover_partner.handover_partner is not None:
            return False
        # Check partner is in compatible state for their type
        if self.handover_partner.person_type == PersonType.ASSISTANT:
            if self.handover_partner.state not in [AssistantState.IDLE, AssistantState.HOLDING]:
                return False
        else:
            if self.handover_partner.state not in [DoctorState.IDLE, DoctorState.HOLDING]:
                return False
        return True
    
    def _start_giving_handover(self):
        """Start giving instrument to doctor."""
        if not self.handover_partner:
            return
        
        self.state = AssistantState.GIVING
        self.handover_partner.state = DoctorState.RECEIVING
        self.handover_partner.handover_partner = self
        self.state_timer = 0
        self.handover_partner.state_timer = 0
        self.state_duration = self.config.handover_duration
        self.handover_partner.state_duration = self.config.handover_duration
        
        if self.use_grid:
            self.path = []
        else:
            self.waypoints = []
    
    def _complete_give_to_doctor(self):
        """Complete giving - handle state transitions (instrument already transferred)."""
        if self.handover_partner:
            partner = self.handover_partner
            
            if self.is_fake_handover:
                # Fake handover: giver goes to IDLE, receiver goes to HOLDING
                self.state = AssistantState.IDLE
                partner.state = AssistantState.HOLDING
                partner.state_timer = 0
                
                # Clear handover state
                self.handover_partner = None
                partner.handover_partner = None
                self.is_fake_handover = False
                partner.is_fake_handover = False
                
                # Separation
                self._start_separation(partner)
                partner._start_separation(self)
            else:
                # Normal handover: Assistant waits by doctor
                self.state = AssistantState.WAITING_BY_DOCTOR
                if self.use_grid:
                    self._home_pos = self._grid_pos
                else:
                    self._original_pos = self._continuous_pos
                
                # Doctor transitions to holding
                partner.state = DoctorState.HOLDING
                partner.state_timer = 0
                partner.handover_partner = None
                
                # Separation
                self._start_separation(partner)
                partner._start_separation(self)
    
    def _complete_receive_from_doctor(self):
        """Complete receiving - handle state transitions (instrument already transferred)."""
        if self.handover_partner:
            partner = self.handover_partner
            
            # Assistant moves back to prep table
            self.state = AssistantState.MOVING_FROM_DOCTOR
            self._set_target_near_prep_table()
            self.move_timeout = 0
            
            # Doctor goes idle
            partner.state = DoctorState.IDLE
            partner.handover_partner = None
            
            # Separation
            self._start_separation(partner)
            partner._start_separation(self)
            
            self.handover_partner = None
    
    def _start_giving_to_assistant(self, assistant: 'Person'):
        """Start returning instrument to assistant."""
        if not self._is_adjacent_to(assistant):
            assistant.handover_partner = self
            assistant.state = AssistantState.MOVING_TO_DOCTOR
            self.state = DoctorState.HOLDING
            self.state_timer = 0
            return
        
        self.handover_partner = assistant
        assistant.handover_partner = self
        
        self.state = DoctorState.GIVING
        assistant.state = AssistantState.RECEIVING
        
        self.state_timer = 0
        assistant.state_timer = 0
        self.state_duration = self.config.handover_duration
        assistant.state_duration = self.config.handover_duration
    
    def _reset_to_idle(self):
        """Reset to idle state."""
        self.state = AssistantState.IDLE
        self.state_timer = 0
        self.handover_partner = None
        self.move_timeout = 0
        if self.use_grid:
            self.path = []
        else:
            self.waypoints = []
    
    # -------------------------------------------------------------------------
    # Instrument Handling
    # -------------------------------------------------------------------------
    
    def _pick_up_instrument(self, available: List['Instrument']) -> Optional['Instrument']:
        """Pick up an instrument."""
        for instrument in available:
            if instrument.state.name == 'ON_TABLE' and instrument.holder is None:
                instrument.attach_to(self)
                self.held_instrument = instrument
                return instrument
        return None
    
    def _lay_down_instrument(self):
        """Put down held instrument."""
        if self.held_instrument:
            self.held_instrument.detach()
            self.held_instrument = None
    
    # -------------------------------------------------------------------------
    # Public Interface
    # -------------------------------------------------------------------------
    
    def get_instrument_attach_point(self) -> Tuple[float, float]:
        """Get pixel position for instrument attachment."""
        return self.position
    
    def get_current_action(self) -> Optional[ActionLabel]:
        """Get current action label for annotation."""
        if self.person_type == PersonType.ASSISTANT:
            if self.state == AssistantState.PREPARING:
                return ActionLabel.ASSISTANT_PREPARES
            elif self.state == AssistantState.GIVING:
                return ActionLabel.ASSISTANT_GIVES
            elif self.state == AssistantState.RECEIVING:
                return ActionLabel.ASSISTANT_RECEIVES
        else:
            if self.state == DoctorState.WORKING:
                return ActionLabel.DOCTOR_WORKS
            elif self.state in [DoctorState.GIVING, DoctorState.RECEIVING]:
                return None
        
        if self.held_instrument is not None:
            return ActionLabel.PERSON_HOLDS
        
        return None
    
    def is_in_handover(self) -> bool:
        """Check if currently in handover."""
        if self.person_type == PersonType.ASSISTANT:
            return self.state in [AssistantState.GIVING, AssistantState.RECEIVING]
        else:
            return self.state in [DoctorState.GIVING, DoctorState.RECEIVING]
    
    def get_bounding_box(self, img_size: int) -> Tuple[float, float, float, float]:
        """Get normalized bounding box."""
        x, y = self.position
        r = self.radius
        return (x / img_size, y / img_size, (2 * r) / img_size, (2 * r) / img_size)