"""
Person class representing doctors and assistants with movement and state management.
"""
from typing import Tuple, Optional, List, TYPE_CHECKING
import math
import numpy as np

from enums import PersonType, AssistantState, DoctorState, ActionLabel, InstrumentState
from utils import (distance, move_towards, circle_rect_collision, 
                   angle_between_points, clamp, find_position_near_rect,
                   get_bounding_box, normalize_bbox)

if TYPE_CHECKING:
    from instrument import Instrument
    from scene_object import SceneObject, PatientTable, PreparationTable
    from config import Config
    from pathfinding_utils import PathfindingManager


class Person:
    """Represents a doctor or assistant in the surgery scene."""
    
    def __init__(self, person_id: int, person_type: PersonType,
                 position: Tuple[float, float], config: 'Config',
                 rng: np.random.Generator):
        """
        Initialize a person.
        
        Args:
            person_id: Unique identifier
            person_type: DOCTOR or ASSISTANT
            position: Initial (x, y) position
            config: Scene configuration
            rng: Random number generator
        """
        self.id = person_id
        self.person_type = person_type
        self.position = position
        self.config = config
        self.rng = rng
        
        self.radius = config.person_radius
        self.speed = config.movement_speed
        self.is_active = False
        
        # State management
        if person_type == PersonType.DOCTOR:
            self.state = DoctorState.IDLE
            self.color = config.doctor_color
        else:
            self.state = AssistantState.IDLE
            self.color = config.assistant_color
        
        # Instrument handling
        self.held_instrument: Optional['Instrument'] = None
        self.instrument_attach_angle: float = 0.0  # Angle where instrument is attached
        
        # Movement
        self.target_position: Optional[Tuple[float, float]] = None
        self.waypoints: List[Tuple[float, float]] = []  # Waypoints for pathfinding
        self.wander_timer: int = 0
        self.idle_movement_range: float = 15.0
        self.original_position: Tuple[float, float] = position
        
        # Pathfinding manager (set by ProcessManager)
        self.pathfinding: Optional['PathfindingManager'] = None
        
        # State timing
        self.state_timer: int = 0
        self.state_duration: int = 0
        self.transition_pause: int = 0
        self.move_timeout: int = 0  # Timeout for reaching target
        self.max_move_timeout: int = 200  # Max frames to reach target
        
        # Handover coordination
        self.handover_partner: Optional['Person'] = None
        self.handover_frame_count: int = 0
        
        # Reference to tables (set by ProcessManager)
        self.patient_table: Optional['PatientTable'] = None
        self.preparation_table: Optional['PreparationTable'] = None
        self.obstacles: List['SceneObject'] = []
    
    def set_tables(self, patient_table: 'PatientTable', 
                   preparation_table: 'PreparationTable',
                   obstacles: List['SceneObject'],
                   pathfinding: Optional['PathfindingManager'] = None):
        """Set references to scene tables, obstacles, and pathfinding."""
        self.patient_table = patient_table
        self.preparation_table = preparation_table
        self.obstacles = obstacles
        self.pathfinding = pathfinding
    
    def set_active(self, active: bool):
        """Set whether this person participates in handovers."""
        self.is_active = active
        if not active:
            if self.person_type == PersonType.DOCTOR:
                self.state = DoctorState.IDLE
            else:
                self.state = AssistantState.IDLE
    
    def update(self, available_instruments: List['Instrument'],
               active_doctors: List['Person'],
               active_assistants: List['Person']):
        """Update person state and position each frame."""
        if self.transition_pause > 0:
            self.transition_pause -= 1
            self._do_idle_movement()
            return
        
        if self.is_active:
            self._update_active_behavior(available_instruments, 
                                         active_doctors, active_assistants)
        else:
            self._update_passive_behavior()
        
        # Update instrument attachment angle to face movement direction
        if self.held_instrument and self.target_position:
            self.instrument_attach_angle = angle_between_points(
                self.position, self.target_position)
    
    def _update_active_behavior(self, available_instruments: List['Instrument'],
                                 active_doctors: List['Person'],
                                 active_assistants: List['Person']):
        """Update behavior for active persons."""
        if self.person_type == PersonType.ASSISTANT:
            self._update_assistant_state(available_instruments, active_doctors)
        else:
            self._update_doctor_state(active_assistants)
    
    def _update_assistant_state(self, available_instruments: List['Instrument'],
                                 active_doctors: List['Person']):
        """Update assistant state machine."""
        state = self.state
        
        if state == AssistantState.IDLE:
            self._do_idle_movement()
            # Maybe transition to preparing
            if self.rng.random() < 0.1 and available_instruments:
                self.state = AssistantState.MOVING_TO_PREP_TABLE
                self._set_target_near_prep_table()
        
        elif state == AssistantState.MOVING_TO_PREP_TABLE:
            if self._move_to_target():
                self.state = AssistantState.PREPARING
                self.state_timer = 0
                self.state_duration = int(self.rng.normal(
                    self.config.prepare_duration_avg, 5))
        
        elif state == AssistantState.PREPARING:
            self._do_idle_movement()
            self.state_timer += 1
            if self.state_timer >= self.state_duration:
                # Pick up an instrument if not having one
                if self.held_instrument is None:
                    instrument = self._pick_up_instrument(available_instruments)
                    if instrument:
                        self.state = AssistantState.HOLDING
                        self.transition_pause = self.rng.integers(
                            0, self.config.max_transition_pause // 4)
                    else:
                        self.state = AssistantState.IDLE
                else: 
                    self._lay_down_instrument()
                    self.state = AssistantState.IDLE
                    self.transition_pause = 0
        
        elif state == AssistantState.HOLDING:
            self._do_idle_movement()
            # Maybe move to doctor
            if self.rng.random() < 0.08 and active_doctors:
                doctor = self.rng.choice(active_doctors)
                if doctor.state in [DoctorState.IDLE, DoctorState.HOLDING, DoctorState.WORKING]:
                    self.handover_partner = doctor
                    self.state = AssistantState.MOVING_TO_DOCTOR
                    self._set_target_near_person(doctor)
            elif self.rng.random() < 0.08:
                self.state = AssistantState.MOVING_TO_PREP_TABLE
                self._set_target_near_prep_table()
        
        elif state == AssistantState.MOVING_TO_DOCTOR:
            self.move_timeout += 1
            reached = self._move_to_target()
            
            # Check if close enough to doctor (within 2x radius distance)
            if self.handover_partner:
                dist_to_partner = distance(self.position, self.handover_partner.position)
                if dist_to_partner <= (self.radius + self.handover_partner.radius + 4):
                    reached = True
            
            if reached:
                if self.handover_partner and self.handover_partner.held_instrument is None:
                    self.state = AssistantState.GIVING
                    self.handover_partner.state = DoctorState.RECEIVING
                    self.handover_partner.handover_partner = self
                    self.handover_frame_count = 0
                    self.state_duration = self.config.handover_duration
                    self.move_timeout = 0
            elif self.move_timeout > self.max_move_timeout:
                # Timeout - give up and return to idle
                self.state = AssistantState.HOLDING
                self.handover_partner = None
                self.move_timeout = 0
        
        elif state == AssistantState.GIVING:
            self.handover_frame_count += 1
            if self.handover_frame_count >= self.state_duration:
                self._complete_handover_give()
        
        elif state == AssistantState.WAITING_BY_DOCTOR:
            # Stay close to the doctor we gave the instrument to
            if self.handover_partner:
                # Do small movements near the doctor
                dist_to_doctor = distance(self.position, self.handover_partner.position)
                if dist_to_doctor > self.radius * 3:
                    # Move closer if we drifted too far
                    self._set_target_near_person(self.handover_partner)
                    self._move_to_target()
                else:
                    self._do_idle_movement()
                
                # Check if doctor wants to give instrument back
                # (Doctor will initiate this by setting our state to RECEIVING)
                
                # If doctor no longer has instrument or moved on, go back to idle
                if (self.handover_partner.held_instrument is None and 
                    self.handover_partner.state not in [DoctorState.GIVING, DoctorState.WORKING, DoctorState.HOLDING]):
                    self.state = AssistantState.IDLE
                    self.handover_partner = None
            else:
                # Lost track of doctor, go back to idle
                self.state = AssistantState.IDLE
        
        elif state == AssistantState.RECEIVING:
            self.handover_frame_count += 1
            if self.handover_frame_count >= self.state_duration:
                self._complete_handover_receive()
        
        elif state == AssistantState.MOVING_FROM_DOCTOR:
            if self._move_to_target():
                self.state = AssistantState.IDLE
                self.transition_pause = self.rng.integers(
                    0, self.config.max_transition_pause)
    
    def _update_doctor_state(self, active_assistants: List['Person']):
        """Update doctor state machine."""
        state = self.state
        
        if state == DoctorState.IDLE:
            self._do_idle_movement()
            # Stay near patient table
            if distance(self.position, self.patient_table.center) > 60:
                self._set_target_near_patient_table()
        
        elif state == DoctorState.HOLDING:
            self._do_idle_movement()
            # Maybe start working
            if self.rng.random() < 0.1:
                self.state = DoctorState.WORKING
                self.state_timer = 0
                self.state_duration = int(self.rng.normal(
                    self.config.work_duration_avg, 5))
                if self.held_instrument:
                    self.held_instrument.start_use()
        
        elif state == DoctorState.WORKING:
            self.state_timer += 1
            if self.state_timer >= self.state_duration:
                if self.held_instrument:
                    self.held_instrument.stop_use()
                
                # Try to give instrument back
                gave_instrument = False
                if self.rng.random() < 0.8:  # 80% chance to give back
                    # First priority: assistant waiting nearby (the one who gave us the instrument)
                    waiting_assistants = [a for a in active_assistants 
                                         if a.state == AssistantState.WAITING_BY_DOCTOR and 
                                         a.handover_partner == self and
                                         a.held_instrument is None]
                    
                    # Second priority: any idle assistant without instrument
                    if not waiting_assistants:
                        waiting_assistants = [a for a in active_assistants 
                                             if a.state == AssistantState.IDLE and 
                                             a.held_instrument is None]
                    
                    if waiting_assistants:
                        # Prefer the closest assistant
                        assistant = min(waiting_assistants, 
                                       key=lambda a: distance(self.position, a.position))
                        self.handover_partner = assistant
                        self.state = DoctorState.GIVING
                        assistant.state = AssistantState.RECEIVING
                        assistant.handover_partner = self
                        # Initialize both participants' handover counters
                        self.handover_frame_count = 0
                        assistant.handover_frame_count = 0
                        self.state_duration = self.config.handover_duration
                        assistant.state_duration = self.config.handover_duration
                        gave_instrument = True
                
                if not gave_instrument:
                    self.state = DoctorState.HOLDING
                    self.transition_pause = self.rng.integers(
                        10, self.config.max_transition_pause // 2)
        
        elif state == DoctorState.GIVING:
            self.handover_frame_count += 1
            if self.handover_frame_count >= self.state_duration:
                self._complete_handover_give()
        
        elif state == DoctorState.RECEIVING:
            # Handled by assistant's GIVING state completion
            pass
    
    def _update_passive_behavior(self):
        """Update behavior for inactive persons (random movement)."""
        self._do_wander_movement()
    
    def _do_idle_movement(self):
        """Small random movements while idle."""
        self.wander_timer -= 1
        if self.wander_timer <= 0:
            # Set new small random target
            offset_x = self.rng.uniform(-self.idle_movement_range, 
                                        self.idle_movement_range)
            offset_y = self.rng.uniform(-self.idle_movement_range, 
                                        self.idle_movement_range)
            target = (self.original_position[0] + offset_x,
                      self.original_position[1] + offset_y)
            target = self._clamp_to_bounds(target)
            self.target_position = target
            self.wander_timer = self.rng.integers(20, 60)
        
        self._move_to_target()
    
    def _do_wander_movement(self):
        """Random wandering movement for inactive persons."""
        self.wander_timer -= 1
        if self.wander_timer <= 0 or self.target_position is None:
            # Set new random target
            margin = self.radius + 10
            target_x = self.rng.uniform(margin, self.config.img_size - margin)
            target_y = self.rng.uniform(margin, self.config.img_size - margin)
            target = (target_x, target_y)
            
            # Avoid tables
            attempts = 0
            while self._would_collide_with_obstacles(target) and attempts < 20:
                target_x = self.rng.uniform(margin, self.config.img_size - margin)
                target_y = self.rng.uniform(margin, self.config.img_size - margin)
                target = (target_x, target_y)
                attempts += 1
            
            self.target_position = target
            self.waypoints = []  # Clear waypoints for wandering
            self.wander_timer = self.rng.integers(60, 180)
        
        self._move_to_target()
    
    def _move_to_target(self) -> bool:
        """Move towards target position using A* pathfinding.
        
        Returns:
            True if target reached, False otherwise
        """
        if self.target_position is None:
            return True
        
        # Check if we've reached the final target
        dist_to_target = distance(self.position, self.target_position)
        if dist_to_target < self.speed * 2:
            self.position = self.target_position
            self.waypoints = []
            return True
        
        # If no waypoints, calculate path using A*
        if not self.waypoints and self.pathfinding:
            path = self.pathfinding.get_path(self.position, self.target_position)
            if path and len(path) > 1:
                # Skip the first point (current position) and use rest as waypoints
                self.waypoints = path[1:]
            elif path and len(path) == 1:
                # Path only contains destination - we're close enough for direct movement
                pass
            else:
                # No path found - might be in blocked zone
                # Try to find escape path to nearest walkable position
                escape_pos = self._find_escape_position()
                if escape_pos:
                    self.waypoints = [escape_pos]
        
        # If we have waypoints, move towards the current waypoint
        if self.waypoints:
            current_waypoint = self.waypoints[0]
            dist_to_waypoint = distance(self.position, current_waypoint)
            
            if dist_to_waypoint < self.speed * 2:
                # Reached waypoint, move to next
                self.waypoints.pop(0)
                if self.waypoints:
                    current_waypoint = self.waypoints[0]
                else:
                    # No more waypoints, move directly to target
                    current_waypoint = self.target_position
            
            # Move towards current waypoint
            next_pos = move_towards(self.position, current_waypoint, self.speed)
            if not self._would_collide_with_obstacles(next_pos):
                self.position = next_pos
                # Update instrument attach angle based on movement direction
                self.instrument_attach_angle = angle_between_points(self.position, current_waypoint)
                return False
            else:
                # Path is blocked, recalculate
                self.waypoints = []
        
        # Try direct movement (fallback if no pathfinding or path blocked)
        next_pos = move_towards(self.position, self.target_position, self.speed)
        if not self._would_collide_with_obstacles(next_pos):
            self.position = next_pos
            self.instrument_attach_angle = angle_between_points(self.position, self.target_position)
            return False
        
        # If stuck, try to move away from obstacles
        self._try_unstuck_movement()
        
        return False
    
    def _try_unstuck_movement(self) -> bool:
        """Try to get unstuck by moving in any valid direction."""
        # Try 8 cardinal/diagonal directions, preferring ones closer to target
        best_pos = None
        best_score = float('inf')
        
        for angle in [0, math.pi/4, math.pi/2, 3*math.pi/4, 
                      math.pi, 5*math.pi/4, 3*math.pi/2, 7*math.pi/4]:
            test_x = self.position[0] + self.speed * math.cos(angle)
            test_y = self.position[1] + self.speed * math.sin(angle)
            test_pos = self._clamp_to_bounds((test_x, test_y))
            
            if not self._would_collide_with_obstacles(test_pos):
                if self.target_position:
                    score = distance(test_pos, self.target_position)
                    if score < best_score:
                        best_score = score
                        best_pos = test_pos
                else:
                    self.position = test_pos
                    return True
        
        if best_pos:
            self.position = best_pos
            return True
        
        # Last resort: random movement
        for _ in range(8):
            angle = self.rng.uniform(0, 2 * math.pi)
            test_x = self.position[0] + self.speed * math.cos(angle)
            test_y = self.position[1] + self.speed * math.sin(angle)
            test_pos = self._clamp_to_bounds((test_x, test_y))
            if not self._would_collide_with_obstacles(test_pos):
                self.position = test_pos
                return True
        
        return False
    
    def _would_collide_with_obstacles(self, position: Tuple[float, float]) -> bool:
        """Check if position would cause collision with obstacles or block pathfinding."""
        # Check actual collision with obstacles
        all_obstacles = self.obstacles.copy()
        if self.patient_table:
            all_obstacles.append(self.patient_table)
        if self.preparation_table:
            all_obstacles.append(self.preparation_table)
        
        for obs in all_obstacles:
            if circle_rect_collision(position, self.radius + 3, obs.rect):
                return True
        
        # Also check if this would put us in a pathfinding blocked zone
        if self.pathfinding and not self.pathfinding.is_walkable(position):
            return True
        
        return False
    
    def _find_escape_position(self) -> Optional[Tuple[float, float]]:
        """Find a nearby position that is walkable in the pathfinding grid.
        
        Used when the person is stuck in a blocked zone of the pathfinding grid
        but not colliding with actual obstacles.
        """
        # Try positions in expanding circles
        for radius in range(1, 15):
            dist = radius * self.speed * 2
            for angle_step in range(8 * radius):
                angle = (angle_step / (8 * radius)) * 2 * math.pi
                test_x = self.position[0] + dist * math.cos(angle)
                test_y = self.position[1] + dist * math.sin(angle)
                test_pos = self._clamp_to_bounds((test_x, test_y))
                
                # Check if position is valid for collision AND pathfinding
                if not self._would_collide_with_obstacles(test_pos):
                    if self.pathfinding and self.pathfinding.is_walkable(test_pos):
                        return test_pos
        
        return None
    
    def _clamp_to_bounds(self, position: Tuple[float, float]) -> Tuple[float, float]:
        """Clamp position to scene bounds."""
        x = clamp(position[0], self.radius + 5, 
                  self.config.img_size - self.radius - 5)
        y = clamp(position[1], self.radius + 5, 
                  self.config.img_size - self.radius - 5)
        return (x, y)
    
    def _set_target_near_prep_table(self):
        """Set target to a position near the preparation table."""
        if self.preparation_table:
            # Position to the side of the prep table, not directly above it
            # This avoids collision with patient table
            side = self.rng.choice(['left', 'right'])
            self.target_position = find_position_near_rect(
                self.preparation_table.rect, side, self.radius + 10,
                (0, 0, self.config.img_size, self.config.img_size),
                self.obstacles, self.radius)
            self.original_position = self.target_position
            self.waypoints = []  # Clear any existing waypoints
    
    def _set_target_near_patient_table(self):
        """Set target to a position near the patient table."""
        if self.patient_table:
            # Doctors work from the top of the table (more accessible)
            side = self.rng.choice(['left', 'right'])
            self.target_position = find_position_near_rect(
                self.patient_table.rect, side, self.radius + 15,
                (0, 0, self.config.img_size, self.config.img_size),
                self.obstacles, self.radius)
            self.original_position = self.target_position
            self.waypoints = []  # Clear any existing waypoints
    
    def _set_target_near_person(self, other: 'Person'):
        """Set target to move next to another person."""
        # Move to a position near the other person - preferably from our direction
        dx = other.position[0] - self.position[0]
        dy = other.position[1] - self.position[1]
        dist = max(1, math.sqrt(dx*dx + dy*dy))
        
        # Position on the approach side of the other person
        offset = self.radius + other.radius + 5
        target_x = other.position[0] - (dx / dist) * offset
        target_y = other.position[1] - (dy / dist) * offset
        
        self.target_position = self._clamp_to_bounds((target_x, target_y))
        self.waypoints = []  # Clear any existing waypoints
    
    def _pick_up_instrument(self, available: List['Instrument']) -> Optional['Instrument']:
        """Pick up an instrument from the preparation table."""
        for instrument in available:
            if (instrument.state.name == 'ON_TABLE' and 
                instrument.holder is None):
                instrument.attach_to(self)
                self.held_instrument = instrument
                return instrument
        return None
    
    def _lay_down_instrument(self) -> None:
        """Put an instrument back on the preaparation table"""
        if self.held_instrument is not None:
                self.held_instrument.detach()
                self.held_instrument = None
    
    def _complete_handover_give(self):
        """Complete giving an instrument to partner."""
        if self.held_instrument and self.handover_partner:
            instrument = self.held_instrument
            self.held_instrument = None
            instrument.complete_handover(self.handover_partner)
            self.handover_partner.held_instrument = instrument
            
            # Update states
            if self.person_type == PersonType.ASSISTANT:
                # Stay near the doctor to potentially receive instrument back
                self.state = AssistantState.WAITING_BY_DOCTOR
                # Keep handover_partner reference so we know which doctor to wait by
                # Set original position near the doctor for idle movement
                self.original_position = self.position
                # Partner (doctor) transitions to HOLDING
                self.handover_partner.handover_partner = None
                self.handover_partner.state = DoctorState.HOLDING
                # Don't clear self.handover_partner - we need it to track the doctor
            else:
                # Doctor giving to assistant
                self.state = DoctorState.IDLE
                self.handover_partner.handover_partner = None
                self.handover_partner.state = AssistantState.HOLDING
                # After receiving, assistant should return to prep table
                self.handover_partner.transition_pause = self.rng.integers(5, 20)
                self.handover_partner = None
    
    def _complete_handover_receive(self):
        """Complete receiving an instrument from partner."""
        # For assistant receiving from doctor: move back to prep table
        if self.person_type == PersonType.ASSISTANT:
            self.state = AssistantState.MOVING_FROM_DOCTOR
            # Set target back towards prep table area
            self._set_target_near_prep_table()
            self.handover_partner = None
    
    def get_instrument_attach_point(self) -> Tuple[float, float]:
        """Get the point on the circle edge where instrument attaches."""
        angle = self.instrument_attach_angle
        x = self.position[0] + self.radius * math.cos(angle)
        y = self.position[1] + self.radius * math.sin(angle)
        return (x, y)
    
    def get_current_action(self) -> Optional[ActionLabel]:
        """Get the current action label for YOLO annotation.
        
        PERSON_HOLDS is active whenever the person is holding an instrument,
        EXCEPT when in PREPARING, WORKING, or HANDOVER states.
        """
        # Check for specific action states first (these override HOLDS)
        if self.person_type == PersonType.ASSISTANT:
            if self.state == AssistantState.PREPARING:
                return ActionLabel.ASSISTANT_PREPARES
            elif self.state == AssistantState.GIVING:
                return ActionLabel.ASSISTANT_GIVES
            elif self.state == AssistantState.RECEIVING:
                return ActionLabel.ASSISTANT_RECEIVES
        else:  # Doctor
            if self.state == DoctorState.WORKING:
                return ActionLabel.DOCTOR_WORKS
            # Doctor RECEIVING/GIVING are part of handovers - the label
            # comes from the other person (assistant) to avoid duplicates
            elif self.state in [DoctorState.GIVING, DoctorState.RECEIVING]:
                return None
        
        # PERSON_HOLDS is active when holding an instrument and not in special states
        if self.held_instrument is not None:
            return ActionLabel.PERSON_HOLDS
        
        return None
    
    def is_in_handover(self) -> bool:
        """Check if person is currently in a handover action."""
        if self.person_type == PersonType.ASSISTANT:
            return self.state in [AssistantState.GIVING, AssistantState.RECEIVING]
        else:
            return self.state in [DoctorState.GIVING, DoctorState.RECEIVING]
    
    def get_bounding_box(self, img_size: int) -> Tuple[float, float, float, float]:
        """Get normalized bounding box for this person."""
        bbox = get_bounding_box(self.position, self.radius)
        return normalize_bbox(bbox, img_size)