"""
Person class representing doctors and assistants with movement and state management.
"""
from typing import Tuple, Optional, List, TYPE_CHECKING
import math
import numpy as np

from enums import PersonType, AssistantState, DoctorState, ActionLabel
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
        self.instrument_attach_angle: float = 0.0
        
        # Movement
        self.target_position: Optional[Tuple[float, float]] = None
        self.waypoints: List[Tuple[float, float]] = []
        self.wander_timer: int = 0
        self.idle_movement_range: float = 15.0
        self.original_position: Tuple[float, float] = position
        
        # Velocity smoothing (prevents jitter)
        self.velocity: Tuple[float, float] = (0.0, 0.0)
        self.velocity_smoothing: float = 0.3
        
        # Path recalculation limiting
        self.frames_since_path_calc: int = 0
        self.min_frames_between_path_calc: int = 15
        
        # Oscillation detection
        self.position_history: List[Tuple[float, float]] = []
        self.position_history_size: int = 10
        self.oscillation_cooldown: int = 0
        
        # Collision avoidance
        self.avoidance_radius = config.person_radius * 4
        self.smoothed_avoidance: Tuple[float, float] = (0.0, 0.0)
        self.avoidance_smoothing: float = 0.4
        self.all_persons: List['Person'] = []
        
        # References (set by ProcessManager)
        self.pathfinding: Optional['PathfindingManager'] = None
        self.patient_table: Optional['PatientTable'] = None
        self.preparation_table: Optional['PreparationTable'] = None
        self.obstacles: List['SceneObject'] = []
        
        # State timing
        self.state_timer: int = 0
        self.state_duration: int = 0
        self.transition_pause: int = 0
        self.move_timeout: int = 0
        self.max_move_timeout: int = 200
        
        # Handover coordination
        self.handover_partner: Optional['Person'] = None
        self.handover_frame_count: int = 0
        
        # Post-handover separation
        self.separation_frames_remaining: int = 0
        self.separation_direction: Tuple[float, float] = (0.0, 0.0)
        self.min_separation_frames: int = 10
        self.separation_distance: float = config.person_radius * 2  # How far to separate
        
        # Pre-handover approach state
        self.is_approaching_for_handover: bool = False
        self.approach_partner: Optional['Person'] = None
    
    def set_tables(self, patient_table: 'PatientTable', 
                   preparation_table: 'PreparationTable',
                   obstacles: List['SceneObject'],
                   pathfinding: Optional['PathfindingManager'] = None):
        """Set references to scene tables, obstacles, and pathfinding."""
        self.patient_table = patient_table
        self.preparation_table = preparation_table
        self.obstacles = obstacles
        self.pathfinding = pathfinding
    
    def set_all_persons(self, persons: List['Person']):
        """Set reference to all persons for collision avoidance."""
        self.all_persons = persons
    
    def set_active(self, active: bool):
        """Set whether this person participates in handovers."""
        self.is_active = active
        if not active:
            if self.person_type == PersonType.DOCTOR:
                self.state = DoctorState.IDLE
            else:
                self.state = AssistantState.IDLE
    
    # -------------------------------------------------------------------------
    # Main Update Loop
    # -------------------------------------------------------------------------
    
    def update(self, available_instruments: List['Instrument'],
               active_doctors: List['Person'],
               active_assistants: List['Person']):
        """Update person state and position each frame."""
        
        # Handle post-handover separation first (highest priority)
        if self.separation_frames_remaining > 0:
            self._do_separation_movement()
            return
        
        if self.transition_pause > 0:
            self.transition_pause -= 1
            self._do_idle_movement()
            return
        
        if self.is_active:
            self._update_active_behavior(available_instruments, 
                                         active_doctors, active_assistants)
        else:
            self._update_passive_behavior()
        
        # Update instrument attachment angle
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
    
    def _update_passive_behavior(self):
        """Update behavior for inactive persons (random movement)."""
        self._do_wander_movement()
    
    # -------------------------------------------------------------------------
    # Separation Movement (Post-Handover)
    # -------------------------------------------------------------------------
    
    def _do_separation_movement(self):
        """Move away from handover partner after handover completes."""
        self.separation_frames_remaining -= 1
        
        # Calculate movement in separation direction
        move_speed = self.speed * 0.8  # Slightly slower separation
        dx = self.separation_direction[0] * move_speed
        dy = self.separation_direction[1] * move_speed
        
        new_pos = self._clamp_to_bounds((
            self.position[0] + dx,
            self.position[1] + dy
        ))
        
        # Only move if we won't collide
        if not self._would_collide_with_obstacles(new_pos):
            self.position = new_pos
        
        # Update instrument angle during separation
        if self.held_instrument:
            self.instrument_attach_angle = math.atan2(dy, dx)
    
    def _start_separation(self, from_partner: 'Person'):
        """Start the separation phase after a handover."""
        self.separation_frames_remaining = self.min_separation_frames
        
        # Calculate direction away from partner
        dx = self.position[0] - from_partner.position[0]
        dy = self.position[1] - from_partner.position[1]
        dist = math.sqrt(dx * dx + dy * dy)
        
        if dist > 0.001:
            self.separation_direction = (dx / dist, dy / dist)
        else:
            # Fallback: random direction
            angle = self.rng.uniform(0, 2 * math.pi)
            self.separation_direction = (math.cos(angle), math.sin(angle))
    
    # -------------------------------------------------------------------------
    # State Machines
    # -------------------------------------------------------------------------
    
    def _update_assistant_state(self, available_instruments: List['Instrument'],
                             active_doctors: List['Person']):
        """Update assistant state machine with goal-oriented behavior."""
        state = self.state
        
        # -------------------------------------------------------------------------
        # IDLE: Brief pause, then decide next action
        # -------------------------------------------------------------------------
        if state == AssistantState.IDLE:
            self._do_idle_movement()
            self.state_timer += 1
            
            # Short idle period, then take action
            if self.state_timer > self.rng.integers(10, 30):
                self.state_timer = 0
                
                if self.held_instrument is None:
                    # Goal: Get an instrument
                    if available_instruments:
                        self.state = AssistantState.MOVING_TO_PREP_TABLE
                        self._set_target_near_prep_table()
                        self.move_timeout = 0
                else:
                    # Goal: Deliver instrument to doctor
                    doctor = self._find_available_doctor(active_doctors)
                    if doctor:
                        self.handover_partner = doctor
                        self.state = AssistantState.MOVING_TO_DOCTOR
                        self.is_approaching_for_handover = True
                        self.approach_partner = doctor
                        self._set_target_for_exact_touch(doctor)
                        self.move_timeout = 0
        
        # -------------------------------------------------------------------------
        # MOVING_TO_PREP_TABLE: Going to pick up an instrument
        # -------------------------------------------------------------------------
        elif state == AssistantState.MOVING_TO_PREP_TABLE:
            self.move_timeout += 1
            
            # Check if close enough to prep table (proximity-based arrival)
            if self.preparation_table:
                dist_to_table = self._distance_to_rect(self.preparation_table.rect)
                if dist_to_table < self.radius + 15:
                    # Close enough - start preparing
                    self.state = AssistantState.PREPARING
                    self.state_timer = 0
                    self.state_duration = int(self.rng.integers(
                        max(1, self.config.prepare_duration_avg - 10),
                        self.config.prepare_duration_avg + 10))
                    self.move_timeout = 0
                    return
            
            if self._move_to_target():
                self.state = AssistantState.PREPARING
                self.state_timer = 0
                self.state_duration = int(self.rng.integers(
                    max(1, self.config.prepare_duration_avg - 10),
                    self.config.prepare_duration_avg + 10))
                self.move_timeout = 0
            elif self.move_timeout > self.max_move_timeout:
                # Timeout - reset and try again
                self._reset_to_idle()
        
        # -------------------------------------------------------------------------
        # PREPARING: At prep table, picking up instrument
        # -------------------------------------------------------------------------
        elif state == AssistantState.PREPARING:
            self._do_idle_movement()
            self.state_timer += 1
            
            if self.state_timer >= self.state_duration:
                if self.held_instrument is None:
                    instrument = self._pick_up_instrument(available_instruments)
                    if instrument:
                        # Successfully picked up - now find a doctor
                        doctor = self._find_available_doctor(active_doctors)
                        if doctor:
                            self.handover_partner = doctor
                            self.state = AssistantState.MOVING_TO_DOCTOR
                            self.is_approaching_for_handover = True
                            self.approach_partner = doctor
                            self._set_target_for_exact_touch(doctor)
                            self.move_timeout = 0
                        else:
                            # No doctor available, wait briefly then try again
                            self.state = AssistantState.HOLDING
                            self.state_timer = 0
                    else:
                        # No instrument available
                        self._reset_to_idle()
                else:
                    # Already holding instrument (returning it)
                    self._lay_down_instrument()
                    self._reset_to_idle()
        
        # -------------------------------------------------------------------------
        # HOLDING: Has instrument, waiting for available doctor
        # -------------------------------------------------------------------------
        elif state == AssistantState.HOLDING:
            self._do_idle_movement()
            self.state_timer += 1
            
            # Check for available doctor every few frames
            if self.state_timer % 10 == 0:
                doctor = self._find_available_doctor(active_doctors)
                if doctor:
                    self.handover_partner = doctor
                    self.state = AssistantState.MOVING_TO_DOCTOR
                    self.is_approaching_for_handover = True
                    self.approach_partner = doctor
                    self._set_target_for_exact_touch(doctor)
                    self.move_timeout = 0
                    self.state_timer = 0
            
            # If waiting too long, return instrument to table
            if self.state_timer > 300:  # ~5 seconds at 60fps
                self.state = AssistantState.MOVING_TO_PREP_TABLE
                self._set_target_near_prep_table()
                self.move_timeout = 0
                self.state_timer = 0
        
        # -------------------------------------------------------------------------
        # MOVING_TO_DOCTOR: Approaching for handover
        # -------------------------------------------------------------------------
        elif state == AssistantState.MOVING_TO_DOCTOR:
            self.move_timeout += 1
            
            if self.handover_partner:
                # Check distance to partner
                dist_to_partner = distance(self.position, self.handover_partner.position)
                touch_distance = self.radius + self.handover_partner.radius
                
                # Update target to track doctor's current position
                if self.move_timeout % 10 == 0:
                    self._set_target_for_exact_touch(self.handover_partner)
                
                # Check if we've reached touch position (at or closer than touch distance + small tolerance)
                if dist_to_partner <= touch_distance + self.speed:
                    # Snap to exact touch position
                    self._snap_to_exact_touch(self.handover_partner)
                    
                    if self._can_start_handover_with_partner():
                        self._start_giving_handover()
                    else:
                        # Doctor became unavailable, find another or wait
                        new_doctor = self._find_available_doctor(active_doctors)
                        if new_doctor and new_doctor.id != (self.handover_partner.id if self.handover_partner else -1):
                            self.handover_partner = new_doctor
                            self.approach_partner = new_doctor
                            self._set_target_for_exact_touch(new_doctor)
                            self.move_timeout = 0
                        else:
                            self.state = AssistantState.HOLDING
                            self.handover_partner = None
                            self.is_approaching_for_handover = False
                            self.approach_partner = None
                            self.state_timer = 0
                else:
                    # Continue approaching
                    self._move_towards_partner_for_handover()
            
            if self.move_timeout > self.max_move_timeout:
                # Timeout - doctor unreachable
                self.state = AssistantState.HOLDING

                if self.held_instrument is None:
                    print("-> PART A: No instrument but holding")
                self.handover_partner = None
                self.is_approaching_for_handover = False
                self.approach_partner = None
                self.move_timeout = 0
                self.state_timer = 0
        
        # -------------------------------------------------------------------------
        # GIVING: Handing instrument to doctor - STAND COMPLETELY STILL
        # -------------------------------------------------------------------------
        elif state == AssistantState.GIVING:
            # NO MOVEMENT during handover - stand completely still
            self.handover_frame_count += 1
            if self.handover_frame_count >= self.state_duration:
                self._complete_handover_give()
        
        # -------------------------------------------------------------------------
        # WAITING_BY_DOCTOR: Waiting for doctor to finish working
        # -------------------------------------------------------------------------
        elif state == AssistantState.WAITING_BY_DOCTOR:
            if self.handover_partner:
                doctor = self.handover_partner
                dist_to_doctor = distance(self.position, doctor.position)
                touch_distance = self.radius + doctor.radius
                
                # Check if doctor is done working and ready to give back
                if doctor.state == DoctorState.HOLDING and doctor.held_instrument is not None:
                    # Doctor is ready - approach for handover
                    if dist_to_doctor > touch_distance + self.speed * 2:
                        # Need to approach
                        self.is_approaching_for_handover = True
                        self.approach_partner = doctor
                        self._move_towards_partner_for_handover()
                    else:
                        # Close enough - just wait for doctor to initiate
                        self._snap_to_exact_touch(doctor)
                elif dist_to_doctor > touch_distance + self.radius * 2:
                    # Stay close to the doctor (but not too close)
                    self._set_target_near_person(doctor)
                    self._move_to_target()
                else:
                    # We're close enough, do minimal idle movement
                    self._do_minimal_idle_movement()
                
                # If doctor lost instrument somehow, go back to idle
                if doctor.held_instrument is None:
                    if doctor.state not in [DoctorState.GIVING, DoctorState.WORKING, DoctorState.HOLDING]:
                        self._reset_to_idle()
            else:
                self._reset_to_idle()
        
        # -------------------------------------------------------------------------
        # RECEIVING: Getting instrument back from doctor - STAND COMPLETELY STILL
        # -------------------------------------------------------------------------
        elif state == AssistantState.RECEIVING:
            # NO MOVEMENT during handover - stand completely still
            self.handover_frame_count += 1
            if self.handover_frame_count >= self.state_duration:
                self._complete_handover_receive()
        
        # -------------------------------------------------------------------------
        # MOVING_FROM_DOCTOR: Returning instrument to prep table
        # -------------------------------------------------------------------------
        elif state == AssistantState.MOVING_FROM_DOCTOR:
            self.move_timeout += 1
            
            # Check if close enough to prep table
            if self.preparation_table:
                dist_to_table = self._distance_to_rect(self.preparation_table.rect)
                if dist_to_table < self.radius + 15:
                    self.state = AssistantState.PREPARING
                    self.state_timer = 0
                    self.state_duration = int(self.rng.integers(10, 30))
                    self.move_timeout = 0
                    return
            
            if self._move_to_target():
                # Arrived at prep table area - go to preparing to put down instrument
                self.state = AssistantState.PREPARING
                self.state_timer = 0
                self.state_duration = int(self.rng.integers(10, 30))
                self.move_timeout = 0
            elif self.move_timeout > self.max_move_timeout:
                # Just go to preparing where we are
                self.state = AssistantState.PREPARING
                self.state_timer = 0
                self.state_duration = int(self.rng.integers(10, 30))
                self.move_timeout = 0

    def _distance_to_rect(self, rect: Tuple[int, int, int, int]) -> float:
        """Calculate distance from current position to nearest point on rectangle."""
        rx, ry, rw, rh = rect
        # Find nearest point on rectangle
        nearest_x = max(rx, min(self.position[0], rx + rw))
        nearest_y = max(ry, min(self.position[1], ry + rh))
        return distance(self.position, (nearest_x, nearest_y))

    def _find_available_doctor(self, active_doctors: List['Person']) -> Optional['Person']:
        """Find a doctor that can receive an instrument."""
        available = []
        
        for doctor in active_doctors:
            # Doctor must not already have an instrument
            if doctor.held_instrument is not None:
                continue
            
            # Doctor must be in a state that can receive
            if doctor.state not in [DoctorState.IDLE, DoctorState.HOLDING]:
                continue
            
            # Doctor must not be separating
            if doctor.separation_frames_remaining > 0:
                continue
            
            # Doctor must not already have an assistant coming to them
            already_targeted = False
            for person in self.all_persons:
                if person.id == self.id:
                    continue
                if (person.person_type == PersonType.ASSISTANT and 
                    person.state == AssistantState.MOVING_TO_DOCTOR and
                    person.handover_partner is not None and
                    person.handover_partner.id == doctor.id):
                    already_targeted = True
                    break
            
            if not already_targeted:
                available.append(doctor)
        
        if not available:
            return None
        
        # Prefer closest doctor
        return min(available, key=lambda d: distance(self.position, d.position))


    def _can_start_handover_with_partner(self) -> bool:
        """Check if we can start a handover with our current partner."""
        if not self.handover_partner:
            return False
        if self.handover_partner.held_instrument is not None:
            return False
        #if self.handover_partner.separation_frames_remaining > 0:
        #    return False
        return True


    def _start_giving_handover(self):
        """Start the giving handover process."""
        if not self.handover_partner:
            return
        
        # Ensure both parties are exactly touching
        self._snap_to_exact_touch(self.handover_partner)
        
        self.state = AssistantState.GIVING
        self.handover_partner.state = DoctorState.RECEIVING
        self.handover_partner.handover_partner = self
        self.handover_frame_count = 0
        self.handover_partner.handover_frame_count = 0
        self.state_duration = self.config.handover_duration
        self.handover_partner.state_duration = self.config.handover_duration
        self.move_timeout = 0
        
        # Clear approach flags
        self.is_approaching_for_handover = False
        self.approach_partner = None


    def _reset_to_idle(self):
        """Reset to idle state with clean slate."""
        self.state = AssistantState.IDLE
        self.state_timer = 0
        self.handover_partner = None
        self.move_timeout = 0
        self.waypoints = []
        self.is_approaching_for_handover = False
        self.approach_partner = None
    

    def _update_doctor_state(self, active_assistants: List['Person']):
        """Update doctor state machine."""
        state = self.state
        
        # -------------------------------------------------------------------------
        # IDLE: Waiting for instrument, stay near patient table
        # -------------------------------------------------------------------------
        if state == DoctorState.IDLE:
            # Check if an assistant is approaching for handover
            approaching_assistant = self._find_approaching_assistant(active_assistants)
            if approaching_assistant:
                # Move towards the assistant to meet them
                self._move_towards_approaching_assistant(approaching_assistant)
            else:
                self._do_idle_movement()
                
                # Stay near patient table
                if self.patient_table:
                    dist_to_table = distance(self.position, self.patient_table.center)
                    if dist_to_table > 60:
                        self._set_target_near_patient_table()
        
        # -------------------------------------------------------------------------
        # HOLDING: Has instrument, will start working
        # -------------------------------------------------------------------------
        elif state == DoctorState.HOLDING:
            self._do_idle_movement()
            self.state_timer += 1
            
            # Brief pause then start working
            if self.state_timer > self.rng.integers(10, 40):
                self.state = DoctorState.WORKING
                self.state_timer = 0
                self.state_duration = int(self.rng.normal(self.config.work_duration_avg, 10))
                self.state_duration = max(30, self.state_duration)  # Minimum work time
                if self.held_instrument:
                    self.held_instrument.start_use()
        
        # -------------------------------------------------------------------------
        # WORKING: Using the instrument
        # -------------------------------------------------------------------------
        elif state == DoctorState.WORKING:
            # Small movements while working
            self._do_idle_movement()
            self.state_timer += 1
            
            if self.state_timer >= self.state_duration:
                if self.held_instrument:
                    self.held_instrument.stop_use()
                
                # Try to give instrument back
                assistant = self._find_assistant_for_return(active_assistants)
                if assistant:
                    self._start_giving_to_assistant(assistant)
                else:
                    # No assistant available, keep holding and try again later
                    self.state = DoctorState.HOLDING
                    self.state_timer = 0
        
        # -------------------------------------------------------------------------
        # GIVING: Handing instrument back to assistant - STAND COMPLETELY STILL
        # -------------------------------------------------------------------------
        elif state == DoctorState.GIVING:
            # NO MOVEMENT during handover - stand completely still
            self.handover_frame_count += 1
            if self.handover_frame_count >= self.state_duration:
                self._complete_handover_give()
        
        # -------------------------------------------------------------------------
        # RECEIVING: Getting instrument from assistant - STAND COMPLETELY STILL
        # -------------------------------------------------------------------------
        elif state == DoctorState.RECEIVING:
            # NO MOVEMENT during handover - stand completely still
            # Handover completion is handled by assistant's GIVING state
            pass


    def _find_approaching_assistant(self, active_assistants: List['Person']) -> Optional['Person']:
        """Find an assistant that is approaching this doctor for a handover."""
        for assistant in active_assistants:
            if (assistant.is_approaching_for_handover and 
                assistant.approach_partner is not None and
                assistant.approach_partner.id == self.id):
                return assistant
        return None
    
    
    def _move_towards_approaching_assistant(self, assistant: 'Person'):
        """Move towards an assistant that is approaching for handover."""
        dist = distance(self.position, assistant.position)
        touch_distance = self.radius + assistant.radius
        
        # If we're close enough to touch, stop and wait for handover to start
        if dist <= touch_distance + self.speed:
            return
        
        # Move towards the assistant (but don't overshoot)
        dx = assistant.position[0] - self.position[0]
        dy = assistant.position[1] - self.position[1]
        
        if dist > 0.001:
            # Move at half speed to let assistant lead the approach
            move_dist = min(self.speed * 0.5, dist - touch_distance)
            if move_dist > 0:
                new_pos = self._clamp_to_bounds((
                    self.position[0] + (dx / dist) * move_dist,
                    self.position[1] + (dy / dist) * move_dist
                ))
                
                if not self._would_collide_with_obstacles(new_pos):
                    self.position = new_pos


    def _find_assistant_for_return(self, active_assistants: List['Person']) -> Optional['Person']:
        """Find an assistant to return the instrument to."""
        # First priority: assistant waiting specifically for us
        for assistant in active_assistants:
            if (assistant.handover_partner is not None and
                assistant.handover_partner.id == self.id and
                assistant.held_instrument is None):
                #assistant.separation_frames_remaining == 0):
                return assistant
        
        # Second priority: any idle assistant without instrument
        idle_assistants = [
            a for a in active_assistants
            if a.held_instrument is None and
               a.separation_frames_remaining == 0
        ]

        if idle_assistants:
            return min(idle_assistants, key=lambda a: distance(self.position, a.position))
        
        print("STILL NO ASSISTANT!")

        for assistant in active_assistants:
            print(assistant.state)

        return None


    def _start_giving_to_assistant(self, assistant: 'Person'):
        """Start giving instrument to an assistant (only if close enough)."""
        dist = distance(self.position, assistant.position)
        touch_distance = self.radius + assistant.radius
        
        # Only start handover if assistant is close enough (no teleporting!)
        if dist > touch_distance + self.speed * 2:
            # Assistant is too far - they need to approach first
            # Signal the assistant to approach us
            assistant.handover_partner = self
            assistant.is_approaching_for_handover = True
            assistant.approach_partner = self
            assistant._set_target_for_exact_touch(self)
            assistant.state = AssistantState.MOVING_TO_DOCTOR
            
            # Doctor waits - will try again next frame
            self.state = DoctorState.HOLDING
            self.state_timer = 0
            return
        
        # Close enough - snap to exact touch and start handover
        assistant._snap_to_exact_touch(self)
        
        self.handover_partner = assistant
        self.state = DoctorState.GIVING
        assistant.state = AssistantState.RECEIVING
        assistant.handover_partner = self
        assistant.is_approaching_for_handover = False
        assistant.approach_partner = None
        
        self.handover_frame_count = 0
        assistant.handover_frame_count = 0
        self.state_duration = self.config.handover_duration
        assistant.state_duration = self.config.handover_duration
    
    # -------------------------------------------------------------------------
    # Handover Approach Movement
    # -------------------------------------------------------------------------
    
    def _set_target_for_exact_touch(self, partner: 'Person'):
        """Set target position to exactly touch the partner (no overlap)."""
        dx = partner.position[0] - self.position[0]
        dy = partner.position[1] - self.position[1]
        dist = max(0.001, math.sqrt(dx * dx + dy * dy))
        
        # Target is at the edge of partner's circle
        touch_distance = self.radius + partner.radius
        
        self.target_position = (
            partner.position[0] - (dx / dist) * touch_distance,
            partner.position[1] - (dy / dist) * touch_distance
        )
        self.waypoints = []
    
    def _snap_to_exact_touch(self, partner: 'Person'):
        """Snap this person to exactly touch the partner (no overlap)."""
        dx = partner.position[0] - self.position[0]
        dy = partner.position[1] - self.position[1]
        dist = max(0.001, math.sqrt(dx * dx + dy * dy))
        
        touch_distance = self.radius + partner.radius
        
        # Only adjust if not already at exact touch
        if abs(dist - touch_distance) > 0.5:
            self.position = (
                partner.position[0] - (dx / dist) * touch_distance,
                partner.position[1] - (dy / dist) * touch_distance
            )
    
    def _move_towards_partner_for_handover(self):
        """Move directly towards handover partner for clean approach."""
        if not self.handover_partner:
            return
        
        partner = self.handover_partner
        dx = partner.position[0] - self.position[0]
        dy = partner.position[1] - self.position[1]
        dist = math.sqrt(dx * dx + dy * dy)
        
        if dist < 0.001:
            return
        
        touch_distance = self.radius + partner.radius
        
        # Calculate how much to move
        dist_to_touch = dist - touch_distance
        move_dist = min(self.speed, max(0, dist_to_touch))
        
        if move_dist > 0:
            new_pos = (
                self.position[0] + (dx / dist) * move_dist,
                self.position[1] + (dy / dist) * move_dist
            )
            new_pos = self._clamp_to_bounds(new_pos)
            
            if not self._would_collide_with_obstacles(new_pos):
                self.position = new_pos
                self.instrument_attach_angle = math.atan2(dy, dx)
    
    # -------------------------------------------------------------------------
    # Movement
    # -------------------------------------------------------------------------
    
    def _do_idle_movement(self):
        """Small random movements while idle."""
        # Don't move during handover
        if self.is_in_handover():
            return
        
        self.wander_timer -= 1
        if self.wander_timer <= 0:
            offset_x = self.rng.uniform(-self.idle_movement_range, self.idle_movement_range)
            offset_y = self.rng.uniform(-self.idle_movement_range, self.idle_movement_range)
            target = (self.original_position[0] + offset_x, self.original_position[1] + offset_y)
            self.target_position = self._clamp_to_bounds(target)
            self.wander_timer = self.rng.integers(20, 60)
        
        self._move_to_target()
    
    def _do_minimal_idle_movement(self):
        """Very small movements while waiting (e.g., waiting by doctor)."""
        # Don't move during handover
        if self.is_in_handover():
            return
        
        self.wander_timer -= 1
        if self.wander_timer <= 0:
            # Much smaller movement range than normal idle
            small_range = self.idle_movement_range * 0.5
            offset_x = self.rng.uniform(-small_range, small_range)
            offset_y = self.rng.uniform(-small_range, small_range)
            target = (self.position[0] + offset_x, self.position[1] + offset_y)
            self.target_position = self._clamp_to_bounds(target)
            self.wander_timer = self.rng.integers(30, 60)
        
        self._move_to_target()
    
    def _do_wander_movement(self):
        """Random wandering movement for inactive persons."""
        self.wander_timer -= 1
        if self.wander_timer <= 0 or self.target_position is None:
            margin = self.radius + 10
            target = self._find_valid_random_position(margin)
            self.target_position = target
            self.waypoints = []
            self.wander_timer = self.rng.integers(60, 180)
        
        self._move_to_target()
    
    def _find_valid_random_position(self, margin: float) -> Tuple[float, float]:
        """Find a valid random position that doesn't collide with obstacles."""
        for _ in range(20):
            target_x = self.rng.uniform(margin, self.config.img_size - margin)
            target_y = self.rng.uniform(margin, self.config.img_size - margin)
            target = (target_x, target_y)
            if not self._would_collide_with_obstacles(target):
                return target
        return self.position
    
    def _move_to_target(self) -> bool:
        """
        Move towards target position using A* pathfinding with collision avoidance.
        
        Returns:
            True if target reached, False otherwise
        """
        self.frames_since_path_calc += 1
        if self.oscillation_cooldown > 0:
            self.oscillation_cooldown -= 1
        
        # Check for oscillation
        self._update_position_history()
        if self._is_oscillating():
            self._handle_oscillation()
            return False
        
        if self.target_position is None:
            self.velocity = (0.0, 0.0)
            return True
        
        # Check if reached target
        dist_to_target = distance(self.position, self.target_position)
        if dist_to_target < self.speed * 2:
            self.position = self.target_position
            self.waypoints = []
            self.velocity = (0.0, 0.0)
            return True
        
        # Recalculate path if needed
        if not self.waypoints and self.pathfinding and self.frames_since_path_calc >= self.min_frames_between_path_calc:
            path = self.pathfinding.get_path(self.position, self.target_position)
            if path and len(path) > 1:
                self.waypoints = path[1:]
            elif not path:
                escape_pos = self._find_escape_position()
                if escape_pos:
                    self.waypoints = [escape_pos]
            self.frames_since_path_calc = 0
        
        # Determine current waypoint
        current_waypoint = self.target_position
        if self.waypoints:
            current_waypoint = self.waypoints[0]
            if distance(self.position, current_waypoint) < self.speed * 1.5:
                self.waypoints.pop(0)
                current_waypoint = self.waypoints[0] if self.waypoints else self.target_position
        
        # Calculate movement direction
        dx = current_waypoint[0] - self.position[0]
        dy = current_waypoint[1] - self.position[1]
        dist = math.sqrt(dx * dx + dy * dy)
        
        if dist < 0.001:
            return False
        
        desired_x, desired_y = dx / dist, dy / dist
        
        # Get avoidance force (skip during handover approach)
        if not self.is_approaching_for_handover:
            avoid_x, avoid_y = self._get_avoidance_force()
            
            # Scale avoidance by priority
            avoidance_strength = max(0.2, 1.0 - (self._get_movement_priority() / 150.0))
            
            # Combine direction with avoidance
            combined_x = desired_x + avoid_x * avoidance_strength * 1.5
            combined_y = desired_y + avoid_y * avoidance_strength * 1.5
        else:
            combined_x = desired_x
            combined_y = desired_y
        
        combined_dist = math.sqrt(combined_x * combined_x + combined_y * combined_y)
        if combined_dist > 0.001:
            combined_x /= combined_dist
            combined_y /= combined_dist
        
        # Apply velocity smoothing
        target_velocity = (combined_x * self.speed, combined_y * self.speed)
        self.velocity = (
            self.velocity[0] + (target_velocity[0] - self.velocity[0]) * self.velocity_smoothing,
            self.velocity[1] + (target_velocity[1] - self.velocity[1]) * self.velocity_smoothing
        )
        
        # Clamp velocity magnitude
        vel_magnitude = math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        if vel_magnitude > self.speed:
            self.velocity = (
                self.velocity[0] / vel_magnitude * self.speed,
                self.velocity[1] / vel_magnitude * self.speed
            )
        
        # Apply movement
        next_pos = self._clamp_to_bounds((
            self.position[0] + self.velocity[0],
            self.position[1] + self.velocity[1]
        ))
        
        if not self._would_collide_with_obstacles(next_pos):
            self.position = next_pos
            self.instrument_attach_angle = math.atan2(self.velocity[1], self.velocity[0])
            return False
        else:
            self.waypoints = []
            self.frames_since_path_calc = self.min_frames_between_path_calc
            self._try_unstuck_movement()
        
        return False
    
    def _try_unstuck_movement(self) -> bool:
        """Try to get unstuck by moving in a valid direction."""
        base_angle = math.atan2(self.velocity[1], self.velocity[0]) if (
            abs(self.velocity[0]) > 0.001 or abs(self.velocity[1]) > 0.001
        ) else 0
        
        best_pos = None
        best_score = float('inf')
        
        for angle_offset in [0, math.pi/4, -math.pi/4, math.pi/2, -math.pi/2, 
                             3*math.pi/4, -3*math.pi/4, math.pi]:
            angle = base_angle + angle_offset
            test_pos = self._clamp_to_bounds((
                self.position[0] + self.speed * math.cos(angle),
                self.position[1] + self.speed * math.sin(angle)
            ))
            
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
        
        return False
    
    # -------------------------------------------------------------------------
    # Oscillation Detection
    # -------------------------------------------------------------------------
    
    def _update_position_history(self):
        """Track recent positions for oscillation detection."""
        self.position_history.append(self.position)
        if len(self.position_history) > self.position_history_size:
            self.position_history.pop(0)
    
    def _is_oscillating(self) -> bool:
        """Detect if the person is oscillating (moving back and forth)."""
        if len(self.position_history) < self.position_history_size or self.oscillation_cooldown > 0:
            return False
        
        recent = self.position_history[-1]
        old = self.position_history[0]
        
        # Calculate total distance traveled vs displacement
        total_dist = sum(
            distance(self.position_history[i-1], self.position_history[i])
            for i in range(1, len(self.position_history))
        )
        displacement = distance(old, recent)
        
        # If we moved a lot but didn't get anywhere, we're oscillating
        return total_dist > self.speed * 8 and displacement < self.speed * 2
    
    def _handle_oscillation(self):
        """Break out of oscillation by making a decisive movement."""
        self.oscillation_cooldown = 30
        self.position_history.clear()
        self.waypoints = []
        
        if self.target_position:
            # Step to the side to break the cycle
            dx = self.target_position[0] - self.position[0]
            dy = self.target_position[1] - self.position[1]
            current_angle = math.atan2(dy, dx)
            
            offset_angle = self.rng.uniform(-math.pi/2, math.pi/2)
            side_angle = current_angle + offset_angle
            side_pos = self._clamp_to_bounds((
                self.position[0] + math.cos(side_angle) * self.radius * 2,
                self.position[1] + math.sin(side_angle) * self.radius * 2
            ))
            
            if not self._would_collide_with_obstacles(side_pos):
                self.position = side_pos
        else:
            # Move in a random valid direction
            for _ in range(8):
                angle = self.rng.uniform(0, 2 * math.pi)
                new_pos = self._clamp_to_bounds((
                    self.position[0] + math.cos(angle) * self.radius,
                    self.position[1] + math.sin(angle) * self.radius
                ))
                if not self._would_collide_with_obstacles(new_pos):
                    self.position = new_pos
                    break
    
    # -------------------------------------------------------------------------
    # Collision Avoidance
    # -------------------------------------------------------------------------
    
    def _get_avoidance_force(self) -> Tuple[float, float]:
        """Calculate smoothed steering force to avoid other persons."""
        raw_force_x, raw_force_y = 0.0, 0.0
        
        for other in self.all_persons:
            if other.id == self.id or self._should_skip_avoidance(other):
                continue
            
            dx = self.position[0] - other.position[0]
            dy = self.position[1] - other.position[1]
            dist = math.sqrt(dx * dx + dy * dy)
            
            if dist < self.avoidance_radius and dist > 0.001:
                min_separation = self.radius + other.radius
                overlap_factor = (self.avoidance_radius - dist) / self.avoidance_radius
                
                if dist < min_separation * 1.5:
                    overlap_factor *= 3.0
                
                raw_force_x += (dx / dist) * overlap_factor
                raw_force_y += (dy / dist) * overlap_factor
        
        # Smooth the avoidance force
        self.smoothed_avoidance = (
            self.smoothed_avoidance[0] + (raw_force_x - self.smoothed_avoidance[0]) * self.avoidance_smoothing,
            self.smoothed_avoidance[1] + (raw_force_y - self.smoothed_avoidance[1]) * self.avoidance_smoothing
        )
        
        return self.smoothed_avoidance
    
    def _should_skip_avoidance(self, other: 'Person') -> bool:
        """Check if we should skip collision avoidance with another person."""
        # Skip if this is our handover partner (either direction)
        if self.handover_partner is not None and self.handover_partner.id == other.id:
            return True
        if other.handover_partner is not None and other.handover_partner.id == self.id:
            return True
        
        # Skip if we're approaching this person for handover
        if self.is_approaching_for_handover and self.approach_partner is not None:
            if self.approach_partner.id == other.id:
                return True
        
        # Skip if other person is approaching us for handover
        if other.is_approaching_for_handover and other.approach_partner is not None:
            if other.approach_partner.id == self.id:
                return True
        
        return False
    
    def _get_movement_priority(self) -> int:
        """Get movement priority. Higher priority persons are avoided by lower priority ones."""
        if self.is_in_handover():
            return 100
        
        if self.person_type == PersonType.ASSISTANT:
            if self.state == AssistantState.MOVING_TO_DOCTOR:
                return 90
            if self.state == AssistantState.WAITING_BY_DOCTOR:
                return 85
        
        if self.person_type == PersonType.DOCTOR:
            if self.state == DoctorState.WORKING:
                return 80
            if self.is_active:
                return 60
        
        if self.is_active:
            return 50
        
        return 10
    
    # -------------------------------------------------------------------------
    # Collision Detection
    # -------------------------------------------------------------------------
    
    def _would_collide_with_obstacles(self, position: Tuple[float, float]) -> bool:
        """Check if position would cause collision with obstacles."""
        all_obstacles = self.obstacles.copy()
        if self.patient_table:
            all_obstacles.append(self.patient_table)
        if self.preparation_table:
            all_obstacles.append(self.preparation_table)
        
        for obs in all_obstacles:
            if circle_rect_collision(position, self.radius + 2, obs.rect):
                return True
        
        if self.pathfinding and not self.pathfinding.is_walkable(position):
            return True
        
        return False
    
    def _find_escape_position(self) -> Optional[Tuple[float, float]]:
        """Find a nearby walkable position when stuck."""
        for radius in range(1, 15):
            dist = radius * self.speed * 2
            for angle_step in range(8 * radius):
                angle = (angle_step / (8 * radius)) * 2 * math.pi
                test_pos = self._clamp_to_bounds((
                    self.position[0] + dist * math.cos(angle),
                    self.position[1] + dist * math.sin(angle)
                ))
                
                if not self._would_collide_with_obstacles(test_pos):
                    if self.pathfinding is None or self.pathfinding.is_walkable(test_pos):
                        return test_pos
        
        return None
    
    def _clamp_to_bounds(self, position: Tuple[float, float]) -> Tuple[float, float]:
        """Clamp position to scene bounds."""
        margin = self.radius + 5
        return (
            clamp(position[0], margin, self.config.img_size - margin),
            clamp(position[1], margin, self.config.img_size - margin)
        )
    
    # -------------------------------------------------------------------------
    # Target Setting
    # -------------------------------------------------------------------------
    
    def _set_target_near_prep_table(self):
        """Set target to a position near the preparation table."""
        if self.preparation_table:
            # Choose a side that's NOT against the scene edge
            valid_sides = self._get_valid_sides_for_table(self.preparation_table)
            if valid_sides:
                side = self.rng.choice(valid_sides)
            else:
                side = 'top'  # Fallback
            
            self.target_position = find_position_near_rect(
                self.preparation_table.rect, side, self.radius + 10,
                (0, 0, self.config.img_size, self.config.img_size),
                self.obstacles, self.radius)
            
            # Verify target is walkable, if not try other sides
            if self.target_position:
                if self._would_collide_with_obstacles(self.target_position):
                    for fallback_side in valid_sides:
                        if fallback_side != side:
                            self.target_position = find_position_near_rect(
                                self.preparation_table.rect, fallback_side, self.radius + 10,
                                (0, 0, self.config.img_size, self.config.img_size),
                                self.obstacles, self.radius)
                            if self.target_position and not self._would_collide_with_obstacles(self.target_position):
                                break
            
            if self.target_position:
                self.original_position = self.target_position
            self.waypoints = []
    
    def _set_target_near_patient_table(self):
        """Set target to a position near the patient table."""
        if self.patient_table:
            # Choose a side that's NOT against the scene edge
            valid_sides = self._get_valid_sides_for_table(self.patient_table)
            if valid_sides:
                side = self.rng.choice(valid_sides)
            else:
                side = 'top'  # Fallback
            
            self.target_position = find_position_near_rect(
                self.patient_table.rect, side, self.radius + 15,
                (0, 0, self.config.img_size, self.config.img_size),
                self.obstacles, self.radius)
            
            # Verify target is walkable, if not try other sides
            if self.target_position:
                if self._would_collide_with_obstacles(self.target_position):
                    for fallback_side in valid_sides:
                        if fallback_side != side:
                            self.target_position = find_position_near_rect(
                                self.patient_table.rect, fallback_side, self.radius + 15,
                                (0, 0, self.config.img_size, self.config.img_size),
                                self.obstacles, self.radius)
                            if self.target_position and not self._would_collide_with_obstacles(self.target_position):
                                break
            
            if self.target_position:
                self.original_position = self.target_position
            self.waypoints = []
    
    def _get_valid_sides_for_table(self, table) -> List[str]:
        """
        Determine which sides of a table are accessible (not against scene boundary).
        
        When tables are placed on edges, one side will be against the wall and
        inaccessible. This method returns only the sides where a person can stand.
        """
        margin = self.radius + 25  # Need enough space to stand
        valid_sides = []
        
        # Get table bounds
        tx, ty, tw, th = table.rect
        
        if tx <= tw:
            return ["right"]
        
        if ty <= th:
            return ["bottom"]
        
        if tx >= self.config.img_size - tw:
            return ["left"]
        
        if ty >= self.config.img_size - th:
            return ["top"]
        
        # Fallback: if somehow no sides are valid, return all (shouldn't happen)
        valid_sides = ['left', 'right', 'top', 'bottom']
        return valid_sides
    
    def _set_target_near_person(self, other: 'Person'):
        """Set target to move next to another person (with some distance)."""
        dx = other.position[0] - self.position[0]
        dy = other.position[1] - self.position[1]
        dist = max(1, math.sqrt(dx*dx + dy*dy))
        
        # Keep some distance when not doing handover
        offset = self.radius + other.radius + self.radius * 0.8
        
        self.target_position = self._clamp_to_bounds((
            other.position[0] - (dx / dist) * offset,
            other.position[1] - (dy / dist) * offset
        ))
        self.waypoints = []
    
    # -------------------------------------------------------------------------
    # Instrument Handling
    # -------------------------------------------------------------------------
    
    def _pick_up_instrument(self, available: List['Instrument']) -> Optional['Instrument']:
        """Pick up an instrument from the preparation table."""
        for instrument in available:
            if instrument.state.name == 'ON_TABLE' and instrument.holder is None:
                instrument.attach_to(self)
                self.held_instrument = instrument
                return instrument
        return None
    
    def _lay_down_instrument(self) -> None:
        """Put an instrument back on the preparation table."""
        if self.held_instrument is not None:
            self.held_instrument.detach()
            self.held_instrument = None
    
    def _complete_handover_give(self):
        """Complete giving an instrument to partner."""
        if self.held_instrument and self.handover_partner:
            instrument = self.held_instrument
            partner = self.handover_partner
            
            self.held_instrument = None
            instrument.complete_handover(partner)
            partner.held_instrument = instrument
            
            if self.person_type == PersonType.ASSISTANT:
                # Assistant gave to doctor
                self.state = AssistantState.WAITING_BY_DOCTOR
                self.original_position = self.position
                partner.handover_partner = None
                partner.state = DoctorState.HOLDING
                partner.state_timer = 0
                
                # Start separation for both
                self._start_separation(partner)
                partner._start_separation(self)
            else:
                # Doctor gave to assistant
                self.state = DoctorState.IDLE
                partner.handover_partner = None
                partner.state = AssistantState.HOLDING
                partner.transition_pause = 0  # Don't pause, start separating instead
                
                # Start separation for both
                self._start_separation(partner)
                partner._start_separation(self)
                
                self.handover_partner = None
    
    def _complete_handover_receive(self):
        """Complete receiving an instrument from partner."""
        if self.person_type == PersonType.ASSISTANT:
            partner = self.handover_partner
            
            self.state = AssistantState.MOVING_FROM_DOCTOR
            self._set_target_near_prep_table()
            
            # Start separation
            if partner:
                self._start_separation(partner)
            
            self.handover_partner = None
    
    # -------------------------------------------------------------------------
    # Public Interface
    # -------------------------------------------------------------------------
    
    def get_instrument_attach_point(self) -> Tuple[float, float]:
        """Get the point on the circle edge where instrument attaches."""
        return (
            self.position[0],# + self.radius * math.cos(self.instrument_attach_angle),
            self.position[1]# + self.radius * math.sin(self.instrument_attach_angle)
        )
    
    def get_current_action(self) -> Optional[ActionLabel]:
        """Get the current action label for YOLO annotation."""
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
        """Check if person is currently in a handover action."""
        if self.person_type == PersonType.ASSISTANT:
            return self.state in [AssistantState.GIVING, AssistantState.RECEIVING]
        else:
            return self.state in [DoctorState.GIVING, DoctorState.RECEIVING]
    
    def get_bounding_box(self, img_size: int) -> Tuple[float, float, float, float]:
        """Get normalized bounding box for this person."""
        bbox = get_bounding_box(self.position, self.radius)
        return normalize_bbox(bbox, img_size)