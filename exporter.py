"""
Export utilities for the dataset generator.
Supports JSON scene descriptions and ASCII grid visualization.
"""
import json
import os
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from ascii_generator import AsciiGenerator


if TYPE_CHECKING:
    from process_manager import ProcessManager
    from person import Person
    from config import Config


class SceneExporter:
    """Handles exporting scene data in JSON and ASCII formats."""
    
    def __init__(self, config: 'Config', output_dir: str):
        self.config = config
        self.output_dir = output_dir
        self.json_dir = os.path.join(output_dir, "json_frames")
        self.ascii_dir = os.path.join(output_dir, "ascii_frames")
        
        # Import enums here to avoid circular imports
        from enums import PersonType, AssistantState, DoctorState
        self.PersonType = PersonType
        self.AssistantState = AssistantState
        self.DoctorState = DoctorState

        self.ascii_generator = None
        
        # Event counters for statistics
        self.total_handovers = 0
        self.total_fake_handovers = 0
        self.total_failed_handovers = 0
        self.total_approach_only = 0
        
    def setup_directories(self):
        """Create output directories if they don't exist."""
        if self.config.export_json:
            os.makedirs(self.json_dir, exist_ok=True)
        if self.config.use_grid_movement and self.config.export_ascii:
            os.makedirs(self.ascii_dir, exist_ok=True)
    
    def export_frame(self, process_manager: 'ProcessManager', frame_number: int):
        """Export a single frame in all enabled formats."""
        # Always export JSON
        if self.config.export_json:
            self.export_json_frame(process_manager, frame_number)
        
        # Export ASCII only if grid mode is enabled
        if self.config.use_grid_movement and self.config.export_ascii:
            self.export_ascii_frame(process_manager, frame_number)
    
    def _is_in_handover(self, person: 'Person') -> bool:
        """Check if person is currently in a handover (GIVING or RECEIVING state)."""
        if person.person_type == self.PersonType.ASSISTANT:
            return person.state in [self.AssistantState.GIVING, self.AssistantState.RECEIVING]
        else:
            return person.state in [self.DoctorState.GIVING, self.DoctorState.RECEIVING]
    
    def _get_role(self, person: 'Person') -> str:
        """Get role string for person."""
        return "green" if person.person_type == self.PersonType.ASSISTANT else "blue"
    
    def export_json_frame(self, pm: 'ProcessManager', frame_number: int):
        """
        Export frame as JSON scene description.
        """
        entities = []
        active_handovers = []
        failed_handovers = []
        approach_only_events = []
        seen_handovers = set()  # To avoid duplicates
        seen_failed = set()
        seen_approach = set()
        
        for person in pm.persons:
            role = self._get_role(person)
            state_name = person.state.name if hasattr(person.state, 'name') else str(person.state)
            
            # Position
            pos = person.position
            position = {"x": float(pos[0]), "y": float(pos[1])}
            
            # Grid position (if applicable)
            grid_position = None
            if self.config.use_grid_movement:
                gpos = person.grid_pos
                grid_position = {"row": int(gpos[0]), "col": int(gpos[1])}
            
            # Holding status
            is_holding = person.held_instrument is not None
            held_object_id = person.held_instrument.id if is_holding else None
            
            # Handover status
            is_in_handover = self._is_in_handover(person)
            handover_partner_id = None
            
            if is_in_handover and person.handover_partner:
                handover_partner_id = person.handover_partner.id
                
                # Record handover event (only from giver's perspective to avoid duplicates)
                is_giver = (
                    (person.person_type == self.PersonType.ASSISTANT and 
                     person.state == self.AssistantState.GIVING) or
                    (person.person_type == self.PersonType.DOCTOR and 
                     person.state == self.DoctorState.GIVING)
                )
                
                if is_giver:
                    # Find the object being transferred
                    # The giver holds the object on frame 1, receiver on frame 2
                    object_id = None
                    if person.held_instrument:
                        object_id = person.held_instrument.id
                    elif person.handover_partner and person.handover_partner.held_instrument:
                        object_id = person.handover_partner.held_instrument.id
                    
                    handover_key = (person.id, person.handover_partner.id)
                    if handover_key not in seen_handovers and object_id is not None:
                        seen_handovers.add(handover_key)
                        
                        # Determine direction
                        if person.person_type == self.PersonType.ASSISTANT:
                            direction = "green_to_blue"
                        else:
                            direction = "blue_to_green"
                        
                        # Check if fake handover (same color)
                        is_fake = person.is_fake_handover if hasattr(person, 'is_fake_handover') else False
                        if is_fake:
                            # Override direction for fake handovers
                            if person.person_type == self.PersonType.ASSISTANT:
                                direction = "green_to_green"
                            else:
                                direction = "blue_to_blue"
                        
                        active_handovers.append({
                            "giver_id": person.id,
                            "receiver_id": person.handover_partner.id,
                            "object_id": object_id,
                            "direction": direction,
                            "is_fake": is_fake
                        })
                        
                        # Increment counters
                        if is_fake:
                            self.total_fake_handovers += 1
                        else:
                            self.total_handovers += 1
            
            # Check for failed handover event
            is_failed = getattr(person, 'is_failed_handover', False)
            if is_failed:
                partner = getattr(person, 'failed_handover_partner', None)
                if partner:
                    # Create unique key to avoid duplicates
                    fail_key = tuple(sorted([person.id, partner.id]))
                    if fail_key not in seen_failed:
                        seen_failed.add(fail_key)
                        object_id = None
                        if person.held_instrument:
                            object_id = person.held_instrument.id
                        failed_handovers.append({
                            "actor1_id": fail_key[0],
                            "actor2_id": fail_key[1],
                            "object_id": object_id
                        })
                        self.total_failed_handovers += 1
            
            # Check for approach-only event (actors reached each other)
            is_approach_event = getattr(person, 'is_approach_only_event', False)
            if is_approach_event:
                partner = getattr(person, 'approach_only_partner', None)
                if partner:
                    approach_key = tuple(sorted([person.id, partner.id]))
                    if approach_key not in seen_approach:
                        seen_approach.add(approach_key)
                        approach_only_events.append({
                            "approacher_id": person.id,
                            "target_id": partner.id
                        })
                        self.total_approach_only += 1
            
            entity = {
                "id": person.id,
                "role": role,
                "state": state_name,
                "position": position,
                "grid_position": grid_position,
                "is_holding": is_holding,
                "held_object_id": held_object_id,
                "is_in_handover": is_in_handover,
                "handover_partner_id": handover_partner_id,
                "is_failed_handover": is_failed,
                "is_approach_only": is_approach_event
            }
            entities.append(entity)
        
        frame_desc = {
            "frame": frame_number,
            "grid_mode": self.config.use_grid_movement,
            "grid_size": self.config.grid_size if self.config.use_grid_movement else None,
            "entities": entities,
            "active_handovers": active_handovers,
            "failed_handovers": failed_handovers,
            "approach_only_events": approach_only_events
        }
        
        # Write JSON file
        json_path = os.path.join(self.json_dir, f"frame_{frame_number:06d}.json")
        with open(json_path, 'w') as f:
            json.dump(frame_desc, f, indent=2)
    
    def export_ascii_frame(self, pm: 'ProcessManager', frame_number: int):
        """Export frame as ASCII grid visualization using AsciiGenerator."""
        if not self.config.use_grid_movement:
            return
        
        if self.ascii_generator is None:
            self.ascii_generator = AsciiGenerator(self.config)
        
        ascii_content = self.ascii_generator.generate_frame(pm, frame_number)
        
        ascii_path = os.path.join(self.ascii_dir, f"frame_{frame_number:06d}.txt")
        with open(ascii_path, 'w') as f:
            f.write(ascii_content)
    
    def export_sequence_summary(self, pm: 'ProcessManager', total_frames: int, 
                                 handover_events: Optional[List[Dict]] = None):
        """Export a summary JSON with metadata about the full sequence."""
        summary = {
            "total_frames": total_frames,
            "config": {
                "num_green_actors": self.config.num_assistants,
                "num_blue_actors": self.config.num_doctors,
                "num_active_green": self.config.num_active_assistants,
                "num_active_blue": self.config.num_active_doctors,
                "num_objects": self.config.num_instruments,
                "grid_mode": self.config.use_grid_movement,
                "grid_size": self.config.grid_size if self.config.use_grid_movement else None,
                "image_size": self.config.img_size,
                "seed": self.config.seed,
                "handover_success_rate": self.config.handover_success_rate,
                "approach_without_ho_rate": self.config.approach_without_ho_rate,
                "enable_fake_handovers": self.config.enable_fake_handovers,
                "fake_handover_prob": self.config.fake_handover_prob if self.config.enable_fake_handovers else None
            },
            "statistics": {
                "total_handovers": self.total_handovers,
                "total_fake_handovers": self.total_fake_handovers,
                "total_failed_handovers": self.total_failed_handovers,
                "total_approach_only": self.total_approach_only
            },
            "handover_events": handover_events or []
        }
        
        summary_path = os.path.join(self.output_dir, "sequence_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def clear_frame_flags(self, pm: 'ProcessManager'):
        """
        Clear per-frame event flags. Call this AFTER both annotation and export 
        have processed the frame to reset flags for the next frame.
        """
        for person in pm.persons:
            if hasattr(person, 'is_failed_handover'):
                person.is_failed_handover = False
            if hasattr(person, 'failed_handover_partner'):
                person.failed_handover_partner = None


def create_exporter(config: 'Config', output_dir: str) -> SceneExporter:
    """Factory function to create and initialize an exporter."""
    exporter = SceneExporter(config, output_dir)
    exporter.setup_directories()
    return exporter