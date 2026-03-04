"""
Export utilities for the dataset generator.
Supports JSON scene descriptions and ASCII grid visualization.
"""
import json
import os
from typing import Dict, List, Any, Optional, TYPE_CHECKING

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
        
    def setup_directories(self):
        """Create output directories if they don't exist."""
        os.makedirs(self.json_dir, exist_ok=True)
        if self.config.use_grid_movement:
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
        seen_handovers = set()  # To avoid duplicates
        
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
                        
                        active_handovers.append({
                            "giver_id": person.id,
                            "receiver_id": person.handover_partner.id,
                            "object_id": object_id,
                            "direction": direction
                        })
            
            entity = {
                "id": person.id,
                "role": role,
                "state": state_name,
                "position": position,
                "grid_position": grid_position,
                "is_holding": is_holding,
                "held_object_id": held_object_id,
                "is_in_handover": is_in_handover,
                "handover_partner_id": handover_partner_id
            }
            entities.append(entity)
        
        frame_desc = {
            "frame": frame_number,
            "grid_mode": self.config.use_grid_movement,
            "grid_size": self.config.grid_size if self.config.use_grid_movement else None,
            "entities": entities,
            "active_handovers": active_handovers
        }
        
        # Write JSON file
        json_path = os.path.join(self.json_dir, f"frame_{frame_number:06d}.json")
        with open(json_path, 'w') as f:
            json.dump(frame_desc, f, indent=2)
    
    def export_ascii_frame(self, pm: 'ProcessManager', frame_number: int):
        """
        Export frame as ASCII grid visualization.
        
        Legend:
            +  = Empty cell
            P  = Preparation table
            W  = Working table (patient table)
            O  = Other scene object
            g  = Green actor
            G  = Green actor (in handover)
            g¹ = Green actor (holding)
            G¹ = Green actor (holding + handover)
            b  = Blue actor
            B  = Blue actor (in handover)
            b¹ = Blue actor (holding)
            B¹ = Blue actor (holding + handover)
        """
        if not self.config.use_grid_movement:
            return
        
        grid_size = self.config.grid_size
        
        # Initialize grid with empty cells
        # Using 2-char width for alignment
        grid = [["+ " for _ in range(grid_size)] for _ in range(grid_size)]
        
        # Place preparation table cells
        if hasattr(pm, 'grid') and pm.grid:
            for row, col in pm.grid.prep_table_cells:
                if 0 <= row < grid_size and 0 <= col < grid_size:
                    grid[row][col] = "P "
            
            # Place patient/working table cells
            for row, col in pm.grid.patient_table_cells:
                if 0 <= row < grid_size and 0 <= col < grid_size:
                    grid[row][col] = "W "
        
        # Place other scene objects
        if hasattr(pm, 'scene_objects'):
            for obj in pm.scene_objects:
                # Get grid cell for this object's center
                obj_x, obj_y = obj.position
                col = int(obj_x // self.config.cell_size)
                row = int(obj_y // self.config.cell_size)
                if 0 <= row < grid_size and 0 <= col < grid_size:
                    # Only place if cell is empty
                    if grid[row][col] == "+ ":
                        grid[row][col] = "O "
        
        # Place persons on grid (overwrite objects if overlapping)
        for person in pm.persons:
            row, col = person.grid_pos
            
            # Bounds check
            if not (0 <= row < grid_size and 0 <= col < grid_size):
                continue
            
            # Determine character
            is_green = person.person_type == self.PersonType.ASSISTANT
            is_holding = person.held_instrument is not None
            is_handover = self._is_in_handover(person)
            
            # Base character: lowercase normal, uppercase in handover
            if is_green:
                char = "G" if is_handover else "g"
            else:
                char = "B" if is_handover else "b"
            
            # Add holding indicator (superscript 1)
            if is_holding:
                char = char + "¹"
            else:
                char = char + " "
            
            grid[row][col] = char
        
        # Build ASCII string
        lines = []
        
        # Header
        #lines.append(f"Frame: {frame_number}")
        #lines.append("")
        
        # Column numbers header
        col_header = "    " + "".join(f"{c:2d}" for c in range(grid_size))
        lines.append(col_header)
        lines.append("   +" + "--" * grid_size)
        
        # Grid rows with row numbers
        for row_idx, row in enumerate(grid):
            row_str = f"{row_idx:2d} |" + "".join(row)
            lines.append(row_str)
        
        # Legend
        #lines.append("")
        #lines.append("Legend:")
        #lines.append("  +  = Empty")
        #lines.append("  P  = Preparation table")
        #lines.append("  W  = Working table")
        #lines.append("  O  = Other object")
        #lines.append("  g  = Green")
        #lines.append("  G  = Green (handover)")
        #lines.append("  g¹ = Green (holding)")
        #lines.append("  G¹ = Green (holding + handover)")
        #lines.append("  b  = Blue")
        #lines.append("  B  = Blue (handover)")
        #lines.append("  b¹ = Blue (holding)")
        #lines.append("  B¹ = Blue (holding + handover)")
        
        # Write to file
        ascii_path = os.path.join(self.ascii_dir, f"frame_{frame_number:06d}.txt")
        with open(ascii_path, 'w') as f:
            f.write("\n".join(lines))
    
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
                "seed": self.config.seed
            },
            "handover_events": handover_events or []
        }
        
        summary_path = os.path.join(self.output_dir, "sequence_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)


def create_exporter(config: 'Config', output_dir: str) -> SceneExporter:
    """Factory function to create and initialize an exporter."""
    exporter = SceneExporter(config, output_dir)
    exporter.setup_directories()
    return exporter