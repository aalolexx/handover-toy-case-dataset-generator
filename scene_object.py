"""
Scene objects like tables and random medical equipment.
"""
from typing import Tuple
from dataclasses import dataclass


@dataclass
class SceneObject:
    """Base class for scene objects (rectangles)."""
    
    x: int
    y: int
    width: int
    height: int
    color: Tuple[int, int, int]
    name: str = "object"
    
    @property
    def rect(self) -> Tuple[int, int, int, int]:
        """Get rectangle as (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center point."""
        return (self.x + self.width / 2, self.y + self.height / 2)
    
    def contains_point(self, point: Tuple[float, float]) -> bool:
        """Check if point is inside the object."""
        px, py = point
        return (self.x <= px <= self.x + self.width and
                self.y <= py <= self.y + self.height)


class PatientTable(SceneObject):
    """The patient/operation table in the center of the scene."""
    
    def __init__(self, x: int, y: int, width: int, height: int,
                 color: Tuple[int, int, int]):
        super().__init__(x, y, width, height, color, "patient_table")


class PreparationTable(SceneObject):
    """The preparation table where instruments are prepared."""
    
    def __init__(self, x: int, y: int, width: int, height: int,
                 color: Tuple[int, int, int]):
        super().__init__(x, y, width, height, color, "preparation_table")
        self.instrument_positions: list = []  # Positions for instruments on table
        self._init_instrument_positions()
    
    def _init_instrument_positions(self):
        """Initialize positions where instruments can be placed on the table."""
        # Create a grid of positions on the table
        num_cols = max(1, self.width // 30)
        num_rows = max(1, self.height // 30)
        
        margin_x = 15
        margin_y = 10
        
        for row in range(num_rows):
            for col in range(num_cols):
                if num_cols > 1:
                    x = self.x + margin_x + col * (self.width - 2 * margin_x) / (num_cols - 1)
                else:
                    x = self.x + self.width / 2
                if num_rows > 1:
                    y = self.y + margin_y + row * (self.height - 2 * margin_y) / (num_rows - 1)
                else:
                    y = self.y + self.height / 2
                self.instrument_positions.append((x, y))
    
    def get_instrument_position(self, index: int) -> Tuple[float, float]:
        """Get a position for placing an instrument."""
        if self.instrument_positions:
            return self.instrument_positions[index % len(self.instrument_positions)]
        return self.center


class RandomMedicalObject(SceneObject):
    """Random medical equipment placed at scene edges."""
    
    def __init__(self, x: int, y: int, width: int, height: int,
                 color: Tuple[int, int, int], obj_id: int):
        super().__init__(x, y, width, height, color, f"medical_object_{obj_id}")
        self.obj_id = obj_id
