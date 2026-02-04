"""
Scene objects like tables and random medical equipment.
"""
from typing import Tuple
from dataclasses import dataclass
import random


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
    

    @property
    def position(self):
        return (self.x, self.y)

    @position.setter
    def position(self, value):
        self.x, self.y = value
    
    def contains_point(self, point: Tuple[float, float]) -> bool:
        """Check if point is inside the object."""
        px, py = point
        return (self.x <= px <= self.x + self.width and
                self.y <= py <= self.y + self.height)
    
    def get_random_position(self, image_size, avoid_points: list = None, avoid_radius: float = 90.0) -> Tuple[float, float]:
        """Get a random position within the object."""
        rx = int(random.uniform(self.x + avoid_radius, self.x + image_size - avoid_radius))
        ry = int(random.uniform(self.y + avoid_radius, self.y + image_size - avoid_radius))
        if (avoid_points is not None and
            rx > avoid_points[0] and rx < avoid_points[0] + avoid_radius and
            ry > avoid_points[1] and ry < avoid_points[1] + avoid_radius):
            return self.get_random_position(image_size, avoid_points, avoid_radius)
        return (rx, ry)
    
    def position_on_edge(self, edge: str, image_size: int, margin: float = 0.0) -> Tuple[int, int]:
        """
        Position this object along a specific edge of the scene.
        
        Args:
            edge: One of 'top', 'bottom', 'left', 'right'
            image_size: Size of the scene (assumes square)
            margin: Margin from the edge
            
        Returns:
            The (x, y) position for the top-left corner of the object
        """
        if edge == 'top':
            # Along top edge, random x position
            x = image_size // 2#random.randint(int(margin), int(image_size - self.width - margin))
            y = int(margin)
        elif edge == 'bottom':
            # Along bottom edge, random x position
            x = image_size // 2#random.randint(int(margin), int(image_size - self.width - margin))
            y = int(image_size - self.height - margin)
        elif edge == 'left':
            # Along left edge, random y position
            x = int(margin)
            y = image_size // 2#random.randint(int(margin), int(image_size - self.height - margin))
        elif edge == 'right':
            # Along right edge, random y position
            x = int(image_size - self.width - margin)
            y = image_size // 2#random.randint(int(margin), int(image_size - self.height - margin))
        else:
            raise ValueError(f"Invalid edge: {edge}. Must be 'top', 'bottom', 'left', or 'right'")
        
        return (x, y)


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
        
        self.instrument_positions = []  # Reset positions
        
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


class OcclusionObject(SceneObject):
    """Rectangle occluding the scene simulating camera errors or general occlusions."""

    def __init__(self, x: int, y: int, width: int, height: int,
                 color: Tuple[int, int, int], obj_id: int):
        super().__init__(x, y, width, height, color, f"occlusions_object_{obj_id}")
        self.obj_id = obj_id