"""
Scene objects like tables and random medical equipment.
Supports both grid-aligned and continuous positioning.
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
    
    def position_on_edge(self, edge: str, image_size: int, margin: float = 10.0, 
                          use_grid: bool = False, grid_size: int = 16) -> Tuple[int, int]:
        """
        Position this object along a specific edge of the scene.
        
        Args:
            edge: One of 'top', 'bottom', 'left', 'right'
            image_size: Size of the scene (assumes square)
            margin: Margin from the edge (only used in continuous mode)
            use_grid: If True, snap to grid cells
            grid_size: Number of grid cells (only used if use_grid=True)
            
        Returns:
            The (x, y) position for the top-left corner of the object
        """
        if use_grid:
            return self._position_on_edge_grid(edge, image_size, grid_size)
        else:
            return self._position_on_edge_continuous(edge, image_size, margin)
    
    def _position_on_edge_grid(self, edge: str, image_size: int, grid_size: int) -> Tuple[int, int]:
        """Position on edge with grid alignment."""
        cell_size = image_size // grid_size
        
        # Snap dimensions to grid
        width_cells = max(1, (self.width + cell_size - 1) // cell_size)
        height_cells = max(1, (self.height + cell_size - 1) // cell_size)
        self.width = width_cells * cell_size
        self.height = height_cells * cell_size
        
        if edge == 'top':
            max_col = grid_size - width_cells
            col = random.randint(0, max(0, max_col))
            row = 0
        elif edge == 'bottom':
            max_col = grid_size - width_cells
            col = random.randint(0, max(0, max_col))
            row = grid_size - height_cells
        elif edge == 'left':
            max_row = grid_size - height_cells
            col = 0
            row = random.randint(0, max(0, max_row))
        elif edge == 'right':
            max_row = grid_size - height_cells
            col = grid_size - width_cells
            row = random.randint(0, max(0, max_row))
        else:
            raise ValueError(f"Invalid edge: {edge}")
        
        return (col * cell_size, row * cell_size)
    
    def _position_on_edge_continuous(self, edge: str, image_size: int, margin: float) -> Tuple[int, int]:
        """Position on edge with continuous coordinates."""
        if edge == 'top':
            x = random.randint(int(margin), int(image_size - self.width - margin))
            y = int(margin)
        elif edge == 'bottom':
            x = random.randint(int(margin), int(image_size - self.width - margin))
            y = int(image_size - self.height - margin)
        elif edge == 'left':
            x = int(margin)
            y = random.randint(int(margin), int(image_size - self.height - margin))
        elif edge == 'right':
            x = int(image_size - self.width - margin)
            y = random.randint(int(margin), int(image_size - self.height - margin))
        else:
            raise ValueError(f"Invalid edge: {edge}")
        
        return (x, y)


class PatientTable(SceneObject):
    """The patient/operation table."""
    
    def __init__(self, x: int, y: int, width: int, height: int,
                 color: Tuple[int, int, int]):
        super().__init__(x, y, width, height, color, "patient_table")


class PreparationTable(SceneObject):
    """The preparation table where instruments are prepared."""
    
    def __init__(self, x: int, y: int, width: int, height: int,
                 color: Tuple[int, int, int]):
        super().__init__(x, y, width, height, color, "preparation_table")
        self.instrument_positions: list = []
        self._init_instrument_positions()
    
    def _init_instrument_positions(self, use_grid: bool = False, 
                                    grid_size: int = 16, img_size: int = 224):
        """Initialize positions where instruments can be placed."""
        self.instrument_positions = []
        
        if use_grid:
            cell_size = img_size // grid_size
            start_col = self.x // cell_size
            start_row = self.y // cell_size
            width_cells = max(1, self.width // cell_size)
            height_cells = max(1, self.height // cell_size)
            
            for row_offset in range(height_cells):
                for col_offset in range(width_cells):
                    x = (start_col + col_offset) * cell_size + cell_size / 2
                    y = (start_row + row_offset) * cell_size + cell_size / 2
                    self.instrument_positions.append((x, y))
        else:
            # Continuous mode - grid of positions within table
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


class OcclusionObject(SceneObject):
    """Rectangle occluding the scene."""

    def __init__(self, x: int, y: int, width: int, height: int,
                 color: Tuple[int, int, int], obj_id: int):
        super().__init__(x, y, width, height, color, f"occlusions_object_{obj_id}")
        self.obj_id = obj_id