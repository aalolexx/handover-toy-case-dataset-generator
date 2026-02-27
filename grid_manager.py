"""
Grid manager for discrete movement system.
Only used when config.use_grid_movement = True.
"""
from typing import Tuple, Optional, List, Set, TYPE_CHECKING
from collections import deque
import numpy as np

if TYPE_CHECKING:
    from person import Person


# Grid cell types
CELL_EMPTY = 0
CELL_TABLE = 1
CELL_OBSTACLE = 2


class GridManager:
    """
    Manages a discrete grid for movement.
    
    Coordinates are (row, col) where:
    - row 0 is top, row (grid_size-1) is bottom
    - col 0 is left, col (grid_size-1) is right
    """
    
    def __init__(self, img_size: int, grid_size: int = 16):
        self.img_size = img_size
        self.grid_size = grid_size
        self.cell_size = img_size // grid_size
        
        self.static_grid = np.zeros((grid_size, grid_size), dtype=np.int8)
        self.person_positions: Set[Tuple[int, int]] = set()
        self.patient_table_cells: Set[Tuple[int, int]] = set()
        self.prep_table_cells: Set[Tuple[int, int]] = set()
    
    def pixel_to_grid(self, pixel_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert pixel coordinates to grid coordinates."""
        col = int(pixel_pos[0] // self.cell_size)
        row = int(pixel_pos[1] // self.cell_size)
        return (
            max(0, min(self.grid_size - 1, row)),
            max(0, min(self.grid_size - 1, col))
        )
    
    def grid_to_pixel(self, grid_pos: Tuple[int, int]) -> Tuple[float, float]:
        """Convert grid coordinates to pixel coordinates (center of cell)."""
        row, col = grid_pos
        x = col * self.cell_size + self.cell_size / 2
        y = row * self.cell_size + self.cell_size / 2
        return (x, y)
    
    def grid_to_pixel_topleft(self, grid_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Convert grid coordinates to pixel coordinates (top-left corner)."""
        row, col = grid_pos
        return (col * self.cell_size, row * self.cell_size)
    
    def add_table(self, rect: Tuple[int, int, int, int], is_patient_table: bool = False, 
                  is_prep_table: bool = False):
        """Mark grid cells occupied by a table."""
        x, y, w, h = rect
        
        start_col = x // self.cell_size
        start_row = y // self.cell_size
        end_col = (x + w - 1) // self.cell_size
        end_row = (y + h - 1) // self.cell_size
        
        for row in range(start_row, end_row + 1):
            for col in range(start_col, end_col + 1):
                if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
                    self.static_grid[row, col] = CELL_TABLE
                    if is_patient_table:
                        self.patient_table_cells.add((row, col))
                    if is_prep_table:
                        self.prep_table_cells.add((row, col))
    
    def add_obstacle(self, rect: Tuple[int, int, int, int]):
        """Mark grid cells occupied by an obstacle."""
        x, y, w, h = rect
        
        start_col = x // self.cell_size
        start_row = y // self.cell_size
        end_col = (x + w - 1) // self.cell_size
        end_row = (y + h - 1) // self.cell_size
        
        for row in range(start_row, end_row + 1):
            for col in range(start_col, end_col + 1):
                if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
                    self.static_grid[row, col] = CELL_OBSTACLE
    
    def is_walkable(self, pos: Tuple[int, int]) -> bool:
        """Check if a grid cell is walkable."""
        row, col = pos
        if not (0 <= row < self.grid_size and 0 <= col < self.grid_size):
            return False
        return self.static_grid[row, col] == CELL_EMPTY
    
    def is_occupied_by_person(self, pos: Tuple[int, int], exclude_id: int = -1, 
                               persons: List['Person'] = None) -> bool:
        """Check if a grid cell is occupied by another person."""
        if persons is None:
            return pos in self.person_positions
        
        for person in persons:
            if person.id != exclude_id and person.grid_pos == pos:
                return True
        return False
    
    def is_adjacent_to_patient_table(self, pos: Tuple[int, int]) -> bool:
        """Check if position is adjacent to patient table."""
        return self._is_adjacent_to_cells(pos, self.patient_table_cells)
    
    def is_adjacent_to_prep_table(self, pos: Tuple[int, int]) -> bool:
        """Check if position is adjacent to preparation table."""
        return self._is_adjacent_to_cells(pos, self.prep_table_cells)
    
    def _is_adjacent_to_cells(self, pos: Tuple[int, int], cells: Set[Tuple[int, int]]) -> bool:
        """Check if position is adjacent to any cell in the set."""
        row, col = pos
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                if (row + dr, col + dc) in cells:
                    return True
        return False
    
    def get_adjacent_walkable_cells(self, pos: Tuple[int, int], 
                                     persons: List['Person'] = None,
                                     exclude_id: int = -1) -> List[Tuple[int, int]]:
        """Get all walkable adjacent cells (4-directional)."""
        row, col = pos
        adjacent = []
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (row + dr, col + dc)
            if self.is_walkable(neighbor):
                if persons is None or not self.is_occupied_by_person(neighbor, exclude_id, persons):
                    adjacent.append(neighbor)
        
        return adjacent
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int],
                  persons: List['Person'] = None, exclude_id: int = -1) -> List[Tuple[int, int]]:
        """Find shortest path using BFS."""
        if start == goal:
            return [start]
        
        if not self.is_walkable(goal):
            goal = self._find_nearest_walkable(goal)
            if goal is None:
                return []
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            for neighbor in self.get_adjacent_walkable_cells(current, persons, exclude_id):
                if neighbor in visited:
                    continue
                
                new_path = path + [neighbor]
                
                if neighbor == goal:
                    return new_path
                
                visited.add(neighbor)
                queue.append((neighbor, new_path))
        
        return []
    
    def find_cell_adjacent_to_table(self, table_cells: Set[Tuple[int, int]], 
                                     from_pos: Tuple[int, int],
                                     persons: List['Person'] = None,
                                     exclude_id: int = -1) -> Optional[Tuple[int, int]]:
        """Find the nearest walkable cell adjacent to a table."""
        candidates = []
        
        for table_cell in table_cells:
            row, col = table_cell
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (row + dr, col + dc)
                if self.is_walkable(neighbor):
                    if persons is None or not self.is_occupied_by_person(neighbor, exclude_id, persons):
                        dist = abs(neighbor[0] - from_pos[0]) + abs(neighbor[1] - from_pos[1])
                        candidates.append((dist, neighbor))
        
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]
        
        return None
    
    def _find_nearest_walkable(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Find nearest walkable cell to given position."""
        row, col = pos
        
        for radius in range(1, self.grid_size):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if abs(dr) == radius or abs(dc) == radius:
                        neighbor = (row + dr, col + dc)
                        if self.is_walkable(neighbor):
                            return neighbor
        
        return None
    
    def get_random_walkable_cell(self, rng: np.random.Generator,
                                  persons: List['Person'] = None,
                                  exclude_id: int = -1) -> Optional[Tuple[int, int]]:
        """Get a random walkable and unoccupied cell."""
        walkable = []
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                pos = (row, col)
                if self.is_walkable(pos):
                    if persons is None or not self.is_occupied_by_person(pos, exclude_id, persons):
                        walkable.append(pos)
        
        if walkable:
            return tuple(walkable[rng.integers(len(walkable))])
        
        return None
    
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two grid positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def are_adjacent(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """Check if two positions are adjacent (including diagonals)."""
        return abs(pos1[0] - pos2[0]) <= 1 and abs(pos1[1] - pos2[1]) <= 1 and pos1 != pos2