"""
Pathfinding utilities for continuous movement.
Only used when config.use_grid_movement = False.
"""
from typing import List, Tuple, Optional
import math


class PathfindingManager:
    """Simple pathfinding for continuous movement using grid-based A*."""
    
    def __init__(self, scene_size: int, cell_size: int = 8):
        self.scene_size = scene_size
        self.cell_size = cell_size
        self.grid_size = scene_size // cell_size
        
        # 1 = walkable, 0 = blocked
        self.matrix = [[1 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.obstacles: List[Tuple[int, int, int, int]] = []
    
    def update_obstacles(self, obstacles: List, person_radius: float):
        """Update the obstacle grid."""
        self.obstacles = [obs.rect for obs in obstacles]
        margin = int(person_radius) + 1
        
        # Reset grid
        self.matrix = [[1 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Mark obstacle cells
        for ox, oy, ow, oh in self.obstacles:
            x_start = max(0, (ox - margin) // self.cell_size)
            y_start = max(0, (oy - margin) // self.cell_size)
            x_end = min(self.grid_size, (ox + ow + margin) // self.cell_size + 1)
            y_end = min(self.grid_size, (oy + oh + margin) // self.cell_size + 1)
            
            for gx in range(x_start, x_end):
                for gy in range(y_start, y_end):
                    self.matrix[gy][gx] = 0
    
    def get_path(self, start: Tuple[float, float], 
                 end: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Find a path from start to end using BFS."""
        start_grid = self._scene_to_grid(start)
        end_grid = self._scene_to_grid(end)
        
        # Clamp to bounds
        start_grid = self._clamp_grid(start_grid)
        end_grid = self._clamp_grid(end_grid)
        
        # Find nearest walkable cells
        start_grid = self._find_nearest_walkable(start_grid) or start_grid
        end_grid = self._find_nearest_walkable(end_grid) or end_grid
        
        # BFS pathfinding
        from collections import deque
        
        queue = deque([(start_grid, [start_grid])])
        visited = {start_grid}
        
        while queue:
            current, path = queue.popleft()
            
            if current == end_grid:
                return [self._grid_to_scene(p) for p in path]
            
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = current[0] + dx, current[1] + dy
                neighbor = (nx, ny)
                
                if neighbor in visited:
                    continue
                if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                    continue
                if self.matrix[ny][nx] == 0:
                    continue
                
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
        
        # No path found - return direct line
        return [start, end]
    
    def is_walkable(self, pos: Tuple[float, float]) -> bool:
        """Check if a scene position is walkable."""
        gx, gy = self._scene_to_grid(pos)
        if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
            return self.matrix[gy][gx] == 1
        return False
    
    def _scene_to_grid(self, pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert scene coordinates to grid coordinates."""
        return (int(pos[0] / self.cell_size), int(pos[1] / self.cell_size))
    
    def _grid_to_scene(self, pos: Tuple[int, int]) -> Tuple[float, float]:
        """Convert grid coordinates to scene coordinates (center of cell)."""
        return ((pos[0] + 0.5) * self.cell_size, (pos[1] + 0.5) * self.cell_size)
    
    def _clamp_grid(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Clamp grid position to valid range."""
        return (
            max(0, min(self.grid_size - 1, pos[0])),
            max(0, min(self.grid_size - 1, pos[1]))
        )
    
    def _find_nearest_walkable(self, pos: Tuple[int, int], 
                                max_search: int = 10) -> Optional[Tuple[int, int]]:
        """Find the nearest walkable cell."""
        gx, gy = pos
        
        if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
            if self.matrix[gy][gx] == 1:
                return pos
        
        for radius in range(1, max_search + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                            if self.matrix[ny][nx] == 1:
                                return (nx, ny)
        
        return None