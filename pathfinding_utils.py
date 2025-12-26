"""
Pathfinding module using A* algorithm for navigating around obstacles.
"""
from typing import List, Tuple, Optional
import math

from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder


class PathfindingGrid:
    """
    A grid-based pathfinding system for navigating around obstacles.
    
    Converts the continuous scene space to a discrete grid for A* pathfinding,
    then converts the resulting path back to continuous coordinates.
    """
    
    def __init__(self, scene_size: int, cell_size: int = 1):
        """
        Initialize the pathfinding grid.
        
        Args:
            scene_size: Size of the scene in pixels (assumes square)
            cell_size: Size of each grid cell in pixels (smaller = more precise but slower)
        """
        self.scene_size = scene_size
        self.cell_size = cell_size
        self.grid_size = scene_size // cell_size
        
        # Initialize grid matrix (1 = walkable, 0 = blocked)
        self.matrix = [[1 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Cache for obstacle rects
        self.obstacles: List[Tuple[int, int, int, int]] = []
        
        # A* finder with diagonal movement
        self.finder = AStarFinder(diagonal_movement=DiagonalMovement.only_when_no_obstacle)
    
    def set_obstacles(self, obstacles: List[Tuple[int, int, int, int]], person_radius: float):
        """
        Set obstacles in the grid.
        
        Args:
            obstacles: List of obstacle rectangles (x, y, width, height)
            person_radius: Radius of the person (for collision margin)
        """
        self.obstacles = obstacles
        margin = int(person_radius) + 8
        
        # Reset grid to all walkable
        self.matrix = [[1 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Mark obstacle cells as blocked
        for ox, oy, ow, oh in obstacles:
            # Expand obstacle by margin
            x_start = max(0, (ox - margin) // self.cell_size)
            y_start = max(0, (oy - margin) // self.cell_size)
            x_end = min(self.grid_size, (ox + ow + margin) // self.cell_size + 1)
            y_end = min(self.grid_size, (oy + oh + margin) // self.cell_size + 1)
            
            for gx in range(x_start, x_end):
                for gy in range(y_start, y_end):
                    self.matrix[gy][gx] = 0
    
    def find_path(self, start: Tuple[float, float], 
                  end: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Find a path from start to end position.
        
        Args:
            start: Starting position (x, y) in scene coordinates
            end: Ending position (x, y) in scene coordinates
        
        Returns:
            List of waypoints in scene coordinates, or empty list if no path found
        """
        # Convert scene coordinates to grid coordinates
        start_grid = self._scene_to_grid(start)
        end_grid = self._scene_to_grid(end)
        
        # Clamp to grid bounds
        start_grid = (
            max(0, min(self.grid_size - 1, start_grid[0])),
            max(0, min(self.grid_size - 1, start_grid[1]))
        )
        end_grid = (
            max(0, min(self.grid_size - 1, end_grid[0])),
            max(0, min(self.grid_size - 1, end_grid[1]))
        )
        
        # Check if start or end are blocked - find nearest walkable cell
        start_grid = self._find_nearest_walkable(start_grid)
        end_grid = self._find_nearest_walkable(end_grid)
        
        if start_grid is None or end_grid is None:
            return []
        
        # Create fresh grid for pathfinding (finder modifies the grid)
        grid = Grid(matrix=self.matrix)
        
        start_node = grid.node(start_grid[0], start_grid[1])
        end_node = grid.node(end_grid[0], end_grid[1])
        
        # Find path
        path, _ = self.finder.find_path(start_node, end_node, grid)
        
        if not path:
            return []
        
        # Convert grid path to scene coordinates
        scene_path = [self._grid_to_scene((node.x, node.y)) for node in path]
        
        # Simplify path by removing unnecessary waypoints
        simplified_path = self._simplify_path(scene_path)
        
        return simplified_path
    
    def _scene_to_grid(self, pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert scene coordinates to grid coordinates."""
        gx = int(pos[0] / self.cell_size)
        gy = int(pos[1] / self.cell_size)
        return (gx, gy)
    
    def _grid_to_scene(self, pos: Tuple[int, int]) -> Tuple[float, float]:
        """Convert grid coordinates to scene coordinates (center of cell)."""
        x = (pos[0] + 0.5) * self.cell_size
        y = (pos[1] + 0.5) * self.cell_size
        return (x, y)
    
    def _find_nearest_walkable(self, pos: Tuple[int, int], 
                                max_search: int = 10) -> Optional[Tuple[int, int]]:
        """Find the nearest walkable cell to a given grid position."""
        gx, gy = pos
        
        # Check if current position is walkable
        if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
            if self.matrix[gy][gx] == 1:
                return pos
        
        # Search in expanding squares
        for radius in range(1, max_search + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                            if self.matrix[ny][nx] == 1:
                                return (nx, ny)
        
        return None
    
    def _simplify_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Simplify path by removing collinear points.
        Keeps only waypoints where direction changes significantly.
        """
        if len(path) <= 2:
            return path
        
        simplified = [path[0]]
        
        for i in range(1, len(path) - 1):
            prev = simplified[-1]
            current = path[i]
            next_pt = path[i + 1]
            
            # Calculate angles
            angle1 = math.atan2(current[1] - prev[1], current[0] - prev[0])
            angle2 = math.atan2(next_pt[1] - current[1], next_pt[0] - current[0])
            
            # Keep point if direction changes significantly (more than ~15 degrees)
            angle_diff = abs(angle1 - angle2)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            
            if angle_diff > 0.25:  # ~15 degrees
                simplified.append(current)
        
        simplified.append(path[-1])
        
        return simplified
    
    def is_position_walkable(self, pos: Tuple[float, float]) -> bool:
        """Check if a scene position is walkable."""
        gx, gy = self._scene_to_grid(pos)
        if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
            return self.matrix[gy][gx] == 1
        return False


class PathfindingManager:
    """
    Manages pathfinding for all persons in the scene.
    Provides a simple interface for the Person class to use.
    """
    
    def __init__(self, scene_size: int, cell_size: int = 8):
        """
        Initialize the pathfinding manager.
        
        Args:
            scene_size: Size of the scene in pixels
            cell_size: Size of each grid cell
        """
        self.grid = PathfindingGrid(scene_size, cell_size)
        self.scene_size = scene_size
    
    def update_obstacles(self, obstacles: List, person_radius: float):
        """
        Update the obstacle grid.
        
        Args:
            obstacles: List of SceneObject instances with .rect property
            person_radius: Radius of persons for collision margin
        """
        obstacle_rects = [obs.rect for obs in obstacles]
        self.grid.set_obstacles(obstacle_rects, person_radius)
    
    def get_path(self, start: Tuple[float, float], 
                 end: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Get a path from start to end, avoiding obstacles.
        
        Args:
            start: Starting position (x, y)
            end: Target position (x, y)
        
        Returns:
            List of waypoints, or empty list if no path found
        """
        return self.grid.find_path(start, end)
    
    def is_walkable(self, pos: Tuple[float, float]) -> bool:
        """Check if a position is walkable."""
        return self.grid.is_position_walkable(pos)
