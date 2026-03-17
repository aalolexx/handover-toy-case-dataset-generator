"""
Instrument class for medical instruments in the surgery scene.
"""
from typing import Tuple, Optional, List, TYPE_CHECKING
import math

from enums import InstrumentState

if TYPE_CHECKING:
    from person import Person
    from config import Config


class Instrument:
    """Represents a medical instrument (displayed as triangle, square, or circle)."""
    
    # Shape constants
    SHAPE_TRIANGLE = "triangle"
    SHAPE_SQUARE = "square"
    SHAPE_CIRCLE = "circle"
    
    def __init__(self, instrument_id: int, position: Tuple[float, float], 
                 size: int = 12, config: Optional['Config'] = None,
                 rng=None):
        """
        Initialize an instrument.
        
        Args:
            instrument_id: Unique identifier
            position: Initial (x, y) position
            size: Size of the shape
            config: Config object for randomization settings
            rng: Random number generator for randomization
        """
        self.id = instrument_id
        self.position = position
        self.size = size
        self.state = InstrumentState.ON_TABLE
        self.holder: Optional['Person'] = None
        self.rotation_angle: float = 0.0  # For spinning during doctor_works
        self.spin_speed: float = 0.3  # Radians per frame when spinning
        self.opacity = 1.0
        
        # Store config and rng for randomization
        self.config = config
        self.rng = rng
        
        # Shape and color (default values, randomized on pickup if enabled)
        self.shape: str = self.SHAPE_TRIANGLE
        self.color: Optional[Tuple[int, int, int]] = None  # None means use default from config
        
        # Track if already randomized (only randomize once per pickup cycle)
        self._randomized_for_current_cycle: bool = False
        
    def update(self):
        """Update instrument state each frame."""
        if self.state == InstrumentState.IN_USE:
            # Spin the instrument when in use
            self.rotation_angle += self.spin_speed
            if self.rotation_angle > 2 * math.pi:
                self.rotation_angle -= 2 * math.pi
    
    def randomize_appearance(self):
        """Randomize the shape and color of the instrument."""
        if self.config is None or self.rng is None:
            return
        
        if not self.config.randomize_instruments:
            return
            
        # Randomize shape
        shapes = self.config.instrument_shapes
        self.shape = shapes[self.rng.integers(0, len(shapes))]
        
        # Randomize color
        colors = self.config.instrument_colors
        self.color = colors[self.rng.integers(0, len(colors))]
        
        self._randomized_for_current_cycle = True
    
    def attach_to(self, person: 'Person'):
        """Attach instrument to a person."""
        self.holder = person
        self.state = InstrumentState.HELD
        
        # Randomize appearance when first picked up in this cycle
        if not self._randomized_for_current_cycle:
            self.randomize_appearance()
    
    def detach(self):
        """Detach instrument from holder and return to table."""
        self.holder = None
        self.state = InstrumentState.ON_TABLE
        # Reset randomization flag so next pickup will randomize again
        self._randomized_for_current_cycle = False
        # Reset to default appearance when returned to table
        self.shape = self.SHAPE_TRIANGLE
        self.color = None
    
    def start_use(self):
        """Start using the instrument (doctor working)."""
        self.state = InstrumentState.IN_USE
    
    def stop_use(self):
        """Stop using the instrument."""
        if self.holder:
            self.state = InstrumentState.HELD
        else:
            self.state = InstrumentState.ON_TABLE
    
    def start_handover(self):
        """Mark instrument as being handed over."""
        self.state = InstrumentState.IN_HANDOVER
    
    def complete_handover(self, new_holder: 'Person'):
        """Complete handover to new person."""
        self.holder = new_holder
        self.state = InstrumentState.HELD
    
    def get_position(self) -> Tuple[float, float]:
        """Get current position of the instrument."""
        if self.holder:
            # Position at the edge of the holder's circle
            return self.holder.get_instrument_attach_point()
        return self.position
    
    def get_color(self) -> Tuple[int, int, int]:
        """Get the current color of the instrument."""
        if self.color is not None:
            return self.color
        if self.config is not None:
            return self.config.instrument_color
        return (134, 134, 145)  # Default gray
    
    def get_shape_points(self) -> List[Tuple[float, float]]:
        """Get the points for rendering the current shape.
        
        Returns:
            List of (x, y) tuples representing shape vertices,
            or for circle, returns center and radius info
        """
        if self.shape == self.SHAPE_TRIANGLE:
            return self.get_triangle_points()
        elif self.shape == self.SHAPE_SQUARE:
            return self.get_square_points()
        else:  # circle
            return self.get_circle_points()
    
    def get_triangle_points(self) -> List[Tuple[float, float]]:
        """Get the three points of the triangle for rendering.
        
        Returns:
            List of 3 (x, y) tuples representing triangle vertices
        """
        cx, cy = self.get_position()
        angle = self.rotation_angle
        
        # Create an equilateral triangle
        points = []
        for i in range(3):
            point_angle = angle + (i * 2 * math.pi / 3) - math.pi / 2
            px = cx + self.size * math.cos(point_angle)
            py = cy + self.size * math.sin(point_angle)
            points.append((px, py))
        
        return points
    
    def get_square_points(self) -> List[Tuple[float, float]]:
        """Get the four points of the square for rendering.
        
        Returns:
            List of 4 (x, y) tuples representing square vertices
        """
        cx, cy = self.get_position()
        angle = self.rotation_angle
        
        # Create a square (rotated by current angle)
        # Distance from center to corner
        half_diag = self.size * math.sqrt(2) / 2
        
        points = []
        for i in range(4):
            point_angle = angle + (i * math.pi / 2) + math.pi / 4
            px = cx + half_diag * math.cos(point_angle)
            py = cy + half_diag * math.sin(point_angle)
            points.append((px, py))
        
        return points
    
    def get_circle_points(self) -> List[Tuple[float, float]]:
        """Get points representing the circle for rendering.
        
        For circles, we return a list with just the center point.
        The renderer should use self.size as the radius.
        
        Returns:
            List with single (x, y) tuple representing circle center
        """
        cx, cy = self.get_position()
        return [(cx, cy)]
    
    def get_circle_params(self) -> Tuple[Tuple[float, float], float]:
        """Get circle center and radius for rendering.
        
        Returns:
            Tuple of (center_point, radius)
        """
        cx, cy = self.get_position()
        return ((cx, cy), self.size * 0.8)  # Slightly smaller to match triangle/square visual size
    
    def set_table_position(self, position: Tuple[float, float]):
        """Set position when on table."""
        self.position = position