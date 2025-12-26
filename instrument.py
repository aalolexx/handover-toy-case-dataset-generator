"""
Instrument class for medical instruments in the surgery scene.
"""
from typing import Tuple, Optional, TYPE_CHECKING
import math

from enums import InstrumentState

if TYPE_CHECKING:
    from person import Person


class Instrument:
    """Represents a medical instrument (displayed as a triangle)."""
    
    def __init__(self, instrument_id: int, position: Tuple[float, float], 
                 size: int = 12):
        """
        Initialize an instrument.
        
        Args:
            instrument_id: Unique identifier
            position: Initial (x, y) position
            size: Size of the triangle
        """
        self.id = instrument_id
        self.position = position
        self.size = size
        self.state = InstrumentState.ON_TABLE
        self.holder: Optional['Person'] = None
        self.rotation_angle: float = 0.0  # For spinning during doctor_works
        self.spin_speed: float = 0.3  # Radians per frame when spinning
        
    def update(self):
        """Update instrument state each frame."""
        if self.state == InstrumentState.IN_USE:
            # Spin the instrument when in use
            self.rotation_angle += self.spin_speed
            if self.rotation_angle > 2 * math.pi:
                self.rotation_angle -= 2 * math.pi
    
    def attach_to(self, person: 'Person'):
        """Attach instrument to a person."""
        self.holder = person
        self.state = InstrumentState.HELD
    
    def detach(self):
        """Detach instrument from holder and return to table."""
        self.holder = None
        self.state = InstrumentState.ON_TABLE
    
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
    
    def get_triangle_points(self) -> list:
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
    
    def set_table_position(self, position: Tuple[float, float]):
        """Set position when on table."""
        self.position = position
