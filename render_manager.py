"""
RenderManager class responsible for rendering the surgery scene.
"""
from typing import List, Tuple
from PIL import Image, ImageDraw
import math

from config import Config
from person import Person
from instrument import Instrument
from scene_object import SceneObject, PatientTable, PreparationTable, RandomMedicalObject
from enums import InstrumentState


class RenderManager:
    """Handles rendering of the surgery scene to images."""
    
    def __init__(self, config: Config):
        """
        Initialize the render manager.
        
        Args:
            config: Scene configuration
        """
        self.config = config
        self.img_size = config.img_size
    
    def render_frame(self, persons: List[Person],
                     instruments: List[Instrument],
                     patient_table: PatientTable,
                     preparation_table: PreparationTable,
                     scene_objects: List[RandomMedicalObject]) -> Image.Image:
        """
        Render a single frame of the scene.
        
        Args:
            persons: List of all persons in the scene
            instruments: List of all instruments
            patient_table: The patient/operation table
            preparation_table: The preparation table
            scene_objects: List of random medical objects
        
        Returns:
            PIL Image of the rendered frame
        """
        # Create image with background color
        img = Image.new('RGB', (self.img_size, self.img_size), 
                        self.config.background_color)
        draw = ImageDraw.Draw(img)
        
        # Render order (back to front):
        # 1. Scene objects (random medical equipment at edges)
        # 2. Tables
        # 3. Instruments on tables
        # 4. People with their held instruments
        
        # 1. Render scene objects
        for obj in scene_objects:
            self._draw_rect(draw, obj)
        
        # 2. Render tables
        self._draw_rect(draw, patient_table)
        self._draw_rect(draw, preparation_table)
        
        # 3. Render instruments on tables
        for instrument in instruments:
            if instrument.state == InstrumentState.ON_TABLE:
                self._draw_instrument(draw, instrument)
        
        # 4. Render people sorted by Y position (for overlapping)
        # People with higher Y are rendered later (appear on top)
        sorted_persons = sorted(persons, key=lambda p: p.position[1])
        
        for person in sorted_persons:
            self._draw_person(draw, person)
        
        return img
    
    def _draw_rect(self, draw: ImageDraw.ImageDraw, obj: SceneObject):
        """Draw a rectangular scene object."""
        x, y, w, h = obj.rect
        draw.rectangle([x, y, x + w, y + h], fill=obj.color, 
                       outline=(50, 50, 50), width=1)
    
    def _draw_person(self, draw: ImageDraw.ImageDraw, person: Person):
        """Draw a person (circle) with their held instrument if any."""
        cx, cy = person.position
        r = person.radius
        
        # Draw circle
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                     fill=person.color, outline=(30, 30, 30), width=2)
        
        # Draw held instrument
        if person.held_instrument:
            self._draw_instrument(draw, person.held_instrument)
    
    def _draw_instrument(self, draw: ImageDraw.ImageDraw, instrument: Instrument):
        """Draw an instrument (triangle)."""
        points = instrument.get_triangle_points()
        # Convert to format expected by PIL
        flat_points = [(int(p[0]), int(p[1])) for p in points]
        draw.polygon(flat_points, fill=self.config.instrument_color,
                     outline=(50, 50, 50))
    
    def save_frame(self, img: Image.Image, path: str, quality: int = 95):
        """Save a frame to disk as JPEG."""
        img.save(path, 'JPEG', quality=quality)
