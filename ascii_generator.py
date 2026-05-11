"""
ASCII Generator for grid-based scene visualization.
Creates 4x4 sub-grid representations of each cell for detailed state visualization.
"""
from typing import List, Tuple, Optional, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from render_manager import RenderManager
    from process_manager import ProcessManager
    from person import Person
    from instrument import Instrument
    from config import Config


class AsciiGenerator:
    """
    Generates detailed ASCII representations of the scene.
    
    Each grid cell is represented as a 4x4 sub-grid:
    - Actors have a border of their role color (g/b) with a 2x2 inner body
    - Tables have borders (P/W/O) with inner body for instruments
    - Empty cells use '.' pattern
    - Occlusion rectangles are rendered as '?'
    - Instruments use ASCII symbols: ^, +, *, x, o
    """
    
    SUBCELL_SIZE = 4  # 4x4 characters per grid cell
    EMPTY_CHAR = '-'
    OCCLUSION_CHAR = '?'
    
    # Instrument symbols for randomized instruments
    INSTRUMENT_SYMBOLS = ['^', '+', '*', 'x', 'o']
    DEFAULT_INSTRUMENT_SYMBOL = '^'
    
    def __init__(self, config: 'Config'):
        """
        Initialize the ASCII generator.
        
        Args:
            config: Scene configuration
        """
        self.config = config
        self.grid_size = config.grid_size
        
        # Full ASCII grid dimensions
        self.ascii_width = self.grid_size * self.SUBCELL_SIZE
        self.ascii_height = self.grid_size * self.SUBCELL_SIZE
        
        # Cache for instrument symbols (consistent across frames)
        self._instrument_symbols: dict = {}
        
        # Import enums
        from enums import PersonType, AssistantState, DoctorState, InstrumentState
        self.PersonType = PersonType
        self.AssistantState = AssistantState
        self.DoctorState = DoctorState
        self.InstrumentState = InstrumentState
    
    def _get_instrument_symbol(self, instrument: 'Instrument') -> str:
        """
        Get the ASCII symbol for an instrument.
        Ensures consistency across frames for the same instrument.
        """
        if instrument.opacity == 0:
            return self.EMPTY_CHAR
        if instrument.id not in self._instrument_symbols:
            if self.config.randomize_instruments:
                symbol_idx = instrument.id % len(self.INSTRUMENT_SYMBOLS)
                self._instrument_symbols[instrument.id] = self.INSTRUMENT_SYMBOLS[symbol_idx]
            else:
                self._instrument_symbols[instrument.id] = self.DEFAULT_INSTRUMENT_SYMBOL
        
        return self._instrument_symbols[instrument.id]
    
    def _create_empty_grid(self) -> List[List[str]]:
        """Create an empty ASCII grid filled with '.' characters."""
        return [[self.EMPTY_CHAR for _ in range(self.ascii_width)] 
                for _ in range(self.ascii_height)]
    
    def _render_cell_border(self, grid: List[List[str]], 
                            grid_row: int, grid_col: int, 
                            border_char: str):
        """Render a 4x4 cell with a border character and '.' interior."""
        start_row = grid_row * self.SUBCELL_SIZE
        start_col = grid_col * self.SUBCELL_SIZE
        
        for i in range(self.SUBCELL_SIZE):
            for j in range(self.SUBCELL_SIZE):
                row = start_row + i
                col = start_col + j
                
                if row >= self.ascii_height or col >= self.ascii_width:
                    continue
                
                is_border = (i == 0 or i == self.SUBCELL_SIZE - 1 or 
                            j == 0 or j == self.SUBCELL_SIZE - 1)
                
                grid[row][col] = border_char if is_border else self.EMPTY_CHAR
    
    def _render_cell_filled(self, grid: List[List[str]], 
                           grid_row: int, grid_col: int, 
                           fill_char: str):
        """Render a 4x4 cell completely filled with a character."""
        start_row = grid_row * self.SUBCELL_SIZE
        start_col = grid_col * self.SUBCELL_SIZE
        
        for i in range(self.SUBCELL_SIZE):
            for j in range(self.SUBCELL_SIZE):
                row = start_row + i
                col = start_col + j
                
                if row < self.ascii_height and col < self.ascii_width:
                    grid[row][col] = fill_char
    
    def _place_instrument_in_cell(self, grid: List[List[str]], 
                                   grid_row: int, grid_col: int,
                                   instrument_symbol: str,
                                   position: int = 0):
        """
        Place an instrument symbol in the inner 2x2 body of a cell.
        
        Position: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
        """
        start_row = grid_row * self.SUBCELL_SIZE
        start_col = grid_col * self.SUBCELL_SIZE
        
        inner_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        pos_offset = inner_positions[position % 4]
        row = start_row + pos_offset[0]
        col = start_col + pos_offset[1]
        
        if row < self.ascii_height and col < self.ascii_width:
            grid[row][col] = instrument_symbol
    
    def _render_preparation_table(self, grid: List[List[str]], pm: 'ProcessManager'):
        """Render the preparation table cells."""
        # Get grid manager from process manager or from a person
        grid_manager = getattr(pm, 'grid', None)
        if grid_manager is None and pm.persons:
            grid_manager = pm.persons[0].grid
        
        if grid_manager is None or not hasattr(grid_manager, 'prep_table_cells'):
            return
        
        for row, col in grid_manager.prep_table_cells:
            if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
                self._render_cell_border(grid, row, col, 'P')
                # Fill inner body with instruments (table always looks full)
                self._place_instrument_in_cell(grid, row, col, '^', 0)
                self._place_instrument_in_cell(grid, row, col, '+', 1)
                self._place_instrument_in_cell(grid, row, col, '*', 2)
                self._place_instrument_in_cell(grid, row, col, 'x', 3)

    def _render_patient_table(self, grid: List[List[str]], pm: 'ProcessManager'):
        """Render the patient/working table cells."""
        grid_manager = getattr(pm, 'grid', None)
        if grid_manager is None and pm.persons:
            grid_manager = pm.persons[0].grid
        
        if grid_manager is None or not hasattr(grid_manager, 'patient_table_cells'):
            return
        
        for row, col in grid_manager.patient_table_cells:
            if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
                self._render_cell_border(grid, row, col, 'W')
    
    def _render_scene_objects(self, grid: List[List[str]], pm: 'ProcessManager'):
        """Render other scene objects."""
        if not hasattr(pm, 'scene_objects'):
            return
        
        cell_size = self.config.cell_size
        
        for obj in pm.scene_objects:
            if hasattr(obj, 'position'):
                obj_x, obj_y = obj.position
            elif hasattr(obj, 'rect'):
                obj_x = obj.rect[0] + obj.rect[2] // 2
                obj_y = obj.rect[1] + obj.rect[3] // 2
            else:
                continue
            
            col = int(obj_x // cell_size)
            row = int(obj_y // cell_size)
            
            if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
                self._render_cell_border(grid, row, col, 'O')
    
    def _render_actor(self, grid: List[List[str]], person: 'Person', frame_number: int):
        """Render an actor (doctor or assistant) on the grid."""
        row, col = person.grid_pos
        
        if not (0 <= row < self.grid_size and 0 <= col < self.grid_size):
            return
        
        is_assistant = person.person_type == self.PersonType.ASSISTANT
        border_char = 'g' if is_assistant else 'b'
        
        self._render_cell_border(grid, row, col, border_char)
        
        if person.held_instrument is not None:
            instrument = person.held_instrument
            is_hidden = getattr(instrument, 'is_hidden', False)
            
            if not is_hidden:
                symbol = self._get_instrument_symbol(instrument)
                
                is_working = (person.person_type == self.PersonType.DOCTOR and 
                             person.state == self.DoctorState.WORKING)
                
                if is_working:
                    position = frame_number % 4
                    self._place_instrument_in_cell(grid, row, col, symbol, position)
                else:
                    self._place_instrument_in_cell(grid, row, col, symbol, 0)
    
    def _render_occlusion_rectangles(self, grid: List[List[str]], pm: 'ProcessManager'):
        """Render occlusion rectangles as blocks of '?' characters."""        
        cell_size = self.config.cell_size
        
        state = pm.get_scene_state()
        occlusion_objects = state['occlusion_objects']

        for occlusion_obj in occlusion_objects:
            if hasattr(occlusion_obj, 'rect'):
                x, y, w, h = occlusion_obj.rect
            elif hasattr(occlusion_obj, 'x'):
                x, y, w, h = occlusion_obj.x, occlusion_obj.y, occlusion_obj.width, occlusion_obj.height
            else:
                continue
            
            start_col = max(0, int(x // cell_size))
            start_row = max(0, int(y // cell_size))
            end_col = min(self.grid_size, int((x + w) // cell_size) + 1)
            end_row = min(self.grid_size, int((y + h) // cell_size) + 1)
            
            for grid_row in range(start_row, end_row):
                for grid_col in range(start_col, end_col):
                    self._render_cell_filled(grid, grid_row, grid_col, self.OCCLUSION_CHAR)
    
    def generate_frame(self, pm: 'ProcessManager',frame_number: int) -> str:
        """Generate ASCII representation of a single frame."""
        grid = self._create_empty_grid()
        
        self._render_preparation_table(grid, pm)
        self._render_patient_table(grid, pm)
        self._render_scene_objects(grid, pm)
        
        for person in pm.persons:
            self._render_actor(grid, person, frame_number)
        
        self._render_occlusion_rectangles(grid, pm)
        
        return self._grid_to_string(grid)
    
    def _grid_to_string(self, grid: List[List[str]]) -> str:
        """Convert the ASCII grid to a string with headers."""
        lines = []
        
        col_header = "     "
        for c in range(self.grid_size):
            col_header += f"{c:<4}"
        lines.append(col_header)
        lines.append("    +" + "-" * self.ascii_width)
        
        for ascii_row in range(self.ascii_height):
            if ascii_row % self.SUBCELL_SIZE == 0:
                grid_row = ascii_row // self.SUBCELL_SIZE
                row_prefix = f"{grid_row:3d} |"
            else:
                row_prefix = "    |"
            
            row_str = row_prefix + "".join(grid[ascii_row])
            lines.append(row_str)
        
        return "\n".join(lines)
    
    def generate_legend(self) -> str:
        """Generate a legend explaining the ASCII symbols."""
        return """
ASCII Grid Legend:
==================
.    = Empty space
g    = Green actor (assistant) border
b    = Blue actor (doctor) border
P    = Preparation table
W    = Working/Patient table
O    = Scene object
?    = Occluded area

Instrument symbols (inside actor cells):
^, +, *, x, o = Different instruments

Actor cell structure (4x4):
  gggg
  g..g  <- Inner 2x2 body (instruments shown here)
  g..g
  gggg

Working state: Instrument rotates through inner body positions
"""