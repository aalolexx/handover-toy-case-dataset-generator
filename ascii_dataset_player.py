"""
AsciiDatasetPlayer - A simple class for playing through generated ASCII frames in the terminal.
"""
import os
import sys
import time
import argparse
from typing import List, Optional
from pathlib import Path


class AsciiDatasetPlayer:
    """
    Play through ASCII dataset frames in the terminal.
    Simple terminal-based playback with basic controls.
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize the ASCII dataset player.
        
        Args:
            dataset_path: Path to the dataset output folder
        """
        self.dataset_path = Path(dataset_path)
        self.ascii_dir = self.dataset_path / 'ascii_frames'
        
        if not self.ascii_dir.exists():
            raise ValueError(f"ASCII frames directory not found: {self.ascii_dir}")
        
        # Get sorted list of ASCII frame files
        self.frame_files = sorted([
            f for f in self.ascii_dir.iterdir() 
            if f.suffix.lower() == '.txt' and f.name.startswith('frame_')
        ])
        
        if not self.frame_files:
            raise ValueError(f"No ASCII frames found in {self.ascii_dir}")
        
        print(f"Loaded {len(self.frame_files)} ASCII frames")
    
    def _clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def _load_frame(self, frame_index: int) -> str:
        """Load ASCII frame content from file."""
        if frame_index < 0 or frame_index >= len(self.frame_files):
            return ""
        
        with open(self.frame_files[frame_index], 'r') as f:
            return f.read()
    
    def play(self, fps: int = 10, loop: bool = True, start_frame: int = 0):
        """
        Play ASCII frames in the terminal.
        
        Args:
            fps: Frames per second (default: 10, ASCII doesn't need high fps)
            loop: Whether to loop playback
            start_frame: Starting frame index
        
        Controls:
            Ctrl+C: Stop playback
        """
        frame_delay = 1.0 / fps
        current_frame = start_frame
        
        print("\nASCII Dataset Player")
        print("=" * 40)
        print(f"Frames: {len(self.frame_files)}")
        print(f"FPS: {fps}")
        print("Press Ctrl+C to stop")
        print("=" * 40)
        time.sleep(1)
        
        try:
            while True:
                # Clear screen and show frame
                self._clear_screen()
                
                # Load and display frame
                frame_content = self._load_frame(current_frame)
                
                # Print header
                print(f"Frame: {current_frame + 1}/{len(self.frame_files)} | FPS: {fps} | Press Ctrl+C to stop")
                print("-" * 60)
                
                # Print frame content
                print(frame_content)
                
                # Wait
                time.sleep(frame_delay)
                
                # Advance frame
                current_frame += 1
                if current_frame >= len(self.frame_files):
                    if loop:
                        current_frame = 0
                    else:
                        break
        
        except KeyboardInterrupt:
            print("\n\nPlayback stopped")
    
    def play_interactive(self, start_frame: int = 0):
        """
        Play ASCII frames with interactive controls (requires getch-like input).
        
        Args:
            start_frame: Starting frame index
        
        Controls:
            n/RIGHT: Next frame
            p/LEFT: Previous frame  
            q: Quit
            SPACE: Toggle auto-play
        """
        try:
            
            HAS_TTY = True
        except ImportError:
            HAS_TTY = False
        
        if not HAS_TTY:
            print("Interactive mode not available on this platform.")
            print("Falling back to auto-play mode...")
            self.play(fps=5, start_frame=start_frame)
            return
        
        current_frame = start_frame
        auto_play = False
        fps = 5
        
        def get_key():
            """Get a single keypress."""
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
                # Handle arrow keys (escape sequences)
                if ch == '\x1b':
                    ch += sys.stdin.read(2)
                return ch
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        
        print("\nInteractive ASCII Player")
        print("=" * 40)
        print("Controls:")
        print("  n / RIGHT : Next frame")
        print("  p / LEFT  : Previous frame")
        print("  SPACE     : Toggle auto-play")
        print("  +/-       : Adjust speed")
        print("  q         : Quit")
        print("=" * 40)
        time.sleep(1)
        
        try:
            while True:
                # Clear and display
                self._clear_screen()
                
                frame_content = self._load_frame(current_frame)
                
                status = "[AUTO]" if auto_play else "[MANUAL]"
                print(f"Frame: {current_frame + 1}/{len(self.frame_files)} {status} | FPS: {fps}")
                print("-" * 60)
                print(frame_content)
                print("-" * 60)
                print("n/→:next | p/←:prev | SPACE:auto | +/-:speed | q:quit")
                
                if auto_play:
                    # Non-blocking check for input during auto-play
                    import select
                    time.sleep(1.0 / fps)
                    if select.select([sys.stdin], [], [], 0)[0]:
                        key = get_key()
                    else:
                        current_frame = (current_frame + 1) % len(self.frame_files)
                        continue
                else:
                    key = get_key()
                
                # Handle input
                if key == 'q':
                    break
                elif key == 'n' or key == '\x1b[C':  # n or right arrow
                    current_frame = (current_frame + 1) % len(self.frame_files)
                elif key == 'p' or key == '\x1b[D':  # p or left arrow
                    current_frame = (current_frame - 1) % len(self.frame_files)
                elif key == ' ':
                    auto_play = not auto_play
                elif key == '+' or key == '=':
                    fps = min(30, fps + 1)
                elif key == '-':
                    fps = max(1, fps - 1)
        
        except KeyboardInterrupt:
            pass
        
        print("\n\nPlayback stopped")
    
    def show_frame(self, frame_index: int):
        """
        Display a single frame.
        
        Args:
            frame_index: Index of frame to display
        """
        if frame_index < 0 or frame_index >= len(self.frame_files):
            print(f"Frame index {frame_index} out of range (0-{len(self.frame_files)-1})")
            return
        
        frame_content = self._load_frame(frame_index)
        print(f"Frame: {frame_index + 1}/{len(self.frame_files)}")
        print("-" * 60)
        print(frame_content)
    
    def __len__(self) -> int:
        """Return number of frames."""
        return len(self.frame_files)
    
    def __getitem__(self, index: int) -> str:
        """Get frame content by index."""
        return self._load_frame(index)


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Play through generated ASCII frames in the terminal')
    parser.add_argument('dataset_path', type=str,
                        help='Path to the dataset output folder')
    parser.add_argument('--fps', '-f', type=int, default=24,
                        help='Frames per second (default: 10)')
    parser.add_argument('--no-loop', action='store_true',
                        help='Do not loop playback')
    parser.add_argument('--start', type=int, default=0,
                        help='Starting frame index (default: 0)')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Interactive mode with manual controls')
    parser.add_argument('--frame', type=int, default=None,
                        help='Show a single frame and exit')
    
    args = parser.parse_args()
    
    # Create player
    player = AsciiDatasetPlayer(args.dataset_path)
    
    # Show single frame if requested
    if args.frame is not None:
        player.show_frame(args.frame)
        return
    
    # Play
    if args.interactive:
        player.play_interactive(start_frame=args.start)
    else:
        player.play(
            fps=args.fps,
            loop=not args.no_loop,
            start_frame=args.start
        )


if __name__ == '__main__':
    main()