"""
DatasetPlayer - A class for playing through generated dataset images with optional bounding box visualization.
"""
import os
import time
import argparse
from typing import List, Tuple, Optional, Dict
from pathlib import Path

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from enums import ActionLabel, LABEL_NAMES


# Colors for each action class (BGR for OpenCV, RGB for PIL)
CLASS_COLORS = {
    ActionLabel.ASSISTANT_PREPARES.value: (255, 165, 0),    # Orange
    ActionLabel.DOCTOR_WORKS.value: (255, 0, 0),            # Red
    ActionLabel.PERSON_HOLDS.value: (0, 255, 0),            # Green
    ActionLabel.ASSISTANT_GIVES.value: (0, 255, 255),       # Cyan
    ActionLabel.ASSISTANT_RECEIVES.value: (255, 0, 255),    # Magenta
    ActionLabel.HANDOVER.value: (255, 255, 0),              # Yellow
}


class DatasetPlayer:
    """
    Play through dataset images with optional bounding box visualization.
    
    Supports both OpenCV (cv2) and PIL-based playback.
    """
    
    def __init__(self, dataset_path: str, split: str = 'train'):
        """
        Initialize the dataset player.
        
        Args:
            dataset_path: Path to the dataset output folder
            split: Which split to play ('train' or 'val')
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        
        self.images_dir = self.dataset_path / 'images' / split
        self.labels_dir = self.dataset_path / 'labels' / split
        
        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found: {self.images_dir}")
        
        # Get sorted list of image files
        self.image_files = sorted([
            f for f in self.images_dir.iterdir() 
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ])
        
        if not self.image_files:
            raise ValueError(f"No images found in {self.images_dir}")
        
        # Load class names
        self.class_names = self._load_class_names()
        
        print(f"Loaded {len(self.image_files)} images from {split} split")
    
    def _load_class_names(self) -> Dict[int, str]:
        """Load class names from classes.txt or use defaults."""
        classes_file = self.dataset_path / 'classes.txt'
        
        if classes_file.exists():
            with open(classes_file, 'r') as f:
                names = [line.strip() for line in f if line.strip()]
            return {i: name for i, name in enumerate(names)}
        else:
            # Use default names from enums
            return {label.value: LABEL_NAMES[label] for label in ActionLabel}
    
    def _load_annotations(self, image_path: Path) -> List[Tuple[int, float, float, float, float]]:
        """
        Load YOLO format annotations for an image.
        
        Returns:
            List of (class_id, x_center, y_center, width, height) tuples (normalized)
        """
        label_file = self.labels_dir / (image_path.stem + '.txt')
        
        if not label_file.exists():
            return []
        
        annotations = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    annotations.append((class_id, x_center, y_center, width, height))
        
        return annotations
    
    def _draw_boxes_pil(self, image: Image.Image, annotations: List[Tuple], 
                        show_labels: bool = True) -> Image.Image:
        """Draw bounding boxes on image using PIL."""
        draw = ImageDraw.Draw(image)
        img_width, img_height = image.size
        
        # Try to load a font, fall back to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        for class_id, x_center, y_center, width, height in annotations:
            # Convert normalized coordinates to pixel coordinates
            x1 = int((x_center - width / 2) * img_width)
            y1 = int((y_center - height / 2) * img_height)
            x2 = int((x_center + width / 2) * img_width)
            y2 = int((y_center + height / 2) * img_height)
            
            # Get color for this class
            color = CLASS_COLORS.get(class_id, (255, 255, 255))
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # Draw label
            if show_labels:
                label = self.class_names.get(class_id, f"class_{class_id}")
                
                # Draw label background
                text_bbox = draw.textbbox((x1, y1 - 15), label, font=font)
                draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 2, 
                               text_bbox[2] + 2, text_bbox[3] + 2], 
                              fill=color)
                draw.text((x1, y1 - 15), label, fill=(0, 0, 0), font=font)
        
        return image
    
    def _draw_boxes_cv2(self, image: np.ndarray, annotations: List[Tuple],
                        show_labels: bool = True) -> np.ndarray:
        """Draw bounding boxes on image using OpenCV."""
        img_height, img_width = image.shape[:2]
        
        for class_id, x_center, y_center, width, height in annotations:
            # Convert normalized coordinates to pixel coordinates
            x1 = int((x_center - width / 2) * img_width)
            y1 = int((y_center - height / 2) * img_height)
            x2 = int((x_center + width / 2) * img_width)
            y2 = int((y_center + height / 2) * img_height)
            
            # Get color for this class (BGR for OpenCV)
            color_rgb = CLASS_COLORS.get(class_id, (255, 255, 255))
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color_bgr, 2)
            
            # Draw label
            if show_labels:
                label = self.class_names.get(class_id, f"class_{class_id}")
                
                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Draw label background
                cv2.rectangle(image, (x1, y1 - text_height - 5), 
                             (x1 + text_width, y1), color_bgr, -1)
                
                # Draw text
                cv2.putText(image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return image
    
    def play_cv2(self, fps: int = 30, show_boxes: bool = True, 
                 show_labels: bool = True, loop: bool = True,
                 start_frame: int = 0, scale: float = 1.0):
        """
        Play dataset using OpenCV window.
        
        Args:
            fps: Frames per second
            show_boxes: Whether to show bounding boxes
            show_labels: Whether to show class labels on boxes
            loop: Whether to loop playback
            start_frame: Starting frame index
            scale: Scale factor for display window
        
        Controls:
            SPACE: Pause/Resume
            LEFT/RIGHT: Previous/Next frame (when paused)
            B: Toggle bounding boxes
            L: Toggle labels
            Q/ESC: Quit
            +/-: Increase/Decrease speed
        """
        if not HAS_CV2:
            print("OpenCV (cv2) not available. Install with: pip install opencv-python")
            print("Falling back to PIL-based playback...")
            self.play_pil(fps, show_boxes, show_labels, loop, start_frame)
            return
        
        frame_delay = int(1000 / fps)
        current_frame = start_frame
        paused = False
        
        print("\nControls:")
        print("  SPACE: Pause/Resume")
        print("  LEFT/RIGHT: Previous/Next frame (when paused)")
        print("  B: Toggle bounding boxes")
        print("  L: Toggle labels")
        print("  +/-: Increase/Decrease speed")
        print("  Q/ESC: Quit")
        print()
        
        cv2.namedWindow('Dataset Player', cv2.WINDOW_AUTOSIZE)
        
        while True:
            # Load and process frame
            image_path = self.image_files[current_frame]
            image = cv2.imread(str(image_path))
            
            if image is None:
                print(f"Failed to load image: {image_path}")
                current_frame = (current_frame + 1) % len(self.image_files)
                continue
            
            # Draw bounding boxes if enabled
            if show_boxes:
                annotations = self._load_annotations(image_path)
                image = self._draw_boxes_cv2(image, annotations, show_labels)
            
            # Draw frame info
            info_text = f"Frame: {current_frame + 1}/{len(self.image_files)}"
            if paused:
                info_text += " [PAUSED]"
            if show_boxes:
                annotations = self._load_annotations(image_path)
                info_text += f" | Annotations: {len(annotations)}"
            
            cv2.putText(image, info_text, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(image, info_text, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            # Scale if needed
            if scale != 1.0:
                new_width = int(image.shape[1] * scale)
                new_height = int(image.shape[0] * scale)
                image = cv2.resize(image, (new_width, new_height))
            
            cv2.imshow('Dataset Player', image)
            
            # Handle input
            key = cv2.waitKey(frame_delay if not paused else 0) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord(' '):  # Space
                paused = not paused
            elif key == ord('b'):
                show_boxes = not show_boxes
                print(f"Bounding boxes: {'ON' if show_boxes else 'OFF'}")
            elif key == ord('l'):
                show_labels = not show_labels
                print(f"Labels: {'ON' if show_labels else 'OFF'}")
            elif key == ord('+') or key == ord('='):
                frame_delay = max(1, frame_delay - 10)
                print(f"Speed increased (delay: {frame_delay}ms)")
            elif key == ord('-'):
                frame_delay = min(1000, frame_delay + 10)
                print(f"Speed decreased (delay: {frame_delay}ms)")
            elif key == 81 or key == 2 or key == ord('p'):  # Left arrow
                if paused:
                    current_frame = (current_frame - 1) % len(self.image_files)
            elif key == 83 or key == 3 or key == ord('n'):  # Right arrow
                if paused:
                    current_frame = (current_frame + 1) % len(self.image_files)
            
            # Advance frame if not paused
            if not paused:
                current_frame += 1
                if current_frame >= len(self.image_files):
                    if loop:
                        current_frame = 0
                    else:
                        break
        
        cv2.destroyAllWindows()
    
    def play_pil(self, fps: int = 30, show_boxes: bool = True,
                 show_labels: bool = True, loop: bool = True,
                 start_frame: int = 0):
        """
        Play dataset using PIL (displays in terminal or saves frames).
        This is a simpler fallback when OpenCV is not available.
        
        Args:
            fps: Frames per second
            show_boxes: Whether to show bounding boxes
            show_labels: Whether to show class labels
            loop: Whether to loop playback
            start_frame: Starting frame index
        """
        frame_delay = 1.0 / fps
        current_frame = start_frame
        
        print("\nPIL-based playback (limited interactivity)")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                image_path = self.image_files[current_frame]
                image = Image.open(image_path)
                
                # Draw bounding boxes if enabled
                if show_boxes:
                    annotations = self._load_annotations(image_path)
                    image = self._draw_boxes_pil(image, annotations, show_labels)
                
                # Display frame info
                print(f"\rFrame: {current_frame + 1}/{len(self.image_files)}", end='', flush=True)
                
                # For PIL, we can show the image (opens default viewer)
                # This is blocking, so we just print info instead
                # image.show()  # Uncomment to open each frame in viewer
                
                time.sleep(frame_delay)
                
                current_frame += 1
                if current_frame >= len(self.image_files):
                    if loop:
                        current_frame = 0
                        print()  # New line after loop
                    else:
                        break
        
        except KeyboardInterrupt:
            print("\nPlayback stopped")
    
    def export_video(self, output_path: str, fps: int = 30, 
                     show_boxes: bool = True, show_labels: bool = True,
                     codec: str = 'mp4v'):
        """
        Export dataset as a video file.
        
        Args:
            output_path: Path for output video file
            fps: Frames per second
            show_boxes: Whether to show bounding boxes
            show_labels: Whether to show class labels
            codec: Video codec (e.g., 'mp4v', 'XVID', 'MJPG')
        """
        if not HAS_CV2:
            print("OpenCV (cv2) required for video export.")
            print("Install with: pip install opencv-python")
            return
        
        # Get frame size from first image
        first_image = cv2.imread(str(self.image_files[0]))
        height, width = first_image.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Exporting video to {output_path}...")
        
        for i, image_path in enumerate(self.image_files):
            image = cv2.imread(str(image_path))
            
            if show_boxes:
                annotations = self._load_annotations(image_path)
                image = self._draw_boxes_cv2(image, annotations, show_labels)
            
            # Add frame counter
            info_text = f"Frame: {i + 1}/{len(self.image_files)}"
            cv2.putText(image, info_text, (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(image, info_text, (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            out.write(image)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(self.image_files)} frames")
        
        out.release()
        print(f"Video exported successfully: {output_path}")
    
    def get_frame(self, frame_index: int, show_boxes: bool = True,
                  show_labels: bool = True) -> Image.Image:
        """
        Get a single frame with optional bounding boxes.
        
        Args:
            frame_index: Index of frame to get
            show_boxes: Whether to draw bounding boxes
            show_labels: Whether to show class labels
        
        Returns:
            PIL Image with optional annotations
        """
        if frame_index < 0 or frame_index >= len(self.image_files):
            raise IndexError(f"Frame index {frame_index} out of range (0-{len(self.image_files)-1})")
        
        image_path = self.image_files[frame_index]
        image = Image.open(image_path)
        
        if show_boxes:
            annotations = self._load_annotations(image_path)
            image = self._draw_boxes_pil(image, annotations, show_labels)
        
        return image
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        total_annotations = 0
        class_counts = {i: 0 for i in range(len(self.class_names))}
        frames_with_annotations = 0
        
        for image_path in self.image_files:
            annotations = self._load_annotations(image_path)
            if annotations:
                frames_with_annotations += 1
                total_annotations += len(annotations)
                for class_id, *_ in annotations:
                    if class_id in class_counts:
                        class_counts[class_id] += 1
        
        return {
            'total_frames': len(self.image_files),
            'frames_with_annotations': frames_with_annotations,
            'total_annotations': total_annotations,
            'class_counts': {self.class_names.get(k, f'class_{k}'): v 
                            for k, v in class_counts.items()},
            'split': self.split
        }
    
    def __len__(self) -> int:
        """Return number of frames in dataset."""
        return len(self.image_files)
    
    def __getitem__(self, index: int) -> Tuple[Image.Image, List[Tuple]]:
        """Get frame and annotations by index."""
        image_path = self.image_files[index]
        image = Image.open(image_path)
        annotations = self._load_annotations(image_path)
        return image, annotations


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Play through generated dataset images with optional bounding boxes')
    parser.add_argument('dataset_path', type=str,
                        help='Path to the dataset output folder')
    parser.add_argument('--split', '-s', type=str, default='train',
                        choices=['train', 'val'],
                        help='Which split to play (default: train)')
    parser.add_argument('--fps', '-f', type=int, default=30,
                        help='Frames per second (default: 30)')
    parser.add_argument('--no-boxes', action='store_true',
                        help='Hide bounding boxes')
    parser.add_argument('--no-labels', action='store_true',
                        help='Hide class labels on boxes')
    parser.add_argument('--no-loop', action='store_true',
                        help='Do not loop playback')
    parser.add_argument('--start', type=int, default=0,
                        help='Starting frame index (default: 0)')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Scale factor for display window (default: 1.0)')
    parser.add_argument('--export', '-e', type=str, default=None,
                        help='Export to video file instead of playing')
    parser.add_argument('--stats', action='store_true',
                        help='Show dataset statistics and exit')
    
    args = parser.parse_args()
    
    # Create player
    player = DatasetPlayer(args.dataset_path, args.split)
    
    # Show statistics if requested
    if args.stats:
        stats = player.get_statistics()
        print("\nDataset Statistics:")
        print("=" * 40)
        print(f"Split: {stats['split']}")
        print(f"Total frames: {stats['total_frames']}")
        print(f"Frames with annotations: {stats['frames_with_annotations']}")
        print(f"Total annotations: {stats['total_annotations']}")
        print("\nAnnotations by class:")
        for class_name, count in stats['class_counts'].items():
            print(f"  {class_name}: {count}")
        print("=" * 40)
        return
    
    # Export video if requested
    if args.export:
        player.export_video(
            args.export,
            fps=args.fps,
            show_boxes=not args.no_boxes,
            show_labels=not args.no_labels
        )
        return
    
    # Play dataset
    player.play_cv2(
        fps=args.fps,
        show_boxes=not args.no_boxes,
        show_labels=not args.no_labels,
        loop=not args.no_loop,
        start_frame=args.start,
        scale=args.scale
    )


if __name__ == '__main__':
    main()
