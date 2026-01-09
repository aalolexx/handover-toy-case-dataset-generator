"""
AnnotationManager class for generating YOLO format annotations.
"""
from typing import List, Tuple, Optional
import os

from person import Person
from enums import ActionLabel, LABEL_NAMES, PersonType
from utils import get_combined_bounding_box, normalize_bbox


class AnnotationManager:
    """Handles generation of YOLO format annotation files."""
    
    def __init__(self, img_size: int, output_dir: str):
        """
        Initialize the annotation manager.
        
        Args:
            img_size: Size of the square images
            output_dir: Directory to save annotation files
        """
        self.img_size = img_size
        self.output_dir = output_dir
        self.labels_dir = os.path.join(output_dir, 'labels')
        os.makedirs(self.labels_dir, exist_ok=True)
    
    def generate_annotations(self, frame_num: int, 
                            persons: List[Person]) -> List[str]:
        """
        Generate YOLO annotations for a frame.
        
        Args:
            frame_num: Frame number (for filename)
            persons: List of all persons in the scene
        
        Returns:
            List of annotation lines in YOLO format
        """
        annotations = []
        
        # Track handover pairs to avoid duplicate handover boxes
        processed_handover_pairs = set()
        
        for person in persons:
            # Always annotate the generic person box
            person_bbox = person.get_bounding_box(self.img_size)
            person_line = self._format_annotation(ActionLabel.PERSON.value, person_bbox)
            annotations.append(person_line)

            if person.person_type == PersonType.DOCTOR:
                doctor_line = self._format_annotation(ActionLabel.DOCTOR.value, person_bbox)
                annotations.append(doctor_line)
            elif person.person_type == PersonType.ASSISTANT:
                assistant_line = self._format_annotation(ActionLabel.ASSISTANT.value, person_bbox)
                annotations.append(assistant_line)
            else:
                print("what the helly?")

            # Action specific annotations
            action = person.get_current_action()
            if action is None:
                continue
            
            # If it's a handover action, add additional handover box
            if person.is_in_handover() and person.handover_partner:
                pair_id = tuple(sorted([person.id, person.handover_partner.id]))
                
                if pair_id not in processed_handover_pairs:
                    processed_handover_pairs.add(pair_id)
                    
                    # Create combined bounding box for handover
                    combined_bbox = get_combined_bounding_box(
                        [person.position, person.handover_partner.position],
                        person.radius,
                        padding=0
                    )
                    normalized_bbox = normalize_bbox(combined_bbox, self.img_size)
                    
                    # Add handover annotation
                    handover_line = self._format_annotation(
                        ActionLabel.HANDOVER.value, normalized_bbox)
                    annotations.append(handover_line)

                    # Add the direction specific handover annotation too
                    line = self._format_annotation(action.value, normalized_bbox)
                    annotations.append(line)
            else:
                # Get bounding box for this person's action
                bbox = person.get_bounding_box(self.img_size)
                # Add the action annotation
                line = self._format_annotation(action.value, bbox)
                annotations.append(line)
        
        return annotations
    
    def _format_annotation(self, class_id: int,
                          bbox: Tuple[float, float, float, float]) -> str:
        """
        Format a single annotation in YOLO format.
        
        Args:
            class_id: Class ID (0-5)
            bbox: Normalized (x_center, y_center, width, height)
        
        Returns:
            YOLO format annotation string
        """
        x, y, w, h = bbox
        # Clamp values to [0, 1]
        x = max(0, min(1, x))
        y = max(0, min(1, y))
        w = max(0, min(1, w))
        h = max(0, min(1, h))
        
        return f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
    
    def save_annotations(self, frame_num: int, annotations: List[str]):
        """
        Save annotations to a file.
        
        Args:
            frame_num: Frame number
            annotations: List of YOLO format annotation strings
        """
        filename = f"frame_{frame_num:06d}.txt"
        filepath = os.path.join(self.labels_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(annotations))
            if annotations:
                f.write('\n')
    
    def save_classes_file(self):
        """Save the classes.txt file with class names."""
        filepath = os.path.join(self.output_dir, 'classes.txt')
        
        with open(filepath, 'w') as f:
            for label in ActionLabel:
                f.write(f"{LABEL_NAMES[label]}\n")
    
    def save_data_yaml(self, train_ratio: float = 0.8):
        """
        Save data.yaml file for YOLO training.
        
        Args:
            train_ratio: Ratio of training data
        """
        filepath = os.path.join(self.output_dir, 'data.yaml')
        
        content = f"""# Surgery Dataset for Action Detection
path: {os.path.abspath(self.output_dir)}
train: images/train
val: images/val

nc: {len(ActionLabel)}
names:
"""
        for label in ActionLabel:
            content += f"  {label.value}: {LABEL_NAMES[label]}\n"
        
        with open(filepath, 'w') as f:
            f.write(content)
