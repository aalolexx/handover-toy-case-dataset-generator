"""
DatasetGenerator - Main class for generating the surgery action dataset.
"""
import os
import shutil
from typing import Optional
from tqdm import tqdm
import numpy as np

from config import Config
from process_manager import ProcessManager
from render_manager import RenderManager
from annotation_manager import AnnotationManager


class DatasetGenerator:
    """Generates synthetic surgery action detection dataset."""
    
    def __init__(self, config: Config):
        """
        Initialize the dataset generator.
        
        Args:
            config: Scene configuration
        """
        self.config = config
        self.process_manager = ProcessManager(config)
        self.render_manager = RenderManager(config)
        
        # Setup output directories
        self.output_dir = "outputs/" + config.output_dir
        self._setup_directories()
        
        self.annotation_manager = AnnotationManager(
            config.img_size, self.output_dir)
    
    def _setup_directories(self):
        """Create output directory structure."""
        # Clear existing output if present
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        
        # Create directories
        os.makedirs(os.path.join(self.output_dir, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'images', 'val'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'labels', 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'labels', 'val'), exist_ok=True)
    
    def generate(self, progress: bool = True):
        """
        Generate the complete dataset.
        
        Args:
            progress: Whether to show progress bar
        """
        if self.config.seed != -1:
            print(f"Initializing scene with seed {self.config.seed}...")
        self.process_manager.initialize_scene()
        
        print(f"Generating {self.config.total_num_frames} frames and {self.config.num_seperated_videos} videos/sequences...")
        
        # Determine train/val split
        train_frames = int(self.config.total_num_frames * 0.8)
        
        iterator = range(self.config.total_num_frames)
        if progress:
            iterator = tqdm(iterator, desc="Generating frames")

        video_idx = 0
        
        for frame_num in iterator:

            if frame_num % (self.config.total_num_frames // self.config.num_seperated_videos) == 0:
                self.process_manager = ProcessManager(self.config)
                self.process_manager.initialize_scene()
                video_idx += 1
            
            # Update scene
            self.process_manager.update()
            
            # Get current state
            state = self.process_manager.get_scene_state()
            
            # Render frame
            img = self.render_manager.render_frame(
                state['persons'],
                state['instruments'],
                state['patient_table'],
                state['preparation_table'],
                state['scene_objects'],
                state['occlusion_objects']
            )
            
            # Generate annotations
            annotations = self.annotation_manager.generate_annotations(
                frame_num, state['persons'])
            
            # Determine if train or val
            split = 'train' if frame_num < train_frames else 'val'
            
            # Save frame
            img_filename = f"video{video_idx:02}_frame_{frame_num:06d}.jpg"
            img_path = os.path.join(self.output_dir, 'images', split, img_filename)
            self.render_manager.save_frame(img, img_path)
            
            # Save annotations
            label_filename = f"video{video_idx:02}_frame_{frame_num:06d}.txt"
            label_path = os.path.join(self.output_dir, 'labels', split, label_filename)
            with open(label_path, 'w') as f:
                f.write('\n'.join(annotations))
                if annotations:
                    f.write('\n')
        
        # Save metadata files
        print("Saving metadata files...")
        self.annotation_manager.save_classes_file()
        self.annotation_manager.save_data_yaml()
        
        # Save config for reproducibility
        self.config.to_yaml(os.path.join(self.output_dir, 'config.yaml'))
        
        print(f"Dataset generated successfully in '{self.output_dir}'")
        self._print_summary()
    
    def _print_summary(self):
        """Print summary of generated dataset."""
        train_images = len(os.listdir(
            os.path.join(self.output_dir, 'images', 'train')))
        val_images = len(os.listdir(
            os.path.join(self.output_dir, 'images', 'val')))
        
        # Count annotations
        total_annotations = 0
        action_counts = {}
        
        for split in ['train', 'val']:
            label_dir = os.path.join(self.output_dir, 'labels', split)
            for filename in os.listdir(label_dir):
                filepath = os.path.join(label_dir, filename)
                with open(filepath, 'r') as f:
                    for line in f:
                        if line.strip():
                            total_annotations += 1
                            class_id = int(line.split()[0])
                            action_counts[class_id] = action_counts.get(class_id, 0) + 1
        
        print("\n" + "="*50)
        print("DATASET SUMMARY")
        print("="*50)
        print(f"Training images: {train_images}")
        print(f"Validation images: {val_images}")
        print(f"Total annotations: {total_annotations}")
        print("\nAnnotation counts by class:")
        
        from enums import ActionLabel, LABEL_NAMES
        for label in ActionLabel:
            count = action_counts.get(label.value, 0)
            print(f"  {LABEL_NAMES[label]}: {count}")
        
        print("="*50)


def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate synthetic surgery action detection dataset')
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Path to configuration YAML file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--frames', '-f', type=int, default=None,
                        help='Number of frames to generate (overrides config)')
    parser.add_argument('--seed', '-s', type=int, default=None,
                        help='Random seed (overrides config)')
    parser.add_argument('--create-config', action='store_true',
                        help='Create a default config file and exit')
    
    args = parser.parse_args()
    
    if args.create_config:
        from config import create_default_config
        config_path = args.config or 'config.yaml'
        create_default_config(config_path)
        print(f"Created default config at '{config_path}'")
        return
    
    # Load or create config
    if args.config and os.path.exists(args.config):
        config = Config.from_yaml(args.config)
    else:
        config = Config()
    
    # Override with command-line arguments
    if args.output:
        config.output_dir = args.output
    if args.frames:
        config.total_num_frames = args.frames
    if args.seed:
        config.seed = args.seed
    
    # Generate dataset
    generator = DatasetGenerator(config)
    generator.generate()


if __name__ == '__main__':
    main()
