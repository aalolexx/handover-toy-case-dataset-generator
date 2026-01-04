# Surgery Action Dataset Generator

A synthetic dataset generator for training YOLO-style action detection models on medical instrument handover scenarios.

## Overview

This tool generates image sequences simulating a surgery room from a top-down perspective, where doctors and assistants perform instrument handovers. Each frame is annotated in YOLO format with bounding boxes for various actions.

## Scene Description

- **Patient Table**: Purple rectangle in the center where the doctor works
- **Preparation Table**: Pink rectangle below the patient table where instruments are prepared
- **Doctors**: Blue circles that work at the patient table
- **Assistants**: Green circles that prepare and transport instruments
- **Instruments**: Gray triangles that can be held, used, or transferred
- **Scene Objects**: Gray rectangles of various sizes at the scene edges

## Action Labels

| Class ID | Label | Description |
|----------|-------|-------------|
| 0 | assistant_prepares | Assistant at prep table handling instruments |
| 1 | doctor_works | Doctor actively using instrument (spinning) |
| 2 | person_holds | Person holding an instrument |
| 3 | assistant_gives | Assistant handing instrument to doctor |
| 4 | assistant_receives | Assistant receiving instrument from doctor |
| 5 | handover | Combined box during any handover action |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python dataset_generator.py
```

### With Configuration File

```bash
python dataset_generator.py --config config.yaml
```

### Create Default Config

```bash
python dataset_generator.py --create-config
```

### Command-Line Options

```
--config, -c     Path to configuration YAML file
--output, -o     Output directory (default: 'output')
--frames, -f     Number of frames to generate
--seed, -s       Random seed for reproducibility
--create-config  Create a default config file and exit
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| num_doctors | 2 | Total doctors in scene |
| num_assistants | 3 | Total assistants in scene |
| num_active_doctors | 1 | Doctors participating in handovers |
| num_active_assistants | 2 | Assistants participating in handovers |
| total_num_frames | 1000 | Total frames to generate |
| num_seperated_videos | 4 | Seperate Sequences with new scene setup |
| handover_rate | 0.02 | Probability of handover initiation |
| img_size | 448 | Image dimensions (square) |
| movement_speed | 3.0 | Person movement speed per frame |
| max_transition_pause | 100 | Max pause between state transitions |
| seed | 42 | Random seed for reproducibility |
|occlusion_obj_appearance_prob | 0.01 | |
|occlusion_obj_max_num | 3 | |
|occlusion_obj_max_size | 0.5 | As a fraction of img_size|

instrument_hidden_prob: 0.01


See `config.yaml` for all available options.

## Output Structure

```
output/
├── images/
│   ├── train/
│   │   ├── frame_000000.jpg
│   │   └── ...
│   └── val/
│       ├── frame_000400.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── frame_000000.txt
│   │   └── ...
│   └── val/
│       ├── frame_000400.txt
│       └── ...
├── classes.txt
├── data.yaml
└── config.yaml
```

## Annotation Format

Annotations use YOLO format:
```
class_id x_center y_center width height
```

All values are normalized to [0, 1] relative to image size.

Example annotation file:
```
2 0.523438 0.310547 0.089286 0.089286
5 0.481641 0.429688 0.187500 0.142578
```

## State Machine

### Assistant States
```
IDLE → MOVING_TO_PREP_TABLE → PREPARING → HOLDING → MOVING_TO_DOCTOR → GIVING
                                    ↑                                      ↓
                              RECEIVING ← ← ← ← ← ← ← ← ← ← MOVING_FROM_DOCTOR
```

### Doctor States
```
IDLE ↔ HOLDING ↔ WORKING
         ↕
   GIVING/RECEIVING
```

## Integration with YOLO

The generated `data.yaml` can be directly used with Ultralytics YOLO:

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(data='output/data.yaml', epochs=100)
```

## Dataset Player

The `DatasetPlayer` class allows you to visualize generated datasets with optional bounding box overlay.

### Command-Line Usage

```bash
# Play dataset with bounding boxes
python dataset_player.py output/

# Play validation split
python dataset_player.py output/ --split val

# Play without bounding boxes
python dataset_player.py output/ --no-boxes

# Export to video
python dataset_player.py output/ --export output_video.mp4

# Show statistics
python dataset_player.py output/ --stats

# Adjust playback speed and scale
python dataset_player.py output/ --fps 15 --scale 1.5
```

### Playback Controls (OpenCV mode)

| Key | Action |
|-----|--------|
| SPACE | Pause/Resume |
| LEFT/RIGHT | Previous/Next frame (when paused) |
| B | Toggle bounding boxes |
| L | Toggle labels |
| +/- | Increase/Decrease speed |
| Q/ESC | Quit |

### Python API

```python
from dataset_player import DatasetPlayer

# Create player
player = DatasetPlayer('output/', split='train')

# Get statistics
stats = player.get_statistics()
print(stats)

# Get a single frame with annotations
frame = player.get_frame(100, show_boxes=True)
frame.show()

# Play interactively (requires OpenCV)
player.play_cv2(fps=30, show_boxes=True)

# Export to video
player.export_video('output.mp4', fps=30, show_boxes=True)

# Iterate through frames
for image, annotations in player:
    # Process frame...
    pass
```

## Reproducibility

The dataset generation is fully reproducible when using the same seed. The config file is saved in the output directory for reference.

## License

MIT License
