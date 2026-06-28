# Workplace Monitoring System

A computer vision monitoring system that uses **YOLOv8** and **ByteTrack** to track managers entering and exiting specific zones in a workplace. It maintains state to handle brief dropouts and logs all entry/exit events into a local SQLite database.

## Features
- **YOLOv8 Custom Object Detection**: Trained specifically to identify managers (and ignore standard workers using Class-Agnostic NMS).
- **ByteTrack Tracking**: Assigns consistent tracking IDs to individuals across frames.
- **State Machine**: Handles occlusions and temporary model dropouts with a customizable grace period.
- **Database Logging**: Automatically logs the exact timestamps of when a manager enters or exits the camera's view into an SQLite database (`workplace.db`).

## Setup

This project uses `uv` for lightning-fast dependency management.

1. Ensure you have Python installed.
2. Install `uv` if you haven't already.
3. Sync the environment and install dependencies:
   ```bash
   uv sync
   ```

## Usage

### 1. Running the Monitor
To start monitoring a video feed or a webcam, use the `monitor.py` script. You must provide a video source and the path to your trained model weights.

```bash
uv run python src/monitor.py --source videos/video1.mp4 --model runs/detect/train-4/weights/best.pt
```

**Arguments:**
- `--source`: Video source. Can be a path to an `.mp4` file or `0` for the default webcam.
- `--model`: Path to the `.pt` YOLO model weights.
- `--conf`: (Optional) Confidence threshold for detections (default: `0.6`).

### 2. Training a Custom Model
If you need to retrain the model on new data, place your Roboflow exported dataset in the project directory and run the training script.

```bash
uv run python src/training/train.py --data "dataset/data.yaml"
```

The script will automatically configure paths and fine-tune a YOLOv8 Nano model.

## File Structure
- `src/monitor.py`: The main script that runs the inference, tracking, and UI.
- `src/tracker_state.py`: The state machine that manages entry/exit logic and grace periods.
- `src/database.py`: Handles SQLite connection and logging.
- `src/training/train.py`: Utility script for training the YOLO model on custom datasets.
