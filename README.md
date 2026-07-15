# Workplace Monitoring System

A computer vision monitoring system that uses **YOLOv8** and **ByteTrack** to track managers entering and exiting specific zones in a workplace. It maintains state to handle brief dropouts and logs all entry/exit events into a local SQLite database.

## Features
- **YOLOv8 Custom Object Detection**: Trained specifically to identify managers and workers.
- **ByteTrack Tracking**: Assigns consistent tracking IDs to individuals across frames.
- **State Machine**: Handles occlusions and temporary model dropouts with a customizable grace period.
- **Database Logging**: Automatically logs the exact timestamps of when a manager enters or exits the camera's view into an SQLite database (`workplace.db`).
- **Streamlit Dashboard**: Live view of active sessions and historical logs.

## Setup

You can install dependencies using standard `pip` or using `uv` (recommended for faster installation).

### Option 1: Using pip (Standard)
1. Ensure you have Python 3.10+ installed.
2. Create and activate a virtual environment (optional but recommended).
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   > **Note**: The `requirements.txt` is configured to install PyTorch with CUDA support.

### Option 2: Using uv (Fast)
1. Ensure you have Python installed.
2. Install `uv` if you haven't already.
3. Sync the environment and install dependencies:
   ```bash
   uv sync
   ```

## Usage

### 1. Interactive Video Selection (Recommended)

The easiest way to start testing the system on available videos or your webcam is using the interactive launcher:

```bash
python main.py
```
*(Or `uv run main.py` if using uv)*

This will list all videos in the `videos/` folder and let you choose one via a simple text prompt.

### 2. Advanced: Running the Monitor Manually

The monitor defaults to the custom-trained model (`runs/detect/train-4/weights/best.pt`) which knows the `manager` and `worker` classes. You can run the monitor script directly with custom arguments:

```bash
# Use the custom model on a specific video
python src/monitor.py --source videos/video1.mp4

# Multiple cameras / video files simultaneously
python src/monitor.py --source videos/video1.mp4 videos/video2.mp4

# Webcam
python src/monitor.py --source 0

# Override the model explicitly
python src/monitor.py --source videos/video1.mp4 --model runs/detect/train-4/weights/best.pt
```

*(Prefix with `uv run` if you are using uv)*

**Arguments:**
- `--source`: Video source(s). Can be a path to an `.mp4` file or `0` for the default webcam.
- `--model`: Path to the `.pt` YOLO model weights (defaults to `best.pt` from `train-4`).
- `--conf`: (Optional) Confidence threshold for detections (default: `0.75`).

### 2. Running the Dashboard

```bash
uv run streamlit run src/dashboard.py
```

### 3. Training a Custom Model

If you need to retrain the model on new data, place your Roboflow exported dataset in the project directory and run the training script.

```bash
uv run python src/training/train.py --data "dataset/data.yaml"
```

The script will automatically configure paths and fine-tune a YOLOv8 Nano model.

## File Structure
- `src/monitor.py`: The main script that runs the inference, tracking, and UI.
- `src/tracker_state.py`: The state machine that manages entry/exit logic and grace periods.
- `src/database.py`: Handles SQLite connection and logging.
- `src/dashboard.py`: Streamlit dashboard for live and historical session data.
- `src/training/train.py`: Utility script for training the YOLO model on custom datasets.
- `runs/detect/train-4/weights/best.pt`: The best custom-trained model weights.
