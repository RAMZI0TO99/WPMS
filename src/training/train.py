import argparse
import os
import yaml
from ultralytics import YOLO

def train_model(data_yaml, epochs=50, imgsz=640):
    if not os.path.exists(data_yaml):
        print(f"Error: Could not find {data_yaml}")
        print("Please make sure you exported the dataset from Roboflow in 'YOLOv8' format and unzipped it.")
        return
        
    # Make paths truly dynamic by injecting the absolute path at runtime
    data_yaml_abs = os.path.abspath(data_yaml)
    dataset_dir = os.path.dirname(data_yaml_abs)
    
    with open(data_yaml_abs, 'r') as f:
        yaml_data = yaml.safe_load(f)
        
    yaml_data['path'] = dataset_dir
    yaml_data['train'] = 'train/images'
    yaml_data['val'] = 'valid/images'
    yaml_data['test'] = 'test/images'
    
    with open(data_yaml_abs, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)
        
    print(f"Starting YOLOv8 training on {data_yaml_abs} for {epochs} epochs...")
    
    # Load a pretrained model (highly recommended for fine-tuning)
    model = YOLO('yolov8n.pt')
    
    # Train the model. 
    # workers=0 fixes Windows multiprocessing pagefile crashes.
    # batch=8 prevents 4GB VRAM out-of-memory errors on RTX 2050.
    results = model.train(data=data_yaml_abs, epochs=epochs, imgsz=imgsz, device=0, workers=0, batch=8)
    
    print("\nTraining completed!")
    try:
        save_dir = results.save_dir
        print(f"Your best model weights are saved at: {save_dir}/weights/best.pt")
        print(f"You can now run the monitor with: uv run python src/monitor.py --model {save_dir}/weights/best.pt")
    except AttributeError:
        print("Model weights are saved in the 'runs/detect/train/weights' directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom YOLOv8 model for Vest Detection")
    parser.add_argument("--data", type=str, required=True, help="Path to the data.yaml file downloaded from Roboflow")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    
    args = parser.parse_args()
    train_model(args.data, args.epochs, args.imgsz)
