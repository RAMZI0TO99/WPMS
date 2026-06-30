import argparse
from ultralytics import YOLO

def export_model(model_path):
    print(f"Loading model {model_path}...")
    model = YOLO(model_path)
    
    print("Exporting to TensorRT Engine format. This may take a few minutes...")
    print("NOTE: This requires NVIDIA CUDA and TensorRT libraries to be installed.")
    
    # Export the model to engine format with FP16 precision for faster inference
    model.export(format='engine', half=True)
    
    print("\nExport complete! You can now pass the .engine file to monitor.py instead of the .pt file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO model to TensorRT Engine")
    parser.add_argument("--model", type=str, required=True, help="Path to the .pt model weights")
    args = parser.parse_args()
    
    export_model(args.model)
