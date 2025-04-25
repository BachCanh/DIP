from ultralytics import YOLO
import multiprocessing
import os
import time
import torch

def main():
    # Check for GPU availability
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Set output directory for results
    output_dir = os.path.join("runs", f"train_cars_{time.strftime('%Y%m%d_%H%M%S')}")
    
    # Load a pretrained YOLOv8n model
    model = YOLO("yolov8n.pt")
    print(f"Model loaded: YOLOv8n")

    # Set dataset path
    dataset_config = "./datasets/cars3.0/data.yaml"
    if not os.path.exists(dataset_config):
        print(f"WARNING: Dataset config not found at {dataset_config}")
    
    # Train the model on your custom dataset
    print(f"Starting training for 50 epochs...")
    train_results = model.train(
        data=dataset_config,
        epochs=50,              
        imgsz=640,               
        device=device,           
        batch=16,                
        patience=20,             
        project=output_dir,      # Save results to custom directory
        name="model",            # Name of the run
        exist_ok=True,           # Overwrite existing run
        pretrained=True,         # Use pretrained weights
        verbose=True             # Print verbose output
    )
    
    print(f"Training completed. Results saved to: {output_dir}")

    # Evaluate model performance on validation set
    print("Evaluating model performance...")
    metrics = model.val()
    print(f"Validation metrics: mAP@0.5 = {metrics.box.map50:.4f}, mAP@0.5:0.95 = {metrics.box.map:.4f}")

    # Export the trained model to ONNX format
    print("Exporting model to ONNX format...")
    exported_path = model.export(format="onnx", imgsz=640)
    print(f"Model exported to: {exported_path}")

if __name__ == "__main__":
    # This is crucial for Windows multiprocessing
    multiprocessing.freeze_support()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time/60:.2f} minutes")