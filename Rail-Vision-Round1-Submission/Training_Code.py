from ultralytics import RTDETR

def train_model():
    
    model = RTDETR("rtdetr-l.pt") 

    
    # NOTE: These settings match the submitted Hyperparameters file.
    results = model.train(
        data="data.yaml",   # Path to dataset config
        epochs=100,         # Fixed epoch count
        imgsz=640,          # Standard input size
        batch=16,           # Batch size for 8GB VRAM
        device=0,           # GPU index
        project="rail_vision_project",
        name="rtdetr_training_run",
        
        # --- LOCKED HYPERPARAMETERS ---
        lr0=0.001,          # Initial Learning Rate
        lrf=0.01,           # Final Learning Rate
        momentum=0.937,     # SGD Momentum
        weight_decay=0.0005,# Regularization
        warmup_epochs=3.0,  # Warmup phase
        box=7.5,            # Box Loss Gain
        cls=0.5,            # Class Loss Gain
        dfl=1.5,            # Distribution Focal Loss
        
        # --- AUGMENTATION SETTINGS ---
        hsv_h=0.015,        # HSV-Hue augmentation
        hsv_s=0.7,          # HSV-Saturation augmentation
        hsv_v=0.4,          # HSV-Value augmentation
        degrees=0.0,        # Rotation
        translate=0.1,      # Translation
        scale=0.5,          # Scaling
        mosaic=1.0,         # Mosaic (Strong augmentation)
        mixup=0.0,          # Mixup
    )

if __name__ == "__main__":
    train_model()