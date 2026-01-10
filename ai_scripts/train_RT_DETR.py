from ultralytics import RTDETR

# --- CONFIGURATION ---
DATA_YAML = r"D:\Rail-Vision-Root\data\wagon_dataset\data.yaml"
MODEL_NAME = "rail_vision_transformer_large"

def train_transformer():
    print("🚀 TRAINING RT-DETR (Transformer Based Detection)...")
    print("   ► Backbone: Vision Transformer (ViT)")
    print("   ► Advantage: Highest Accuracy for Small Defects")

    # Load the Large Transformer (Most accurate version)
    # 'rtdetr-l.pt' is heavy but powerful. 
    # If OOM Error, switch to 'rtdetr-x.pt' (Even bigger) or 'rtdetr-l.pt' with lower batch
    model = RTDETR("rtdetr-l.pt") 
    
    model.train(
        data=DATA_YAML,
        epochs=75,          # Transformers need time to converge
        imgsz=640,          # Standard size (Transformers are heavy on high res)
        device=0,           # GPU
        batch=4,            # Keep small to save VRAM
        name=MODEL_NAME,
        
        # --- TRANSFORMER SPECIFIC SETTINGS ---
        optimizer='AdamW',  # AdamW is mandatory for Transformers
        lr0=0.0001,         # Low learning rate
        warmup_epochs=5,    # Gentle start
        
        # --- AUGMENTATION ---
        mosaic=1.0,         # Helps see small objects
        hsv_h=0.015,        # Lighting robustness
        hsv_s=0.7, 
        hsv_v=0.4,
    )
    print(f"🎉 Training Complete! Model: runs/detect/{MODEL_NAME}/weights/best.pt")

if __name__ == "__main__":
    train_transformer()
