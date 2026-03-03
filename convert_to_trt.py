from ultralytics import RTDETR
import torch

# 1. Path to your best PyTorch model
model_path = r"D:\Rail-Vision-Root\scripts\runs\detect\rail_vision_transformer_large6\weights\best.pt"

# 2. Load the model
print(f"Loading model from: {model_path}")
model = RTDETR(model_path)

# 3. Export to TensorRT (The Magic Step)
# half=True  -> Uses FP16 precision (Faster, less memory, same accuracy)
# dynamic=True -> Allows valid operation even if image size changes slightly
print("Starting TensorRT conversion... (This might take 5-10 minutes)")
try:
    model.export(
        format="engine",  # TensorRT format
        half=True,        # FP16 Quantization (The Speed Boost)
        dynamic=False,    # Static is slightly faster for fixed CCTV cameras
        imgsz=640,        # Standard size
        simplify=True,    # Cleans up the graph
        workspace=4       # Allocates 4GB RAM for building the engine
    )
    print("✅ Conversion Success! A new '.engine' file has been created next to your .pt file.")
except Exception as e:
    print(f"❌ Conversion Failed: {e}")
    print("Make sure you have 'tensorrt' installed: pip install tensorrt")