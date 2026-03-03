import cv2
import torch
import numpy as np
import time
from ultralytics import RTDETR
from torchvision import transforms

# --- CONFIGURATION ---
# 1. Path to your input video (Updated for your .ts file)
INPUT_VIDEO = r"D:\Rail-Vision-Root\data\test_video_generated.mp4"

# 2. Where to save the result
OUTPUT_VIDEO = r"D:\Rail-Vision-Root\runs\output_video.mp4"

# 3. Path to MASUM'S Model (Defect Detection)
# NOTE: To use TensorRT later, change this to ending in .engine
# UPDATE THIS LINE to point to your new trained model:
MASUM_MODEL_PATH = r"D:\Rail-Vision-Root\scripts\runs\detect\rail_vision_transformer_large6\weights\best.pt"
# 4. Path to KAVYA'S Model (Restoration)
KAVYA_MODEL_PATH = r"D:\Rail-Vision-Root\models\kavya_deblur_real.pth"

# --- KAVYA'S MODEL ARCHITECTURE (Must match training!) ---
class SimpleUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = torch.nn.Sequential(torch.nn.Conv2d(3, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(64, 64, 3, padding=1), torch.nn.ReLU())
        self.pool = torch.nn.MaxPool2d(2)
        self.e2 = torch.nn.Sequential(torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(128, 128, 3, padding=1), torch.nn.ReLU())
        self.b = torch.nn.Sequential(torch.nn.Conv2d(128, 256, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(256, 128, 3, padding=1), torch.nn.ReLU())
        self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d1 = torch.nn.Sequential(torch.nn.Conv2d(128+128, 128, 3, padding=1), torch.nn.ReLU())
        self.d2 = torch.nn.Sequential(torch.nn.Conv2d(128+64, 64, 3, padding=1), torch.nn.ReLU())
        self.out = torch.nn.Conv2d(64, 3, 1)

    def forward(self, x):
        x1 = self.e1(x)
        p1 = self.pool(x1)
        x2 = self.e2(p1)
        p2 = self.pool(x2)
        b = self.b(p2)
        u1 = self.up(b)
        u1 = torch.cat([u1, x2], dim=1)
        d1 = self.d1(u1)
        u2 = self.up(d1)
        u2 = torch.cat([u2, x1], dim=1)
        d2 = self.d2(u2)
        return torch.sigmoid(self.out(d2))

# --- MAIN EXECUTION ---
def main():
    print(f"🚀 Initializing Video Pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {device}")

    # 1. Load Masum's Model
    try:
        print(f"   ► Loading Defect Detector from: {MASUM_MODEL_PATH}")
        detector = RTDETR(MASUM_MODEL_PATH)
    except Exception as e:
        print(f"❌ Error loading Masum's model: {e}")
        return

    # 2. Load Kavya's Model
    try:
        print(f"   ► Loading Restoration Model from: {KAVYA_MODEL_PATH}")
        restorer = SimpleUNet().to(device)
        restorer.load_state_dict(torch.load(KAVYA_MODEL_PATH, map_location=device))
        restorer.eval()
    except Exception as e:
        print(f"❌ Error loading Kavya's model: {e}")
        return

    # 3. Setup Video Capture
    print(f"   ► Opening Video: {INPUT_VIDEO}")
    cap = cv2.VideoCapture(INPUT_VIDEO)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {INPUT_VIDEO}")
        print("   -> Tip: Check if the file name or path is exactly correct.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
    
    print(f"🎥 Processing Started: {width}x{height} @ {fps:.2f} FPS")
    print("   Press 'Q' to stop early.")

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break # End of video
            
        # --- STAGE 1: RESTORATION (Kavya) ---
        # Resize to 256x256 for the UNet
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (256, 256))
        
        # Normalize & Tensor
        img_t = torch.from_numpy(img_resized / 255.0).permute(2, 0, 1).float().unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            restored_t = restorer(img_t)
            
        # Convert back to Full Size Image
        restored_np = restored_t.squeeze().cpu().permute(1, 2, 0).numpy() * 255
        restored_np = restored_np.astype(np.uint8)
        restored_full = cv2.resize(restored_np, (width, height))
        restored_bgr = cv2.cvtColor(restored_full, cv2.COLOR_RGB2BGR)

        # --- STAGE 2: DETECTION (Masum) ---
        # Run detection on the CLEAN image
        results = detector(restored_bgr, verbose=False)
        
        # --- STAGE 3: VISUALIZATION ---
        # Draw boxes
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{detector.names[cls]} {conf:.2f}"
                
                # Draw Red Box & Label
                cv2.rectangle(restored_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(restored_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Calculate Real-Time FPS
        process_fps = 1.0 / (time.time() - start_time)
        cv2.putText(restored_bgr, f"System FPS: {process_fps:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show & Save
        cv2.imshow("RailVision AI Output", restored_bgr)
        out.write(restored_bgr)

        # Exit on 'Q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Done! Output saved to: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()