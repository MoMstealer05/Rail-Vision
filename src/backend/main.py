import cv2
import torch
import numpy as np
import easyocr
from ultralytics import RTDETR

# --- CONFIGURATION ---
MASUM_MODEL_PATH = r"D:\Rail-Vision-Root\scripts\runs\detect\rail_vision_transformer_large6\weights\best.pt"
KAVYA_MODEL_PATH = r"D:\Rail-Vision-Root\models\kavya_deblur_real.pth"

# Global variable to hold the OCR model internally
# This prevents us from having to pass it around and break app.py
_GLOBAL_OCR_READER = None

# --- RESTORATION MODEL ---
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

def get_models():
    """
    Loads Detector, Restorer, and initializes OCR internally.
    Returns: detector, restorer, device (Tuple of 3, matches app.py expectation)
    """
    global _GLOBAL_OCR_READER
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector, restorer = None, None

    # 1. Load Object Detector
    try:
        print(f"Loading Detector: {MASUM_MODEL_PATH}")
        detector = RTDETR(MASUM_MODEL_PATH)
    except Exception as e:
        print(f"❌ Detector Error: {e}")

    # 2. Load Restoration Model
    try:
        print(f"Loading Restorer: {KAVYA_MODEL_PATH}")
        restorer = SimpleUNet().to(device)
        restorer.load_state_dict(torch.load(KAVYA_MODEL_PATH, map_location=device))
        restorer.eval()
    except Exception as e:
        print(f"❌ Restorer Error: {e}")
    
    # 3. Load OCR Model (Saved to Global Variable)
    if _GLOBAL_OCR_READER is None:
        try:
            print("Loading OCR Model (EasyOCR)...")
            _GLOBAL_OCR_READER = easyocr.Reader(['en'], gpu=(device.type == 'cuda')) 
        except Exception as e:
            print(f"❌ OCR Error: {e}")
        
    # Return exactly 3 values to satisfy app.py
    return detector, restorer, device

def process_single_frame(frame, detector, restorer, device, wagon_id=1):
    global _GLOBAL_OCR_READER
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # --- 1. RESTORATION ---
    img_resized = cv2.resize(img_rgb, (256, 256))
    img_t = torch.from_numpy(img_resized / 255.0).permute(2, 0, 1).float().unsqueeze(0).to(device)
    
    restored_full = img_rgb 
    if restorer:
        with torch.no_grad():
            restored_t = restorer(img_t)
        restored_np = restored_t.squeeze().cpu().permute(1, 2, 0).numpy() * 255
        restored_np = restored_np.astype(np.uint8)
        restored_full = cv2.resize(restored_np, (w, h))

    # --- 2. VISUALIZATION PREP ---
    final_img = restored_full.copy()
    
    # Red & Blue Box logic (Wagon & Defects)
    margin = int(h * 0.05)
    cv2.rectangle(final_img, (margin, margin), (w - margin, h - margin), (255, 0, 0), 3) 
    cv2.rectangle(final_img, (margin, margin), (margin + 280, margin + 40), (255, 0, 0), -1)
    cv2.putText(final_img, f"WAGON ID: #{wagon_id}", (margin + 10, margin + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    defect_found = False
    if detector:
        results = detector(img_rgb, verbose=False, conf=0.40, iou=0.5)
        for result in results:
            for box in result.boxes:
                defect_found = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(final_img, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # --- 3. OCR (Updated to return text) ---
    detected_text_str = ""  # Variable to hold text
    if _GLOBAL_OCR_READER:
        try:
            ocr_results = _GLOBAL_OCR_READER.readtext(restored_full)
            for (bbox, text, prob) in ocr_results:
                (tl, tr, br, bl) = bbox
                tl = (int(tl[0]), int(tl[1]))
                br = (int(br[0]), int(br[1]))

                if prob > 0.35: 
                    detected_text_str = text # Capture the text
                    cv2.rectangle(final_img, tl, br, (0, 255, 0), 2)
                    cv2.putText(final_img, text, (tl[0], tl[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        except Exception:
            pass

    # RETURN 5 VALUES NOW
    return img_rgb, restored_full, final_img, defect_found, detected_text_str
