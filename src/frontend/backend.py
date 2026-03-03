import cv2
import torch
import numpy as np
import easyocr
from ultralytics import RTDETR

# --- CONFIGURATION ---
MASUM_MODEL_PATH = r"D:\Rail-Vision-Root\scripts\runs\detect\rail_vision_transformer_large6\weights\best.pt"
# UPDATE: Path to your newly retrained residual model
KAVYA_MODEL_PATH = r"D:\Rail-Vision-Root\models\kavya_deblur_real.pth"

_GLOBAL_OCR_READER = None

# --- UPDATED MODEL: RESIDUAL UNET ---
class SimpleUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.e1 = torch.nn.Sequential(torch.nn.Conv2d(3, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(64, 64, 3, padding=1), torch.nn.ReLU())
        self.pool = torch.nn.MaxPool2d(2)
        self.e2 = torch.nn.Sequential(torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(128, 128, 3, padding=1), torch.nn.ReLU())
        
        # Bridge
        self.b = torch.nn.Sequential(torch.nn.Conv2d(128, 256, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(256, 128, 3, padding=1), torch.nn.ReLU())
        
        # Decoder
        self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d1 = torch.nn.Sequential(torch.nn.Conv2d(256, 128, 3, padding=1), torch.nn.ReLU()) # Channels: 128 (up) + 128 (skip)
        self.d2 = torch.nn.Sequential(torch.nn.Conv2d(192, 64, 3, padding=1), torch.nn.ReLU())  # Channels: 128 (up) + 64 (skip)
        
        self.out = torch.nn.Conv2d(64, 3, 1)

    def forward(self, x):
        identity = x  # Store input for residual connection
        
        x1 = self.e1(x)
        p1 = self.pool(x1)
        x2 = self.e2(p1)
        p2 = self.pool(x2)
        b = self.b(p2)
        
        u1 = torch.cat([self.up(b), x2], dim=1) # Skip Connection 1
        d1 = self.d1(u1)
        u2 = torch.cat([self.up(d1), x1], dim=1) # Skip Connection 2
        d2 = self.d2(u2)
        
        # Identity mapping added to output for sharpening
        return torch.sigmoid(self.out(d2) + identity)

def apply_laplacian_sharpening(img):
    """Post-inference sharpening to highlight 4K-style details."""
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(img, -1, kernel)
    return cv2.addWeighted(img, 0.8, sharpened, 0.2, 0)

def get_models():
    global _GLOBAL_OCR_READER
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Detector (RT-DETR)
    detector = RTDETR(MASUM_MODEL_PATH)

    # 2. Load New Residual Restorer
    restorer = SimpleUNet().to(device)
    restorer.load_state_dict(torch.load(KAVYA_MODEL_PATH, map_location=device))
    restorer.eval() # Set to eval mode for inference
    
    # 3. Load OCR
    if _GLOBAL_OCR_READER is None:
        _GLOBAL_OCR_READER = easyocr.Reader(['en'], gpu=(device.type == 'cuda')) 
        
    return detector, restorer, device

def process_single_frame(frame, detector, restorer, device, wagon_id=1):
    global _GLOBAL_OCR_READER
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # --- 1. RESIDUAL RESTORATION ---
    img_resized = cv2.resize(img_rgb, (256, 256))
    img_t = torch.from_numpy(img_resized / 255.0).permute(2, 0, 1).float().unsqueeze(0).to(device)
    
    with torch.no_grad(): # Disable gradients for faster inference
        restored_t = restorer(img_t)
    
    restored_np = restored_t.squeeze().cpu().permute(1, 2, 0).numpy() * 255
    restored_full = cv2.resize(restored_np.astype(np.uint8), (w, h))

    # --- 2. LAPLACIAN SHARPENING ---
    restored_full = apply_laplacian_sharpening(restored_full)

    # --- 3. DETECTION & OCR ---
    final_img = restored_full.copy()
    defect_found = False
    
    if detector:
        results = detector(img_rgb, verbose=False, conf=0.40)
        for result in results:
            if len(result.boxes) > 0: defect_found = True
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(final_img, (x1, y1), (x2, y2), (0, 0, 255), 3)

    detected_text_str = ""
    if _GLOBAL_OCR_READER:
        ocr_results = _GLOBAL_OCR_READER.readtext(restored_full)
        for (bbox, text, prob) in ocr_results:
            if prob > 0.25: 
                detected_text_str = text
                tl = (int(bbox[0][0]), int(bbox[0][1]))
                br = (int(bbox[2][0]), int(bbox[2][1]))
                cv2.rectangle(final_img, tl, br, (0, 255, 0), 2)
                cv2.putText(final_img, text, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img_rgb, restored_full, final_img, defect_found, detected_text_str