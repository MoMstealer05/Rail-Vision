import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import time
import math
from tqdm import tqdm  # Live progress bar

# --- CONFIGURATION ---
DATASET_ROOT = r"D:\Rail-Vision-Root\data\blurred_sharp\blurred_sharp"
BLUR_DIR = os.path.join(DATASET_ROOT, "blurred")
SHARP_DIR = os.path.join(DATASET_ROOT, "sharp")

MODEL_NAME = "kavya_deblur_real"
SAVE_DIR = "runs/restoration_logs"
IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 100 

# --- METRIC CALCULATOR (PSNR) ---
def calculate_psnr(img1, img2):
    """Calculates 'Accuracy' (PSNR) for image restoration."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# --- DATASET ---
class RealRestorationDataset(Dataset):
    def __init__(self, blur_dir, sharp_dir):
        self.blur_dir = blur_dir
        self.sharp_dir = sharp_dir
        self.filenames = sorted(os.listdir(blur_dir))
        
        # Verify valid pairs
        self.valid_files = []
        for f in self.filenames:
            if os.path.exists(os.path.join(sharp_dir, f)):
                self.valid_files.append(f)
        
        print(f"   ✅ Dataset Ready: Found {len(self.valid_files)} pairs.")

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        file_name = self.valid_files[idx]
        
        # Load Images
        blur_path = os.path.join(self.blur_dir, file_name)
        sharp_path = os.path.join(self.sharp_dir, file_name)
        
        blur_img = cv2.imread(blur_path)
        sharp_img = cv2.imread(sharp_path)
        
        if blur_img is None or sharp_img is None:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), torch.zeros(3, IMG_SIZE, IMG_SIZE)

        # Resize & Normalize
        blur_img = cv2.resize(blur_img, (IMG_SIZE, IMG_SIZE))
        sharp_img = cv2.resize(sharp_img, (IMG_SIZE, IMG_SIZE))
        
        input_t = torch.from_numpy(blur_img / 255.0).permute(2, 0, 1).float()
        target_t = torch.from_numpy(sharp_img / 255.0).permute(2, 0, 1).float()
        
        return input_t, target_t

# --- MODEL (UNET) ---
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.e2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.b = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 128, 3, padding=1), nn.ReLU())
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d1 = nn.Sequential(nn.Conv2d(128+128, 128, 3, padding=1), nn.ReLU())
        self.d2 = nn.Sequential(nn.Conv2d(128+64, 64, 3, padding=1), nn.ReLU())
        self.out = nn.Conv2d(64, 3, 1)

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

# --- PREVIEW GENERATOR ---
def save_preview(model, dataset, device, epoch, psnr_score):
    model.eval()
    with torch.no_grad():
        inp, targ = dataset[0] 
        inp_batch = inp.unsqueeze(0).to(device)
        pred = model(inp_batch)
        
        img_in = (inp.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_out = (pred.squeeze().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_gt = (targ.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        combined = np.hstack((img_in, img_out, img_gt))
        
        # Add PSNR score to the image itself so judges see it
        cv2.putText(combined, f"Epoch {epoch} | Quality: {psnr_score:.2f} dB", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(combined, "Input", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(combined, "Restored", (270, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        os.makedirs(SAVE_DIR, exist_ok=True)
        cv2.imwrite(f"{SAVE_DIR}/preview_epoch_{epoch}.jpg", combined)

# --- TRAINING LOOP ---
def main():
    print(f"\n{'='*60}")
    print(f"🚀 INITIALIZING KAVYA'S RESTORATION AI (Real Data Mode)")
    print(f"{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Hardware Accelerator: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    if not os.path.exists(BLUR_DIR):
        print("❌ ERROR: Data not found.")
        return

    dataset = RealRestorationDataset(BLUR_DIR, SHARP_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = SimpleUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss() 
    
    print(f"🎯 Target: Maximizing PSNR (Image Quality). >25dB is good.\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        total_psnr = 0
        
        # 🟢 LIVE PROGRESS BAR
        loop = tqdm(loader, leave=True)
        loop.set_description(f"Epoch [{epoch}/{EPOCHS}]")
        
        for batch_in, batch_target in loop:
            batch_in, batch_target = batch_in.to(device), batch_target.to(device)
            
            optimizer.zero_grad()
            output = model(batch_in)
            loss = criterion(output, batch_target)
            loss.backward()
            optimizer.step()
            
            # Compute Stats
            batch_psnr = calculate_psnr(output, batch_target)
            total_loss += loss.item()
            total_psnr += batch_psnr.item()
            
            # UPDATE BAR: Show Loss AND Accuracy (PSNR)
            loop.set_postfix(loss=loss.item(), quality_dB=batch_psnr.item())
        
        # Epoch Stats
        avg_psnr = total_psnr / len(loader)
        
        # Save preview with Score
        if epoch % 5 == 0 or epoch == 1:
            save_preview(model, dataset, device, epoch, avg_psnr)
            
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/{MODEL_NAME}.pth")
    print(f"\n✅ Training Complete. Final Quality: {avg_psnr:.2f} dB")

if __name__ == "__main__":
    main()
