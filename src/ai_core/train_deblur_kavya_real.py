import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from tqdm import tqdm

# --- CONFIGURATION ---
DATASET_ROOT = r"D:\Rail-Vision-Root\data\blurred_sharp\blurred_sharp"
BLUR_DIR = os.path.join(DATASET_ROOT, "blurred")
SHARP_DIR = os.path.join(DATASET_ROOT, "sharp")
MODEL_NAME = "kavya_deblur_res_v2"
SAVE_DIR = "runs/restoration_logs"
IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 100 

# --- NEW: SSIM LOSS (Ensures 4K Sharpness) ---
def ssim_loss(img1, img2, window_size=11):
    """Calculates Structural Similarity Loss to prevent blurring."""
    mu1 = nn.functional.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = nn.functional.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = nn.functional.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = nn.functional.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = nn.functional.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return 1 - ssim_map.mean()

# --- METRIC CALCULATOR ---
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 100.0 if mse == 0 else 20 * torch.log10(1.0 / torch.sqrt(mse))

# --- DATASET ---
class RealRestorationDataset(Dataset):
    def __init__(self, blur_dir, sharp_dir):
        self.blur_dir, self.sharp_dir = blur_dir, sharp_dir
        self.filenames = sorted([f for f in os.listdir(blur_dir) if os.path.exists(os.path.join(sharp_dir, f))])
        print(f"✅ Dataset Loaded: {len(self.filenames)} pairs.")

    def __len__(self): return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        blur = cv2.resize(cv2.imread(os.path.join(self.blur_dir, name)), (IMG_SIZE, IMG_SIZE))
        sharp = cv2.resize(cv2.imread(os.path.join(self.sharp_dir, name)), (IMG_SIZE, IMG_SIZE))
        return (torch.from_numpy(blur / 255.0).permute(2, 0, 1).float(), 
                torch.from_numpy(sharp / 255.0).permute(2, 0, 1).float())

# --- IMPROVED MODEL: RESIDUAL UNET ---
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.e2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.b = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 128, 3, padding=1), nn.ReLU())
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d1 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU())
        self.d2 = nn.Sequential(nn.Conv2d(192, 64, 3, padding=1), nn.ReLU())
        self.out = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        identity = x #
        x1 = self.e1(x)
        p1 = self.pool(x1)
        x2 = self.e2(p1)
        p2 = self.pool(x2)
        b = self.b(p2)
        u1 = torch.cat([self.up(b), x2], dim=1)
        d1 = self.d1(u1)
        u2 = torch.cat([self.up(d1), x1], dim=1)
        d2 = self.d2(u2)
        # Learn the residual to preserve sharpness
        return torch.sigmoid(self.out(d2) + identity)

# --- PREVIEW ---
def save_preview(model, dataset, device, epoch, psnr):
    model.eval()
    with torch.no_grad():
        inp, targ = dataset[0]
        pred = model(inp.unsqueeze(0).to(device)).squeeze().cpu().permute(1, 2, 0).numpy()
        img_in = (inp.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_out = (pred * 255).astype(np.uint8)
        img_gt = (targ.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        combined = np.hstack((img_in, img_out, img_gt))
        cv2.putText(combined, f"Epoch {epoch} | PSNR: {psnr:.2f}dB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        os.makedirs(SAVE_DIR, exist_ok=True)
        cv2.imwrite(f"{SAVE_DIR}/epoch_{epoch}.jpg", combined)

# --- MAIN ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RealRestorationDataset(BLUR_DIR, SHARP_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = SimpleUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005) # Lower LR for stability
    criterion_mse = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_psnr = 0
        loop = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for b_in, b_targ in loop:
            b_in, b_targ = b_in.to(device), b_targ.to(device)
            optimizer.zero_grad()
            output = model(b_in)
            
            # JOINT LOSS: MSE for pixels + SSIM for 4K edges
            loss = criterion_mse(output, b_targ) + 0.5 * ssim_loss(output, b_targ)
            
            loss.backward()
            optimizer.step()
            psnr = calculate_psnr(output, b_targ)
            total_psnr += psnr.item()
            loop.set_postfix(psnr=f"{psnr.item():.2f}dB")

        avg_psnr = total_psnr / len(loader)
        if epoch % 5 == 0: save_preview(model, dataset, device, epoch, avg_psnr)

    torch.save(model.state_dict(), f"models/{MODEL_NAME}.pth")
    print(f"✅ Retraining complete! Quality: {avg_psnr:.2f} dB")

if __name__ == "__main__": main()