# =============================================================================
#  ATTENTION U-NET – SAME STYLE AS YOUR U-NET CODE
# =============================================================================

import os, random, numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

# ================= GPU =================
DEVICE = torch.device("cuda")
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

# ================= CONFIG =================
IMAGE_DIR = "/content/drive/MyDrive/LAST DATASET/RAWIMG"
MASK_DIR  = "/content/drive/MyDrive/LAST DATASET/BINIMG"
SAVE_PATH = "/content/drive/MyDrive/attention_unet_best.pth"

IMG_SIZE = 256
BATCH_SIZE = 16
NUM_EPOCHS = 100
LR = 1e-4
VAL_SPLIT = 0.2
THRESHOLD = 0.5

# ================= DATASET =================
class VertebraeDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.images[idx]).convert("L"), dtype=np.float32)/255.0
        msk = (np.array(Image.open(self.masks[idx]).convert("L")) > 127).astype(np.float32)

        if self.transform:
            aug = self.transform(image=img, mask=msk)
            img = aug["image"]
            msk = aug["mask"].unsqueeze(0)   # FIX
        else:
            img = torch.tensor(img).unsqueeze(0)
            msk = torch.tensor(msk).unsqueeze(0)

        return img.float(), msk.float()

# ================= TRANSFORMS =================
def get_train_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        ToTensorV2()
    ], is_check_shapes=False)

def get_val_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        ToTensorV2()
    ], is_check_shapes=False)

# ================= MODEL =================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, 1)
        self.W_x = nn.Conv2d(F_l, F_int, 1)
        self.psi = nn.Sequential(nn.Conv2d(F_int,1,1), nn.Sigmoid())
        self.relu = nn.ReLU()

    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.e1 = ConvBlock(1,64)
        self.e2 = ConvBlock(64,128)
        self.e3 = ConvBlock(128,256)
        self.e4 = ConvBlock(256,512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(512,1024)

        self.up4 = nn.ConvTranspose2d(1024,512,2,2)
        self.att4 = AttentionBlock(512,512,256)
        self.d4 = ConvBlock(1024,512)

        self.up3 = nn.ConvTranspose2d(512,256,2,2)
        self.att3 = AttentionBlock(256,256,128)
        self.d3 = ConvBlock(512,256)

        self.up2 = nn.ConvTranspose2d(256,128,2,2)
        self.att2 = AttentionBlock(128,128,64)
        self.d2 = ConvBlock(256,128)

        self.up1 = nn.ConvTranspose2d(128,64,2,2)
        self.att1 = AttentionBlock(64,64,32)
        self.d1 = ConvBlock(128,64)

        self.out = nn.Conv2d(64,1,1)

    def forward(self,x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        e4 = self.att4(d4,e4)
        d4 = self.d4(torch.cat([d4,e4],1))

        d3 = self.up3(d4)
        e3 = self.att3(d3,e3)
        d3 = self.d3(torch.cat([d3,e3],1))

        d2 = self.up2(d3)
        e2 = self.att2(d2,e2)
        d2 = self.d2(torch.cat([d2,e2],1))

        d1 = self.up1(d2)
        e1 = self.att1(d1,e1)
        d1 = self.d1(torch.cat([d1,e1],1))

        return self.out(d1)

# ================= LOSS =================
class DiceLoss(nn.Module):
    def forward(self, logits, target):
        pred = torch.sigmoid(logits)
        inter = (pred * target).sum()
        return 1 - (2*inter+1)/(pred.sum()+target.sum()+1)

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    def forward(self, logits, target):
        return 0.5*self.bce(logits,target)+0.5*self.dice(logits,target)

# ================= METRICS =================
def dice_score(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * target).sum()
    return ((2*inter+1)/(pred.sum()+target.sum()+1)).item()

# ================= TRAIN =================
def train():
    imgs = sorted(Path(IMAGE_DIR).glob("*"))
    masks = sorted(Path(MASK_DIR).glob("*"))

    train_i, val_i, train_m, val_m = train_test_split(imgs, masks, test_size=VAL_SPLIT)

    train_ds = VertebraeDataset(train_i, train_m, get_train_transform())
    val_ds   = VertebraeDataset(val_i, val_m, get_val_transform())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = AttentionUNet().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5)
    scaler = GradScaler()
    criterion = CombinedLoss()

    best_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS+1):

        # TRAIN
        model.train()
        train_loss = 0
        for x,y in train_loader:
            x,y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            with autocast("cuda"):
                pred = model(x)
                loss = criterion(pred,y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # VALIDATE
        model.eval()
        val_loss = val_dice = 0

        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                loss = criterion(pred,y)

                val_loss += loss.item()
                val_dice += dice_score(pred.cpu(), y.cpu())

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        star = "★" if val_loss < best_loss else " "
        print(f"Epoch [{epoch:03d}/{NUM_EPOCHS}] {star}  "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Dice: {val_dice:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH)

# ================= RUN =================
if __name__ == "__main__":
    train()\