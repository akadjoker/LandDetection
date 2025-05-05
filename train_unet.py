
import os
import cv2
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# ==== CONFIG ====
IMAGE_DIR = 'images'
MASK_DIR = 'masks'
IMG_SIZE = (256,256)
BATCH_SIZE = 4
EPOCHS = 20
LR = 1e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==== DATASET ====
class LineDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, IMG_SIZE)
        mask = (mask > 5).astype(np.float32)

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return img, mask

# ==== MODELO ====
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(),
            )
        self.enc1 = conv_block(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.middle = conv_block(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        m = self.middle(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(m), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.final(d1))

# ==== TREINO ====
def train():
    img_paths = sorted(glob(os.path.join(IMAGE_DIR, '*')))
    mask_paths = sorted(glob(os.path.join(MASK_DIR, '*')))
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(img_paths, mask_paths, test_size=0.2, random_state=42)

    train_loader = DataLoader(LineDataset(train_imgs, train_masks), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(LineDataset(val_imgs, val_masks), batch_size=BATCH_SIZE)

    model = UNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCELoss()

    train_loss_list = []
    val_loss_list = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for img, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Train"):
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            pred = model(img)
            loss = loss_fn(pred, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        train_loss_list.append(total_loss / len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(DEVICE), mask.to(DEVICE)
                pred = model(img)
                loss = loss_fn(pred, mask)
                val_loss += loss.item()
        val_loss_list.append(val_loss / len(val_loader))
        print(f"Val Loss: {val_loss_list[-1]:.4f}")

    torch.save(model.state_dict(), "unet_trained.pth")
    print("Modelo guardado como unet_trained.pth")

    # Mostrar curva de treino
    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(val_loss_list, label="Val Loss")
    plt.legend()
    plt.title("Curva de treino")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.savefig("loss_curve.png")
    plt.show()

if __name__ == "__main__":
    train()
