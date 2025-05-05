import os
import cv2
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt

# ==== CONFIG ====
IMAGE_DIR = 'images'
MASK_DIR = 'masks'
IMG_SIZE = (256, 256)
BATCH_SIZE = 4
EPOCHS = 25
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==== DATASET COM AUGMENT ====
class LineDataset(Dataset):
    def __init__(self, image_paths, mask_paths, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment

        self.img_transform = T.Compose([
            T.ToTensor(),
            T.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.05
            )
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, IMG_SIZE)
        mask = (mask > 5).astype(np.float32)  # binário

        if self.augment and np.random.rand() > 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        img = self.img_transform(img)
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return img, mask

# ==== MODELO MELHORADO ====
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            )
        self.enc1 = conv_block(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.middle = nn.Sequential(
            conv_block(256, 512),
            nn.Dropout(0.3)
        )

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
        return self.final(d1)  # Sem sigmoid aqui!

# ==== TREINO ====
def train():
    img_paths = sorted(glob(os.path.join(IMAGE_DIR, '*')))
    mask_paths = sorted(glob(os.path.join(MASK_DIR, '*')))

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        img_paths, mask_paths, test_size=0.2, random_state=42
    )

    train_loader = DataLoader(LineDataset(train_imgs, train_masks, augment=True),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(LineDataset(val_imgs, val_masks, augment=False),
                            batch_size=BATCH_SIZE)

    model = UNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loss_list = []
    val_loss_list = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for img, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Train"):
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            pred = model(img)
            loss = loss_fn(pred, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(DEVICE), mask.to(DEVICE)
                pred = model(img)
                loss = loss_fn(pred, mask)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_loss_list.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    # Guardar modelo
    torch.save(model.state_dict(), "unet_trained.pth")
    print("Modelo guardado como unet_trained.pth")

    # Curvas
    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(val_loss_list, label="Val Loss")
    plt.legend()
    plt.title("Curva de treino")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.show()

if __name__ == "__main__":
    train()

