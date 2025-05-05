import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import LaneDataset  # O teu script do Dataset
from model import UNetSmall      # O modelo que criámos
from tqdm import tqdm

# ------ CONFIGURAÇÕES ------
IMAGENS_DIR = "imagens"
MASKS_DIR = "masks"

BATCH_SIZE = 4
EPOCHS = 40
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {DEVICE}")

# ------ DATASET ------
from albumentations import Compose, Resize, HorizontalFlip, RandomBrightnessContrast, ShiftScaleRotate, GaussNoise, Normalize
from albumentations.pytorch import ToTensorV2

transform = Compose([
    Resize(256, 256),
    #HorizontalFlip(p=0.5),
    RandomBrightnessContrast(p=0.5),
    #ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
    GaussNoise(p=0.2),
    Normalize(),
    ToTensorV2()
])

dataset = LaneDataset(IMAGENS_DIR, MASKS_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ------ MODELO ------
model = UNetSmall().to(DEVICE)

# ------ LOSS e OPTIMIZER ------
criterion = nn.BCELoss()  # Binary Cross Entropy
optimizer = optim.Adam(model.parameters(), lr=LR)

# ------ LOOP DE TREINO ------
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    loop = tqdm(dataloader, desc=f"Época [{epoch+1}/{EPOCHS}]")

    for imgs, masks in loop:
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        preds = model(imgs)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} concluída. Loss médio: {epoch_loss / len(dataloader)}")

# ------ GUARDAR MODELO ------
torch.save(model.state_dict(), "lane_model.pth")
print("Modelo guardado em 'lane_model.pth'.")

