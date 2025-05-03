import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class LaneDataset(Dataset):
    def __init__(self, imagens_dir, masks_dir, transform=None):
        self.imagens_dir = imagens_dir
        self.masks_dir = masks_dir
        self.imagens = sorted([f for f in os.listdir(imagens_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        self.transform = transform

    def __len__(self):
        return len(self.imagens)

    def __getitem__(self, idx):
        img_name = self.imagens[idx]
        mask_name = os.path.splitext(img_name)[0] + "_mask.png"

        # Ler imagem e máscara
        img_path = os.path.join(self.imagens_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # PyTorch espera RGB

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)  # Normalizar máscara para 0.0 ou 1.0

        # Se tiver transformações (augmentations)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)

        else:
            image = ToTensorV2()(image=image)["image"]
            mask = torch.tensor(mask).unsqueeze(0)

        return image, mask

# ------ EXEMPLO DE AUGMENTATION ------
transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
    A.GaussNoise(p=0.2),
    A.Normalize(),
    ToTensorV2()
])

# ------ TESTAR DATASET ------
if __name__ == "__main__":
    dataset = LaneDataset("imagens", "masks", transform=transform)
    print("Número de imagens:", len(dataset))

    # Testar um sample
    img, mask = dataset[0]
    print("Imagem shape:", img.shape)
    print("Máscara shape:", mask.shape)

