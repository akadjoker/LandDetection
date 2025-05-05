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
        img = cv2.imread(self.image_paths[idx])
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.img_size)
        mask = (mask > 127).astype(np.float32)

        if self.augment:
            # Flip horizontal
            if random.random() > 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()

            # Flip vertical
            if random.random() > 0.5:
                img = np.flipud(img).copy()
                mask = np.flipud(mask).copy()

            # Rotação pequena (-15 a +15 graus)
            angle = random.uniform(-15, 15)
            matrix = cv2.getRotationMatrix2D((self.img_size[0] // 2, self.img_size[1] // 2), angle, 1)
            img = cv2.warpAffine(img, matrix, self.img_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, matrix, self.img_size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

            # Color jitter sempre aplicado
            jitter = transforms.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.1
            )
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img_tensor = jitter(img_tensor)
            img = (img_tensor * 255).permute(1, 2, 0).byte().numpy()

            # Zoom/crop leve
            if random.random() > 0.3:
                crop = random.uniform(0.9, 1.0)
                h, w = self.img_size
                nh, nw = int(h * crop), int(w * crop)
                top = random.randint(0, h - nh)
                left = random.randint(0, w - nw)
                img = img[top:top+nh, left:left+nw]
                mask = mask[top:top+nh, left:left+nw]
                img = cv2.resize(img, self.img_size)
                mask = cv2.resize(mask, self.img_size)

        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return img, mask


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

