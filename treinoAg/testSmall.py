import cv2
import numpy as np
import torch
from small import MobileUNet
from torchvision import transforms

# ==== CONFIG ====
VIDEO_PATH = '../estrada.mp4'
MODEL_PATH = 'output_lite/best_model.pth'
IMG_SIZE = (96, 96)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==== PREPARAR MODELO ====
model = MobileUNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ==== TRANSFORM ====
transform = transforms.Compose([
    transforms.ToTensor()
])

# ==== PROCESSAR VÍDEO ====
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Erro ao abrir vídeo.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('output_segmentado.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    original = frame.copy()

    img = cv2.resize(frame, IMG_SIZE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(tensor)
    pred = pred.cpu().squeeze().numpy()
    pred = (pred > 0.3).astype(np.uint8) * 255
    pred = cv2.resize(pred, (width, height))

    # Combinar com o frame original
    pred_resized = cv2.resize(pred, (width, height))

    # Criar máscara verde
    mask_color = np.zeros_like(original)
    mask_color[:, :, 1] = pred_resized  # canal verde
    overlay = cv2.addWeighted(original, 0.7, mask_color, 0.9, 0)

    cv2.imshow('Overlay', overlay)
    cv2.imshow('Mask', mask_color)
    out.write(overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

