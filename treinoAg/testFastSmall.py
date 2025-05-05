import cv2
import time
import numpy as np
import torch
from torch import nn
from fastSmall import JetsonRoadNet  
from torchvision import transforms

# ==== CONFIGURAÇÕES ====
MODEL_PATH = 'road_model/best_model.pth'
IMG_SIZE = 96
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


USE_CSI = False  

# ==== PREPARAR MODELO ====
model = JetsonRoadNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ==== TRANSFORMAÇÃO ====
def preprocess(frame):
    img_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    return img_tensor

# ==== CAPTURA DE VÍDEO ====
if USE_CSI:
    cam_set = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
    cap = cv2.VideoCapture(cam_set, cv2.CAP_GSTREAMER)
else:
    cap = cv2.VideoCapture("../centro.mp4")

if not cap.isOpened():
    print("Erro ao abrir câmara.")
    exit()

# ==== LOOP DE INFERÊNCIA ====
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    original = frame.copy()

    # Pré-processamento
    tensor = preprocess(frame)

    # Inferência
    with torch.no_grad():
        pred = model(tensor)
    pred_np = pred.squeeze().cpu().numpy()
    pred_binary = (pred_np > 0.2).astype(np.uint8) * 255

    # Redimensionar máscara para sobrepor no frame original
    pred_resized = cv2.resize(pred_binary, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Criar overlay verde
    mask_color = np.zeros_like(original)
    mask_color[:, :, 1] = pred_resized
    overlay = cv2.addWeighted(original, 0.7, mask_color, 0.8, 0)

    # ==== FPS ====
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Road Detection', overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

