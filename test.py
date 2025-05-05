import cv2
import torch
import numpy as np
from train_unet import UNet   

# ------ CONFIGURAÇÕES ------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIDEO_PATH = "estrada.mp4"
VIDEO_PATH = "small_portagem.mp4"
VIDEO_PATH = "centro.mp4"
#VIDEO_PATH = "portagem.mp4"
MODEL_PATH = "unet_trained.pth"
INPUT_SIZE = (256, 256)

# ------ CARREGAR MODELO ------
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ------ VIDEO ------
cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    original = frame.copy()

    # Prepara imagem para o modelo
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, INPUT_SIZE)
    img = img / 255.0
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

    # Predição
    with torch.no_grad():
        pred = model(img)
        pred = pred.squeeze().cpu().numpy()

    # Binarizar a saída
    mask = (pred > 0.1).astype(np.uint8) * 255

    # Upscale para tamanho original
    mask_up = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

    # Overlay na imagem original
    frame[mask_up > 127] = [0, 0, 255]  # Pinta as linhas a vermelho

    cv2.imshow("Lane Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

