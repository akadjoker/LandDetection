import cv2
import torch
import numpy as np
from torch import nn
from torchvision import transforms
from train import UNet   

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----- PARÂMETROS -----
VIDEO_PATH = "../portagem.mp4"
MODEL_PATH = "output/best_model.pth"
IMG_SIZE = 256  # Igual ao usado no treino

# ----- FUNÇÃO PARA PREPARAR FRAME -----
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
    tensor = tensor.unsqueeze(0).to(DEVICE)
 
    return tensor, frame_resized

# ----- CARREGAR MODELO -----
model = UNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ----- LER VÍDEO -----
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Erro ao abrir vídeo: {VIDEO_PATH}")
    exit()
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('saida_segmentada.avi', fourcc, fps, (width, height))
while True:
    ret, frame = cap.read()
    if not ret:
        break

    original_size = frame.shape[1], frame.shape[0]

    # Prepara o frame
    tensor, frame_resized = preprocess_frame(frame)

    with torch.no_grad():
        pred = model(tensor)
        mask = pred[0, 0].cpu().numpy()

    # Normaliza e binariza a máscara
    mask = (mask > 0.2).astype(np.uint8) * 255
    #mask = (mask > 0.2).astype(np.uint8) * 255


    # Redimensiona para o tamanho original do vídeo
    mask_resized = cv2.resize(mask, original_size)

    # Cria uma máscara colorida (vermelho)
    mask_color = np.zeros_like(frame)
    mask_color[:, :, 2] = mask_resized  # Vermelho

    # Sobrepõe no frame original
    overlaid = cv2.addWeighted(frame, 0.5, mask_color, 1.0, 0)

    cv2.imshow("Normal", overlaid)
    cv2.imshow("Mask", mask_color)
    out.write(overlaid)

    key = cv2.waitKey(30)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

