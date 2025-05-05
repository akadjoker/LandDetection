import cv2
import numpy as np
import torch
import time
from small import MobileUNet
from torchvision import transforms

MODEL_PATH = 'output_lite/best_model.pth'
IMG_SIZE = (96, 96)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


model = MobileUNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


transform = transforms.Compose([
    transforms.ToTensor()
])


def gstreamer_pipeline(
    capture_width=640,
    capture_height=480,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=2
):
    return (
        f'nvarguscamerasrc ! '
        f'video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, '
        f'format=NV12, framerate={framerate}/1 ! '
        f'nvvidconv flip-method={flip_method} ! '
        f'video/x-raw, width={display_width}, height={display_height}, format=BGRx ! '
        f'videoconvert ! '
        f'video/x-raw, format=BGR ! appsink'
    )

cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Erro ao abrir a câmara CSI.")
    exit()


prev_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame não capturado.")
        break

    original = frame.copy()

    img = cv2.resize(frame, IMG_SIZE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(tensor)
    pred = pred.cpu().squeeze().numpy()
    pred = (pred > 0.3).astype(np.uint8) * 255
    pred = cv2.resize(pred, (original.shape[1], original.shape[0]))

    # Criar máscara verde
    mask_color = np.zeros_like(original)
    mask_color[:, :, 1] = pred  # canal verde

    overlay = cv2.addWeighted(original, 0.7, mask_color, 0.9, 0)


    frame_count += 1
    if frame_count >= 10:
        current_time = time.time()
        fps = frame_count / (current_time - prev_time)
        prev_time = current_time
        frame_count = 0
    else:
        fps = 0

    if fps > 0:
        cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Overlay', overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

