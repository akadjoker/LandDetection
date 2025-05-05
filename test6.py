import cv2
import torch
import numpy as np
import time
from torchvision import transforms
from train_unet import UNet   

# Configs
VIDEO_PATH = "centro.mp4"
MODEL_PATH = "unet_trained.pth"
IMG_SIZE = (256, 256)
THRESHOLD = 0.5
FALLBACK_DURATION = 2.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

class AngleMemory:
    def __init__(self):
        self.last_angle = 0.0
        self.timestamp = 0
        self.mode = "auto"

    def update(self, angle):
        self.last_angle = angle
        self.timestamp = time.time()
        self.mode = "real"

    def fallback(self):
        if time.time() - self.timestamp > FALLBACK_DURATION:
            self.mode = "cenoura"
        return self.last_angle

memory = AngleMemory()

cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('saida_segmentada.avi', fourcc, fps, (width, height))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    original = frame.copy()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(img_tensor)
        mask = (pred.squeeze().cpu().numpy() > THRESHOLD).astype(np.uint8) * 255

    mask = cv2.resize(mask, (width, height))
    overlay = frame.copy()

    ys, xs = np.where(mask == 255)
    angle_rad = memory.fallback()
    center_x = width // 2
    base_y = height - 10

    if len(xs) > 0:
        sorted_pts = sorted(zip(ys, xs))
        sampled = []
        seen_y = -100
        for y, x in sorted_pts:
            if y - seen_y >= 20:
                sampled.append((x, y))
                seen_y = y
            if len(sampled) >= 5:
                break

        if len(sampled) >= 2:
            target_x, target_y = sampled[-1]   
            dx = target_x - center_x
            dy = base_y - target_y
            angle_rad = np.arctan2(dx, dy)
            memory.update(angle_rad)
            cv2.circle(overlay, (target_x, target_y), 5, (0, 255, 0), -1)

 
    length = 60
    arrow_x = int(center_x + length * np.sin(angle_rad))
    arrow_y = int(base_y - length * np.cos(angle_rad))
    cv2.arrowedLine(overlay, (center_x, base_y), (arrow_x, arrow_y), (0, 255, 255), 4)

 
    guiador_center = (80, height - 80)
    radius = 50
    guiador_x = int(guiador_center[0] + radius * np.sin(angle_rad))
    guiador_y = int(guiador_center[1] - radius * np.cos(angle_rad))
    cv2.circle(overlay, guiador_center, radius, (200, 200, 200), 2)
    cv2.line(overlay, guiador_center, (guiador_x, guiador_y), (0, 255, 0), 3)

    cv2.putText(overlay, f"Modo: {memory.mode.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Direcao Simples", overlay)
    out.write(overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

