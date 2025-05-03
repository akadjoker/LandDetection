import cv2
import os

VIDEO_PATH = "centro.mp4"
OUTPUT_DIR = "frames"
FPS_EXTRAIR = 1  # Quantos frames por segundo queres extrair

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

cap = cv2.VideoCapture(VIDEO_PATH)
fps_video = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps_video / FPS_EXTRAIR)

count = 0
saved = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if count % frame_interval == 0:
        nome = f"{saved:05d}.jpg"
        path = os.path.join(OUTPUT_DIR, nome)
        cv2.imwrite(path, frame)
        print(f"Guardado: {path}")
        saved += 1

    count += 1

cap.release()
print("Extração concluída.")

