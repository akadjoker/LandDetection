import cv2
import numpy as np
import tensorflow as tf
from fast_scnn import build_fast_scnn  

# ==== CONFIG ====
VIDEO_PATH = 'centro.mp4'
MODEL_PATH = 'fast_scnn_best.keras'
IMG_SIZE = (256, 256)  # igual ao treino
OUTPUT_VIDEO = 'output_segmentado.mp4'

# ==== CARREGAR MODELO ====
model = build_fast_scnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
model.load_weights(MODEL_PATH)

# ==== ABRIR VÍDEO ====
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Erro ao abrir o vídeo")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ==== PREPARAR FRAME ====
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, IMG_SIZE)
    norm = resized / 255.0
    input_tensor = np.expand_dims(norm, axis=0)

    # ==== PREVISÃO ====
    pred = model.predict(input_tensor)[0, :, :, 0]
    mask = (pred > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (width, height))

    # ==== SOBREPOSIÇÃO ====
    mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.7, mask_color, 0.3, 0)

    # Mostrar e gravar
    cv2.imshow("Overlay", overlay)
    out.write(overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

