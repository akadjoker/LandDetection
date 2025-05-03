import cv2
import os

VIDEO_PATH = "centro.mp4"
OUTPUT_DIR = "frames"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

cap = cv2.VideoCapture(VIDEO_PATH)

frame_id = 0
saved = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Mostrar o frame
    frame_resized = cv2.resize(frame, (640, 480)) 
    cv2.imshow("Video - Pressiona ENTER para guardar o frame", frame_resized)

    key = cv2.waitKey(0)

    if key == 13:  # ENTER
        nome = f"{saved:05d}.jpg"
        path = os.path.join(OUTPUT_DIR, nome)
        cv2.imwrite(path, frame)
        print(f"Guardado: {path}")
        saved += 1

    elif key == ord('q'):
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
print("Processo concluído.")

