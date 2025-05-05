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

# Região de interesse (ROI) - definir área relevante para detecção
# Assumindo que queremos focar na parte inferior da imagem
roi_height = height // 3  # Usar o terço inferior da imagem
roi_top = height - roi_height

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
    pred_resized = cv2.resize(pred, (width, height))

    # Criar máscara verde
    mask_color = np.zeros_like(original)
    mask_color[:, :, 1] = pred_resized  # canal verde
    overlay = cv2.addWeighted(original, 0.7, mask_color, 0.9, 0)
    
    # ===== EXTRAÇÃO DE LINHAS E CÁLCULO DO ÂNGULO =====
    # Focar na região de interesse (parte inferior da imagem)
    roi_mask = pred_resized[roi_top:height, :]
    
    # Encontrar linhas usando transformada de Hough
    edges = cv2.Canny(roi_mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=100)
    
    # Variáveis para armazenar informações sobre as linhas
    left_lines = []
    right_lines = []
    center_x = width // 2
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:  # Evitar divisão por zero
                continue
                
            slope = (y2 - y1) / (x2 - x1)
            
            # Filtrar linhas horizontais
            if abs(slope) < 0.3:
                continue
                
            # Classificar como linha esquerda ou direita
            if slope < 0:  # Linha esquerda (negativa na imagem)
                left_lines.append(line[0])
            else:  # Linha direita (positiva na imagem)
                right_lines.append(line[0])
        
        # Desenhar linhas detectadas no frame
        if len(left_lines) > 0:
            for x1, y1, x2, y2 in left_lines:
                cv2.line(overlay, (x1, y1 + roi_top), (x2, y2 + roi_top), (255, 0, 0), 2)
                
        if len(right_lines) > 0:
            for x1, y1, x2, y2 in right_lines:
                cv2.line(overlay, (x1, y1 + roi_top), (x2, y2 + roi_top), (0, 0, 255), 2)
        
        # Calcular pontos médios para cada lado
        left_x = 0
        right_x = width
        
        if len(left_lines) > 0:
            left_points = np.array(left_lines)
            left_x = np.mean(left_points[:, [0, 2]])
            
        if len(right_lines) > 0:
            right_points = np.array(right_lines)
            right_x = np.mean(right_points[:, [0, 2]])
        
        # Calcular o ponto central desejado (entre as duas linhas)
        target_position = (left_x + right_x) / 2
        
        # Calcular o erro de posição (diferença entre o centro do carro e a posição alvo)
        error = center_x - target_position
        
        # Calcular o ângulo de direção (positivo: virar à direita, negativo: virar à esquerda)
        # O fator de escala pode ser ajustado baseado na sensibilidade desejada
        max_angle = 30  # ângulo máximo de esterçamento em graus
        steering_angle = np.clip(error / (width/4) * max_angle, -max_angle, max_angle)
        
        # Desenhar uma indicação do ângulo de direção
        direction_text = f"angle: {steering_angle:.1f}°"
        cv2.putText(overlay, direction_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Desenhar uma linha indicando a direção
        start_point = (center_x, height)
        end_point = (int(center_x - steering_angle * 5), height - 150)  # Multiplicador para visualização
        cv2.line(overlay, start_point, end_point, (0, 255, 255), 3)
        
        # Desenhar indicadores de posição
        cv2.circle(overlay, (int(target_position), height - 50), 10, (255, 255, 0), -1)  # Posição alvo
        cv2.circle(overlay, (center_x, height - 50), 10, (0, 0, 255), -1)  # Centro do carro
    
    cv2.imshow('Steering', overlay)
    out.write(overlay)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
