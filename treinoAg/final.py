import cv2
import time
import numpy as np
import torch
import collections
from torch import nn
from fastSmall import JetsonRoadNet  
from torchvision import transforms

# ==== CONFIGURAÇÕES ====
MODEL_PATH = 'road_model/best_model.pth'
IMG_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


USE_CSI = False  
USE_MORPH=True


class LaneFollowingSystem:
    def __init__(self):
        self.roi_height_ratio = 0.4  # Usar 40% inferior da imagem
        self.threshold = 0.3  # Limiar para a segmentação
        self.max_angle = 30  # Ângulo máximo de esterçamento em graus
        
        # Filtro temporal (média móvel)
        self.angle_history = collections.deque(maxlen=5)
        
        # Rastreamento de linhas perdidas
        self.frames_without_lines = 0
        self.max_frames_without_lines = 10
        self.last_valid_angle = 0
        
        # Pontos para predição de trajetória
        self.num_trajectory_points = 5
        
        # Ajuste dinâmico de sensibilidade (simulando velocidade)
        self.speed = 50
        self.base_sensitivity = 1.0
        
        # Inicializar modelo
        self.model = JetsonRoadNet()
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        self.model.to(DEVICE)
        self.model.eval()
        
        # Transformação de imagem
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # Inicializar fits para as linhas (para polynomials)
        self.left_fit = None
        self.right_fit = None
        self.center_fit = None
        
        # Coeficientes de confiança para as linhas
        self.left_confidence = 0
        self.right_confidence = 0
        
    def detect_lanes(self, frame):
        """Detecta as linhas da estrada usando o modelo de segmentação"""
        height, width = frame.shape[:2]
        
        # Preparar imagem para o modelo
        img = cv2.resize(frame, IMG_SIZE)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img_rgb).unsqueeze(0).to(DEVICE)
        
        # Inferência
        with torch.no_grad():
            pred = self.model(tensor)
        pred = pred.cpu().squeeze().numpy()
        pred = (pred > self.threshold).astype(np.uint8) * 255
        pred_resized = cv2.resize(pred, (width, height))
        
        return pred_resized
    
    def process_frame(self, frame):
        """Processa um frame e retorna o ângulo de direção"""
        original = frame.copy()
        height, width = frame.shape[:2]
        
        # Definir ROI
        roi_height = int(height * self.roi_height_ratio)
        roi_top = height - roi_height
        
        # Detectar linhas
        lane_mask = self.detect_lanes(frame)
        
        # Criar overlay visual
        mask_color = np.zeros_like(original)
        mask_color[:, :, 1] = lane_mask  # canal verde
        overlay = cv2.addWeighted(original, 0.7, mask_color, 0.9, 0)
        
        # Focar na região de interesse
        roi_mask = lane_mask[roi_top:height, :]
        
        # Encontrar pontos das linhas em diferentes alturas para polynomials
        left_points = []
        right_points = []
        center_points = []
        
        # Dividir ROI em várias alturas para amostragem
        row_samples = np.linspace(0, roi_height-1, self.num_trajectory_points, dtype=np.int32)
        center_x = width // 2
        
        # Encontrar pontos em cada altura
        for i, y in enumerate(row_samples):
            row = roi_mask[y, :]
            left_x = 0
            right_x = width - 1
            
            # Encontrar pontos da linha à esquerda do centro
            left_indices = np.where(row[:center_x] > 0)[0]
            if len(left_indices) > 0:
                left_x = left_indices[-1]  # Ponto mais à direita das detecções à esquerda
                left_points.append((left_x, y + roi_top))
                self.left_confidence = min(1.0, self.left_confidence + 0.1)
            else:
                self.left_confidence = max(0.0, self.left_confidence - 0.1)
            
            # Encontrar pontos da linha à direita do centro
            right_indices = np.where(row[center_x:] > 0)[0]
            if len(right_indices) > 0:
                right_x = right_indices[0] + center_x  # Ponto mais à esquerda das detecções à direita
                right_points.append((right_x, y + roi_top))
                self.right_confidence = min(1.0, self.right_confidence + 0.1)
            else:
                self.right_confidence = max(0.0, self.right_confidence - 0.1)
            
            # Calcular ponto central estimado entre as linhas
            center_points.append(((left_x + right_x) // 2, y + roi_top))
                
        # Ajustar polynomial para cada conjunto de pontos se houver pontos suficientes
        if len(left_points) >= 3 and self.left_confidence > 0.5:
            left_x = [p[0] for p in left_points]
            left_y = [p[1] for p in left_points]
            self.left_fit = np.polyfit(left_y, left_x, 2)
            
            # Desenhar linha esquerda ajustada
            plot_y = np.linspace(roi_top, height-1, 100)
            plot_x = self.left_fit[0]*plot_y**2 + self.left_fit[1]*plot_y + self.left_fit[2]
            pts = np.vstack((plot_x, plot_y)).T.astype(np.int32)
            cv2.polylines(overlay, [pts], False, (255, 0, 0), 2)
        else:
            # Usar fit anterior com confiança reduzida
            pass
            
        if len(right_points) >= 3 and self.right_confidence > 0.5:
            right_x = [p[0] for p in right_points]
            right_y = [p[1] for p in right_points]
            self.right_fit = np.polyfit(right_y, right_x, 2)
            
            # Desenhar linha direita ajustada
            plot_y = np.linspace(roi_top, height-1, 100)
            plot_x = self.right_fit[0]*plot_y**2 + self.right_fit[1]*plot_y + self.right_fit[2]
            pts = np.vstack((plot_x, plot_y)).T.astype(np.int32)
            cv2.polylines(overlay, [pts], False, (0, 0, 255), 2)
        else:
            # Usar fit anterior com confiança reduzida
            pass
        
        # Decidir qual abordagem usar com base na confiança das linhas detectadas
        steering_angle = 0
        has_valid_lines = False
        
        # Calcular polynomial para a linha central (média das duas linhas)
        if self.left_fit is not None and self.right_fit is not None and self.left_confidence > 0.3 and self.right_confidence > 0.3:
            # Temos ambas as linhas - usar a média
            self.center_fit = [(self.left_fit[i] + self.right_fit[i])/2 for i in range(3)]
            has_valid_lines = True
        elif self.left_fit is not None and self.left_confidence > 0.3:
            # Só temos a linha esquerda - estimar a direita
            estimated_lane_width = 320  # pixels (ajustar conforme necessário)
            self.center_fit = [self.left_fit[0], self.left_fit[1], self.left_fit[2] + estimated_lane_width/2]
            has_valid_lines = True
        elif self.right_fit is not None and self.right_confidence > 0.3:
            # Só temos a linha direita - estimar a esquerda
            estimated_lane_width = 320  # pixels (ajustar conforme necessário)
            self.center_fit = [self.right_fit[0], self.right_fit[1], self.right_fit[2] - estimated_lane_width/2]
            has_valid_lines = True
            
        if has_valid_lines:
            # Desenhar linha central estimada
            if self.center_fit is not None:
                plot_y = np.linspace(roi_top, height-1, 100)
                plot_x = self.center_fit[0]*plot_y**2 + self.center_fit[1]*plot_y + self.center_fit[2]
                pts = np.vstack((plot_x, plot_y)).T.astype(np.int32)
                cv2.polylines(overlay, [pts], False, (0, 255, 255), 2)
            
            # Calcular curvatura para ajustar o ângulo
            # Avaliamos a curvatura na base da imagem
            y_eval = height
            if self.center_fit is not None:
                # Calcular a derivada da curva no ponto mais baixo
                dx_dy = 2*self.center_fit[0]*y_eval + self.center_fit[1]
                
                # Converter a inclinação para ângulo
                angle_rad = np.arctan(dx_dy)
                curvature_angle = np.degrees(angle_rad)
                
                # Calcular o offset do centro
                # Posição ideal é o valor da polynomial no ponto mais baixo
                ideal_position = self.center_fit[0]*y_eval**2 + self.center_fit[1]*y_eval + self.center_fit[2]
                current_position = center_x
                offset = current_position - ideal_position
                
                # Ajustar com base no offset e curvatura
                steering_angle = -curvature_angle + 0.5 * offset / (width/4) * self.max_angle
                
                # Ajustar sensibilidade com base na velocidade
                speed_factor = 1.0 - (self.speed - 30) / 120  # Menor sensibilidade em velocidades altas
                speed_factor = max(0.5, min(1.5, speed_factor))  # Limitar entre 0.5 e 1.5
                
                steering_angle *= self.base_sensitivity * speed_factor
                
                # Limitar o ângulo máximo
                steering_angle = np.clip(steering_angle, -self.max_angle, self.max_angle)
                
                # Adicionar ao histórico para filtragem temporal
                self.angle_history.append(steering_angle)
                
                # Aplicar filtragem temporal (média móvel)
                if len(self.angle_history) > 0:
                    steering_angle = sum(self.angle_history) / len(self.angle_history)
                
                # Armazenar o último ângulo válido
                self.last_valid_angle = steering_angle
                self.frames_without_lines = 0
            
        else:
            # Nenhuma linha válida detectada
            self.frames_without_lines += 1
            
            if self.frames_without_lines < self.max_frames_without_lines:
                # Usar o último ângulo válido com decaimento
                decay_factor = 0.9 ** self.frames_without_lines
                steering_angle = self.last_valid_angle * decay_factor
            else:
                # Muitos frames sem linhas, diminuir progressivamente o ângulo
                steering_angle = self.last_valid_angle * 0.5
                
        # Visualizar o ângulo de direção
        direction_text = f"Angulo: {steering_angle:.1f}°"
        cv2.putText(overlay, direction_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Visualizar velocidade simulada
        speed_text = f"Vel: {self.speed:.0f} km/h"
        cv2.putText(overlay, speed_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Visualizar confiança
        conf_text = f"Conf L: {self.left_confidence:.1f} R: {self.right_confidence:.1f}"
        cv2.putText(overlay, conf_text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Desenhar indicador de direção
        start_point = (center_x, height)
        steer_length = 150
        end_point = (int(center_x - steering_angle * 5), height - steer_length)
        cv2.line(overlay, start_point, end_point, (0, 255, 255), 5)
        
        # Desenhar região de interesse
        cv2.line(overlay, (0, roi_top), (width, roi_top), (255, 0, 0), 2)
        
        # Simular mudança de velocidade (aqui poderia ser baseado em input real)
        # Em um sistema real, isso seria baseado no comando do acelerador
        self.speed += np.random.uniform(-0.5, 0.5)  # Pequena variação aleatória
        self.speed = np.clip(self.speed, 30, 120)  # Limitar entre 30 e 120 km/h
        
        return overlay, steering_angle



# ==== TRANSFORMAÇÃO ====
def preprocess(frame):
    img_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    return img_tensor

 
if USE_CSI:
    cam_set = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
    cap = cv2.VideoCapture(cam_set, cv2.CAP_GSTREAMER)
else:
    cap = cv2.VideoCapture("../centro.mp4")

if not cap.isOpened():
    print("Erro ao abrir câmara.")
    exit()


fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

lane_system = LaneFollowingSystem()
 
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    overlay, angle = lane_system.process_frame(frame)
    overlay = cv2.resize(overlay, (640, 480))
    

    # ==== FPS ====
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(overlay, f"FPS: {fps:.1f} ", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Road Detection', overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

