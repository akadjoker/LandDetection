import os
import cv2
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import random
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

# Número de pontos por faixa de estrada e número de faixas
NUM_LANES = 1
POINTS_PER_LANE = 6
TOTAL_POINTS = 12

# Transformações para augmentação
transform_augment = transforms.ColorJitter(
    brightness=0.4,
    contrast=0.4,
    saturation=0.4,
    hue=0.1
)

transform_augment = transforms.Compose([
    transforms.ColorJitter(
        brightness=0.6,
        contrast=0.6,
        saturation=0.6,
        hue=0.3
    ),
    transforms.RandomApply(
        [transforms.GaussianBlur(3)], p=0.1
    )
])

transform_augment = transforms.Compose([
    transforms.ColorJitter(
        brightness=0.5,
        contrast=0.5,
        saturation=0.4,
        hue=0.1
    ),
    transforms.RandomApply(
        [transforms.GaussianBlur(3)], p=0.1
    ),
    transforms.RandomApply(
        [transforms.Lambda(
            lambda img: img + 0.02 * torch.randn_like(img)
        )], p=0.2  # Ruído gaussiano leve
    )
])


def set_seed(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train model for lane keypoint detection')
    parser.add_argument('--image_dir', type=str, default='images', help='Directory with input images')
    parser.add_argument('--label_dir', type=str, default='labels', help='Directory with label files (.lines.txt)')
    parser.add_argument('--img_size', type=int, default=256, help='Image size (square)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=80, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--patience', type=int, default=7, help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    return parser.parse_args()

# ==== DATASET ====
class LaneKeypointDataset(Dataset):
    def __init__(self, image_paths, label_paths, img_size=(256, 256), transform=None, augment=False):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.img_size = img_size
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def load_labels(self, label_path):
        """Load keypoints from label file"""
        keypoints = np.zeros((NUM_LANES * POINTS_PER_LANE, 2), dtype=np.float32)

        try:
            with open(label_path, 'r') as f:
                lines = f.read().strip().split('\n')

            coords = []
            for line in lines:
                if not line.strip():
                    continue
                parts = list(map(int, line.strip().split()))
                coords.extend(parts)

 
            points = []
            for i in range(0, len(coords), 2):
                points.append( (coords[i], coords[i+1]) )

            # Se tem menos de 6 pontos, interpolar
            while len(points) < POINTS_PER_LANE:
                if len(points) >= 2:
                    # Interpolar entre os primeiros dois pontos
                    p1 = np.array(points[0])
                    p2 = np.array(points[-1])
                    alpha = (len(points)) / (POINTS_PER_LANE - 1)
                    new_point = ((1 - alpha) * p1 + alpha * p2).astype(int)
                    points.insert(-1, tuple(new_point))
                else:
                    # Se não tem pontos suficientes, preenche com (0,0)
                    print("menos que 6")
                    points.append( (0,0) )

            # Se tem mais de 6 pontos, corta
            points = points[:POINTS_PER_LANE]

            # Preencher keypoints
            for idx, (x, y) in enumerate(points):
                keypoints[idx] = [x, y]

        except Exception as e:
            logger.warning(f"Error loading label from {label_path}: {e}")

        # Normalizar
        img_h, img_w = 480, 640
        keypoints[:, 0] /= img_w
        keypoints[:, 1] /= img_h

        return keypoints.flatten()


    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        # Load image
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load keypoints
        keypoints = self.load_labels(label_path)
        
        # Apply augmentations
        if self.augment:
            # Flip horizontal with keypoints
            if random.random() > 0.5:
                img = np.fliplr(img).copy()
                # Ajustar coordenadas x dos keypoints (reflexão horizontal)
                for i in range(0, len(keypoints), 2):
                    keypoints[i] = 1.0 - keypoints[i]  # inverte x (já normalizado)

            if random.random() > 0.5:
                alpha = random.uniform(0.7, 1.3)  # contraste
                beta = random.uniform(-30, 30)    # brilho
                img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)


            # Color jitter
            img = transform_augment(torch.from_numpy(img).permute(2,0,1).float()/255.0)
            img = (img * 255).permute(1,2,0).byte().numpy()
            
            # Pequenas variações de brilho e contraste
            if random.random() > 0.5:
                alpha = random.uniform(0.8, 1.2)  # Contraste
                beta = random.uniform(-20, 20)    # Brilho
                img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

        # Normalizar imagem
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        keypoints = torch.from_numpy(keypoints).float()
        
        return img, keypoints

# ==== MODEL ====
class LaneKeypointNet(nn.Module):
    def __init__(self, num_points=TOTAL_POINTS):
        super(LaneKeypointNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # Testar tamanho dinamicamente
        dummy_input = torch.zeros(1, 3, 256, 256)
        dummy_output = self.features(dummy_input)
        cnn_output_size = dummy_output.numel()

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_output_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_points)
        )

        
    def forward(self, x):
        x = self.features(x)
        keypoints = self.regressor(x)
        # Aplicar sigmoid para restringir as coordenadas ao intervalo [0,1]
        return torch.sigmoid(keypoints)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, verbose=True, path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.path = path
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            if self.verbose:
                logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
            return False
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info("Early stopping triggered")
                return True
        return False

def train(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Get data paths
    img_paths = sorted(glob(os.path.join(args.image_dir, '*.*')))
    label_paths = []
    
    # Encontrar os arquivos de label correspondentes
    for img_path in img_paths:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(args.label_dir, f"{img_name}.lines.txt")
        if os.path.exists(label_path):
            label_paths.append(label_path)
        else:
            logger.warning(f"Label not found for {img_path}")
            img_paths.remove(img_path)
    
    if len(img_paths) == 0 or len(label_paths) == 0:
        logger.error(f"No valid image-label pairs found")
        return
    
    logger.info(f"Found {len(img_paths)} valid image-label pairs")
    
    # Split data
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        img_paths, label_paths, test_size=0.2, random_state=args.seed
    )
    
    # Create dataloaders
    train_dataset = LaneKeypointDataset(
        train_imgs, train_labels, img_size=(args.img_size, args.img_size), augment=True
    )
    val_dataset = LaneKeypointDataset(
        val_imgs, val_labels, img_size=(args.img_size, args.img_size)
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True
    )
    
    # Initialize model, optimizer, scheduler, and loss function
    model = LaneKeypointNet(num_points=TOTAL_POINTS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    loss_fn = nn.MSELoss()  # MSE para regressão de coordenadas
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=args.patience, 
        verbose=True, 
        path=os.path.join(args.output_dir, 'best_model.pth')
    )
    
    # Track metrics
    train_losses = []
    val_losses = []
    
    # Training loop
    start_time = datetime.now()
    logger.info(f"Starting training at {start_time}")
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Train")
        for batch_idx, (images, keypoints) in enumerate(train_loop):
            images, keypoints = images.to(device), keypoints.to(device)
            
            # Forward pass
            pred_keypoints = model(images)
            loss = loss_fn(pred_keypoints, keypoints)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            
            # Update progress bar
            train_loop.set_postfix(loss=loss.item())
        
        # Calculate average metrics for the epoch
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Val")
            for batch_idx, (images, keypoints) in enumerate(val_loop):
                images, keypoints = images.to(device), keypoints.to(device)
                
                # Forward pass
                pred_keypoints = model(images)
                loss = loss_fn(pred_keypoints, keypoints)
                
                # Update metrics
                val_loss += loss.item()
                
                # Update progress bar
                val_loop.set_postfix(loss=loss.item())
        
        # Calculate average metrics for the epoch
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Log results
        logger.info(f"Epoch {epoch+1}/{args.epochs} - "
                   f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Save sample predictions
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            save_predictions(model, val_loader, device, epoch, args.output_dir)
        
        # Check early stopping
        if early_stopping(avg_val_loss, model):
            logger.info("Early stopping triggered")
            break
    
    # Training complete
    end_time = datetime.now()
    training_duration = end_time - start_time
    logger.info(f"Training completed in {training_duration}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Plot and save training curves
    plot_training_curves(train_losses, val_losses, args.output_dir)

def save_predictions(model, dataloader, device, epoch, output_dir):
    """Save sample predictions for visualization"""
    model.eval()
    # Get a batch of validation data
    images, keypoints = next(iter(dataloader))
    images, keypoints = images.to(device), keypoints.to(device)
    
    with torch.no_grad():
        pred_keypoints = model(images)
    
    # Converter para numpy para visualização
    images_np = images.cpu().numpy()
    keypoints_np = keypoints.cpu().numpy()
    pred_keypoints_np = pred_keypoints.cpu().numpy()
    
    # Salvar as primeiras 4 amostras (ou menos se o batch size for menor)
    num_samples = min(4, len(images))
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        # Imagem original
        img = images_np[i].transpose(1, 2, 0)
        axes[i].imshow(img)
        
        # Origem das dimensões da imagem
        img_h, img_w = img.shape[0], img.shape[1]
        
        # Desenhar ground truth keypoints (verde)
        gt_points = keypoints_np[i].reshape(-1, 2)
        gt_points[:, 0] *= img_w  # Desnormalizar x
        gt_points[:, 1] *= img_h  # Desnormalizar y
        
        for lane in range(NUM_LANES):
            start_idx = lane * POINTS_PER_LANE
            end_idx = start_idx + POINTS_PER_LANE
            lane_points = gt_points[start_idx:end_idx]
            
            # Desenhar pontos
            axes[i].scatter(lane_points[:, 0], lane_points[:, 1], c='g', s=30, label='Ground Truth' if lane==0 else "")
            
            # Conectar pontos com linhas
            if np.any(lane_points):  # Se há pontos válidos
                axes[i].plot(lane_points[:, 0], lane_points[:, 1], 'g-', linewidth=1)
        
        # Desenhar predicted keypoints (vermelho)
        pred_points = pred_keypoints_np[i].reshape(-1, 2)
        pred_points[:, 0] *= img_w  # Desnormalizar x
        pred_points[:, 1] *= img_h  # Desnormalizar y
        
        for lane in range(NUM_LANES):
            start_idx = lane * POINTS_PER_LANE
            end_idx = start_idx + POINTS_PER_LANE
            lane_points = pred_points[start_idx:end_idx]
            
            # Desenhar pontos
            axes[i].scatter(lane_points[:, 0], lane_points[:, 1], c='r', s=30, label='Prediction' if lane==0 else "")
            
            # Conectar pontos com linhas
            if np.any(lane_points):  # Se há pontos válidos
                axes[i].plot(lane_points[:, 0], lane_points[:, 1], 'r-', linewidth=1)
        
        axes[i].set_title(f"Sample {i+1}")
        if i == 0:  # Mostrar legenda apenas na primeira imagem
            axes[i].legend()
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"predictions_epoch_{epoch+1}.png"))
    plt.close()

def plot_training_curves(train_losses, val_losses, output_dir):
    """Plot and save training curves"""
    plt.figure(figsize=(10, 5))
    
    # Plot losses
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"))
    plt.close()


def inference_smoth_video(model_path, video_path, output_path=None, img_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LaneKeypointNet(num_points=TOTAL_POINTS)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Erro ao abrir o vídeo: {video_path}")
        return

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]  # Verde, Azul, Vermelho

    # Inicializar pontos suavizados como None
    smoothed_points = None
    alpha = 0.6  # fator de suavização (quanto menor, mais suave)

    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            original_h, original_w = frame.shape[:2]
            img = cv2.resize(frame, (img_size, img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.transpose(2, 0, 1) / 255.0
            img = torch.from_numpy(img).float().unsqueeze(0).to(device)

            pred_keypoints = model(img).cpu().numpy()[0]

            pred_points = pred_keypoints.reshape(-1, 2)
            pred_points[:, 0] *= original_w
            pred_points[:, 1] *= original_h

            # Aplicar suavização
            if smoothed_points is None:
                smoothed_points = pred_points.copy()
            else:
                smoothed_points = alpha * pred_points + (1 - alpha) * smoothed_points

            pred_points_int = smoothed_points.astype(np.int32)

            # Desenhar faixas
            for lane_idx in range(NUM_LANES):
                start_idx = lane_idx * POINTS_PER_LANE
                end_idx = start_idx + POINTS_PER_LANE
                lane_points = pred_points_int[start_idx:end_idx]
                color = colors[lane_idx % len(colors)]

                for i in range(len(lane_points) - 1):
                    p1, p2 = tuple(lane_points[i]), tuple(lane_points[i + 1])
                    cv2.line(frame, p1, p2, color, 3)

                for point in lane_points:
                    cv2.circle(frame, tuple(point), 5, color, -1)

            cv2.imshow('Lane Detection', frame)

            if output_path:
                out.write(frame)

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

    logger.info(f"Processados {frame_count} frames")


def inference_video(model_path, video_path, output_path=None, img_size=256):

    # Configurar device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Carregar modelo
    model = LaneKeypointNet(num_points=TOTAL_POINTS)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Abrir vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Erro ao abrir o vídeo: {video_path}")
        return
    
    # Configurar gravação de vídeo (opcional)
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    # Cores para cada faixa
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]  # Verde, Azul, Vermelho
    
    # Processamento do vídeo
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocessar frame para o modelo
            original_h, original_w = frame.shape[:2]
            img = cv2.resize(frame, (img_size, img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.transpose(2, 0, 1) / 255.0
            img = torch.from_numpy(img).float().unsqueeze(0).to(device)
            
            # Inferência
            pred_keypoints = model(img).cpu().numpy()[0]
            
            # Remodelar e desnormalizar as coordenadas
            pred_points = pred_keypoints.reshape(-1, 2)
            pred_points[:, 0] = pred_points[:, 0] * original_w
            pred_points[:, 1] = pred_points[:, 1] * original_h
            pred_points = pred_points.astype(np.int32)
            
            # Desenhar faixas
            for lane_idx in range(NUM_LANES):
                start_idx = lane_idx * POINTS_PER_LANE
                end_idx = start_idx + POINTS_PER_LANE
                lane_points = pred_points[start_idx:end_idx]
                color = colors[lane_idx % len(colors)]
                
                # Desenhar linhas entre pontos
                for i in range(len(lane_points) - 1):
                    p1, p2 = tuple(lane_points[i]), tuple(lane_points[i + 1])
                    cv2.line(frame, p1, p2, color, 3)
                
                # Desenhar pontos
                for point in lane_points:
                    cv2.circle(frame, tuple(point), 5, color, -1)
            
            # Exibir frame
            cv2.imshow('Lane Detection', frame)
            
            # Gravar frame processado (opcional)
            if output_path:
                out.write(frame)
            
            frame_count += 1
            
            # Sair com a tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Liberar recursos
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    
    logger.info(f"Processados {frame_count} frames")

class Kalman2D:
    def __init__(self, init_x, init_y):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kalman.statePre = np.array([[init_x], [init_y], [0], [0]], dtype=np.float32)

    def correct(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kalman.correct(measurement)

    def predict(self):
        prediction = self.kalman.predict()
        return int(prediction[0]), int(prediction[1])


def inference_video_with_kalman(model_path, video_path, output_path=None, img_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LaneKeypointNet(num_points=TOTAL_POINTS)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    colors = [(0, 255, 0)]  # Só uma linha central
    kalman_filters = None  # Para inicializar depois do primeiro frame

    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            original_h, original_w = frame.shape[:2]
            img = cv2.resize(frame, (img_size, img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.transpose(2, 0, 1) / 255.0
            img = torch.from_numpy(img).float().unsqueeze(0).to(device)

            pred_keypoints = model(img).cpu().numpy()[0]
            pred_points = pred_keypoints.reshape(-1, 2)
            pred_points[:, 0] *= original_w
            pred_points[:, 1] *= original_h

            # === Aplicar Kalman ===
            if kalman_filters is None:
                kalman_filters = []
                for x, y in pred_points:
                    kalman_filters.append(Kalman2D(x, y))

            smoothed_points = []
            for i, (x, y) in enumerate(pred_points):
                kalman_filters[i].correct(x, y)
                smoothed_x, smoothed_y = kalman_filters[i].predict()
                smoothed_points.append((smoothed_x, smoothed_y))

            lane_points = np.array(smoothed_points, dtype=np.int32)

            # === Desenhar ===
            for i in range(len(lane_points) - 1):
                p1, p2 = tuple(lane_points[i]), tuple(lane_points[i + 1])
                cv2.line(frame, p1, p2, colors[0], 3)

            for point in lane_points:
                cv2.circle(frame, tuple(point), 5, colors[0], -1)

            cv2.imshow('Lane Detection with Kalman', frame)

            if output_path:
                out.write(frame)

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

def calculate_steering(points):
    """
    Calcula o ângulo e steering normalizado (-1 a 1) usando pontos 3 e 4.
    """
    if len(points) < 4:
        return 0.0, 0.0

    p1 = points[3]
    p2 = points[4]

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    if dx == 0 and dy == 0:
        return 0.0, 0.0

    angle_rad = np.arctan2(dy, dx)
    center_angle = np.pi / 2  # Para baixo

    angle_offset = angle_rad - center_angle

    # Normalizar steering (-1 a 1)
    max_offset = np.radians(45)  # Até 45 graus de desvio
    steering = np.clip(angle_offset / max_offset, -1, 1)

    return angle_rad, steering

def draw_steering_arrow(frame, point1, point2, angle_rad, length=100, color=(255, 0, 0)):
    """
    Desenha seta a partir do ponto médio entre point1 e point2.
    """
    x0 = int((point1[0] + point2[0]) / 2)
    y0 = int((point1[1] + point2[1]) / 2)
    x1 = int(x0 + length * np.cos(angle_rad))
    y1 = int(y0 + length * np.sin(angle_rad))

    cv2.arrowedLine(frame, (x0, y0), (x1, y1), color, 4)

def draw_fitted_line_and_angle(frame, points):
    """Ajusta uma reta aos pontos e desenha uma seta com o steering."""

    # Ajustar reta usando polyfit (reta simples)
    x = points[:, 0]
    y = points[:, 1]

    # polyfit pode falhar se todos x forem iguais (linha vertical), proteger:
    if np.allclose(x, x[0]):
        slope = float('inf')
        intercept = 0
    else:
        slope, intercept = np.polyfit(x, y, 1)

    # Centro da imagem
    h, w = frame.shape[:2]
    center_x = w // 2
    center_y = h

    # Ponto base da seta (posição vertical perto do carro)
    y0 = int(h * 0.8)
    x0 = int((y0 - intercept) / slope) if slope != float('inf') else int(x.mean())

    # Outro ponto na direção da reta (para desenhar a seta)
    y1 = int(h * 0.5)
    x1 = int((y1 - intercept) / slope) if slope != float('inf') else int(x.mean())

    # Desenhar seta
    cv2.arrowedLine(frame, (x0, y0), (x1, y1), (0, 0, 255), 4, tipLength=0.2)

    # Calcular ângulo da reta em relação ao eixo vertical
    dx = x1 - x0
    dy = y0 - y1  # y invertido porque cresce para baixo nas imagens

    angle = np.arctan2(dx, dy)  # atan2(x, y) para ângulo em relação ao eixo vertical

    # Normalizar o steering entre -1 e 1
    steering = angle / (np.pi / 2)
    steering = np.clip(steering, -1, 1)

    # Mostrar valor do steering
    cv2.putText(frame, f"Steering: {steering:.2f}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame, steering

def inference_video_with_steering(model_path, video_path, output_path=None, img_size=256, use_kalman=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LaneKeypointNet(num_points=TOTAL_POINTS)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Erro ao abrir o vídeo: {video_path}")
        return

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    else:
        out = None

    frame_count = 0

    # Cores
    color = (0, 255, 0)  # Verde para os pontos e linhas
    kalman_filters = None

    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            original_h, original_w = frame.shape[:2]
            img = cv2.resize(frame, (img_size, img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.transpose(2, 0, 1) / 255.0
            img = torch.from_numpy(img).float().unsqueeze(0).to(device)

            pred_keypoints = model(img).cpu().numpy()[0]
            pred_points = pred_keypoints.reshape(-1, 2)
            pred_points[:, 0] *= original_w
            pred_points[:, 1] *= original_h

            # === Kalman ===
            if use_kalman:
                if kalman_filters is None:
                    kalman_filters = []
                    for x, y in pred_points:
                        kalman_filters.append(Kalman2D(x, y))
                smoothed_points = []
                for i, (x, y) in enumerate(pred_points):
                    kalman_filters[i].correct(x, y)
                    smoothed_x, smoothed_y = kalman_filters[i].predict()
                    smoothed_points.append((smoothed_x, smoothed_y))
                pred_points = np.array(smoothed_points, dtype=np.float32)

            # === Desenhar os pontos e linhas ===
            for i in range(len(pred_points) - 1):
                p1, p2 = tuple(pred_points[i]), tuple(pred_points[i + 1])
                cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 2)

            for point in pred_points:
                cv2.circle(frame, (int(point[0]), int(point[1])), 5, color, -1)

            # === Ajustar reta e desenhar seta com steering ===
            frame, steering = draw_fitted_line_and_angle(frame, pred_points)

            cv2.imshow('Lane Detection', frame)

            if out:
                out.write(frame)

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    logger.info(f"Processados {frame_count} frames")



# ===== Suavização exponencial =====
def exponential_smoothing(new_points, prev_points, alpha=0.3):
    if prev_points is None:
        return new_points.copy()
    return alpha * new_points + (1 - alpha) * prev_points

# ===== Fit de reta e cálculo do ângulo =====
def compute_line_angle(points):
    xs = points[:, 0]
    ys = points[:, 1]
    fit = np.polyfit(xs, ys, 1)
    slope = fit[0]
    angle_rad = np.arctan(slope)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# ===== Converter ângulo em steering =====
def angle_to_steering(angle_deg, dead_zone=5):
    if abs(angle_deg) < dead_zone:
        return 0
    max_angle = 45
    steering = np.clip(angle_deg / max_angle, -1, 1)
    return steering

# ===== Desenhar seta =====
def draw_direction_arrow(frame, line_points, angle_deg):
    p3 = line_points[2]
    p4 = line_points[3]
    center = ((p3 + p4) / 2).astype(int)
    length = 80
    angle_rad = -np.radians(angle_deg)
    end_x = int(center[0] + length * np.cos(angle_rad))
    end_y = int(center[1] - length * np.sin(angle_rad))
    cv2.arrowedLine(frame, tuple(center), (end_x, end_y), (0, 0, 255), 3)

# ===== Função principal de inferência =====
def inference_video_direction(model_path, video_path, output_path=None,img_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from main import LaneKeypointNet, NUM_LANES, POINTS_PER_LANE

    TOTAL_POINTS = 12

    model = LaneKeypointNet(num_points=TOTAL_POINTS)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return

    prev_points = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        original_h, original_w = frame.shape[:2]
        img = cv2.resize(frame, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1) / 255.0
        img = torch.from_numpy(img).float().unsqueeze(0).to(device)

        with torch.no_grad():
            pred_keypoints = model(img).cpu().numpy()[0]

        pred_points = pred_keypoints.reshape(-1, 2)
        pred_points[:, 0] *= original_w
        pred_points[:, 1] *= original_h

        # Clipping
        pred_points[:, 0] = np.clip(pred_points[:, 0], 0, original_w)
        pred_points[:, 1] = np.clip(pred_points[:, 1], 0, original_h)

        # Suavização
        pred_points = exponential_smoothing(pred_points, prev_points)
        prev_points = pred_points.copy()

        # Ângulo
        angle = compute_line_angle(pred_points)
        steering = angle_to_steering(angle)

        # Desenho
        for i in range(len(pred_points) - 1):
            p1 = tuple(pred_points[i].astype(int))
            p2 = tuple(pred_points[i + 1].astype(int))
            cv2.line(frame, p1, p2, (0, 255, 0), 2)

        for point in pred_points:
            cv2.circle(frame, tuple(point.astype(int)), 5, (0, 255, 0), -1)

        draw_direction_arrow(frame, pred_points, angle)

        # Texto
        cv2.putText(frame, f"Angle: {angle:.1f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Steering: {steering:.2f}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Direction Prediction', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
 
    args = parse_args()
 
 
    
    #train(args)

    inference_video_direction(
         model_path='output/best_model.pth',
         video_path='../centro.mp4',
         output_path='estrada_processado.mp4'
     )
