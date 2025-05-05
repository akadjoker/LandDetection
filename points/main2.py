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
        if img is None:
            # Handle corrupt images
            logger.warning(f"Could not read image {img_path}")
            # Return a zero image and zero points as fallback
            img = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
            keypoints = np.zeros(TOTAL_POINTS, dtype=np.float32)
            return torch.zeros(3, *self.img_size).float(), torch.from_numpy(keypoints).float()
            
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load keypoints
        keypoints = self.load_labels(label_path)
        
        # Apply augmentations
        if self.augment:
            # Horizontal flip with 50% probability
            if random.random() > 0.5:
                img = np.fliplr(img).copy()
                # Adjust x coordinates (horizontal reflection)
                for i in range(0, len(keypoints), 2):
                    keypoints[i] = 1.0 - keypoints[i]  # invert x (already normalized)
            
            # Random brightness/contrast adjustment
            if random.random() > 0.5:
                alpha = random.uniform(0.7, 1.3)  # contrast
                beta = random.uniform(-30, 30)    # brightness
                img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
            

            

            
            # Color jitter using transform_augment
            img_tensor = torch.from_numpy(img).permute(2,0,1).float()/255.0
            img_tensor = transform_augment(img_tensor)
            img = (img_tensor * 255).permute(1,2,0).byte().numpy()
        
        # Normalize image
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        keypoints = torch.from_numpy(keypoints).float()
        
        return img, keypoints

class CombinedLoss(nn.Module):
    def __init__(self, smoothness_weight=0.2):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.smoothness_weight = smoothness_weight
        
    def forward(self, pred, target):
        # Basic MSE loss for position accuracy
        position_loss = self.mse_loss(pred, target)
        
        # Smoothness loss to ensure curve continuity
        pred_reshaped = pred.view(-1, POINTS_PER_LANE, 2)
        
        # Calculate differences between consecutive points (for smoothness)
        pred_diff = pred_reshaped[:, 1:] - pred_reshaped[:, :-1]
        target_reshaped = target.view(-1, POINTS_PER_LANE, 2)
        target_diff = target_reshaped[:, 1:] - target_reshaped[:, :-1]
        
        # Smoothness loss
        smoothness_loss = self.mse_loss(pred_diff, target_diff)
        
        # Combined loss
        total_loss = position_loss + self.smoothness_weight * smoothness_loss
        
        return total_loss

# ==== MODEL ====
class ImprovedLaneKeypointNet(nn.Module):
    def __init__(self, num_points=TOTAL_POINTS):
        super(ImprovedLaneKeypointNet, self).__init__()
        
        # Encoder com ResNet-like blocks
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet-like blocks
        self.res_block1 = self._make_res_block(32, 64)
        self.res_block2 = self._make_res_block(64, 128)
        self.res_block3 = self._make_res_block(128, 256)
        
        # Attention module
        self.attention = self._make_attention_module(256)
        
        # Test output size dynamically
        dummy_input = torch.zeros(1, 3, 256, 256)
        x = self.conv1(dummy_input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.attention(x)
        
        cnn_output_size = x.numel()
        
        # Regressor with dropout for better generalization
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_output_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_points)
        )
    
    def _make_res_block(self, in_channels, out_channels):
        return nn.Sequential(
            # First conv layer
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Second conv layer
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_attention_module(self, channels):
        return nn.Sequential(
            # Spatial attention
            nn.Conv2d(channels, channels//8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//8, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        # Apply attention
        attn = self.attention(x)
        x = x * attn
        
        # Regressor
        keypoints = self.regressor(x)
        
        # Apply sigmoid to constrain to [0,1]
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

def mixup_data(x, y, alpha=0.2):
    """Create mixed samples and targets for mixup regularization"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y

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
    model = ImprovedLaneKeypointNet(num_points=TOTAL_POINTS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.1
    )
    loss_fn = CombinedLoss(smoothness_weight=0.2)
    
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

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step() 
            
            # Update metrics
            train_loss += loss.item()
            
            # Update progress bar
            train_loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        
        
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


def inference_video(model_path, video_path, output_path=None, img_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LaneKeypointDataset(num_points=TOTAL_POINTS)
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
    alpha = 0.3  # fator de suavização (quanto menor, mais suave)

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


def inference_video2(model_path, video_path, output_path=None, img_size=256):
    """
    Realiza detecção de faixas em um vídeo usando o modelo treinado
    
    Args:
        model_path: Caminho para o modelo salvo
        video_path: Caminho para o vídeo de entrada
        output_path: Caminho para salvar o vídeo processado (opcional)
        img_size: Tamanho da imagem para processamento
    """
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

if __name__ == "__main__":
 
    args = parse_args()
 
 
    
    train(args)

    inference_video(
         model_path='output/best_model.pth',
         video_path='../estrada.mp4',
         output_path='estrada_processado.mp4'
     )
