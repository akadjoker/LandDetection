import os
import cv2
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import random
from tqdm import tqdm
import logging

# Configure GPU memory growth to avoid OOM errors
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"Found {len(physical_devices)} GPU(s)")
else:
    print("No GPU found, using CPU")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_tf.log')
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train TensorFlow model for line detection on Jetson Nano')
    parser.add_argument('--image_dir', type=str, default='../images', help='Directory with input images')
    parser.add_argument('--mask_dir', type=str, default='../masks', help='Directory with mask images')
    parser.add_argument('--img_size', type=int, default=64, help='Image size (square)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='output_tf', help='Output directory')
    parser.add_argument('--model_type', type=str, default='mobilenet', 
                        choices=['unet', 'mobilenet', 'lite'], help='Model architecture')
    parser.add_argument('--quantize', action='store_true', help='Apply post-training quantization')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    return parser.parse_args()

# Data generator
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, img_paths, mask_paths, batch_size=8, img_size=(64, 64), augment=False):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.indexes = np.arange(len(self.img_paths))
        
    def __len__(self):
        return int(np.ceil(len(self.img_paths) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_img_paths = [self.img_paths[i] for i in batch_indexes]
        batch_mask_paths = [self.mask_paths[i] for i in batch_indexes]
        
        X = np.zeros((len(batch_indexes), self.img_size[0], self.img_size[1], 3), dtype=np.float32)
        y = np.zeros((len(batch_indexes), self.img_size[0], self.img_size[1], 1), dtype=np.float32)
        
        for i, (img_path, mask_path) in enumerate(zip(batch_img_paths, batch_mask_paths)):
            # Load image
            img = cv2.imread(img_path)
            img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Load mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.float32)
            
            # Simple augmentation
            if self.augment and np.random.random() > 0.5:
                # Horizontal flip
                img = np.fliplr(img)
                mask = np.fliplr(mask)
            
            # Normalize
            X[i] = img / 255.0
            y[i] = np.expand_dims(mask, axis=-1)
            
        return X, y
    
    def on_epoch_end(self):
        if self.augment:
            np.random.shuffle(self.indexes)

# Metrics
def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def combined_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

# IoU metric
def iou_score(y_true, y_pred, smooth=1.0):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred > 0.5, dtype=tf.float32))
    
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    
    return (intersection + smooth) / (union + smooth)

# Model architectures
def build_unet_model(img_size, channels=3):
    """Build a basic U-Net model"""
    inputs = layers.Input((img_size, img_size, channels))
    
    # Encoder
    c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    
    # Decoder
    u5 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u5)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    u6 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
    u7 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c7)
    
    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

def build_mobilenet_unet(img_size, channels=3):
    """Build a MobileNet-based U-Net model (lighter and faster)"""
    inputs = layers.Input((img_size, img_size, channels))
    
    # Use MobileNetV2 as encoder (without top layers)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, channels),
        include_top=False,
        weights=None,  # No pre-trained weights for faster training and smaller model
    )
    
    # Use specific layers from base model
    skip_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
    ]
    
    # Get the output from skip connections
    skip_outputs = [base_model.get_layer(name).output for name in skip_names]
    base_output = base_model.get_layer('block_16_project').output
    
    # Decoder path
    x = layers.Conv2DTranspose(96, (4, 4), strides=(2, 2), padding='same')(base_output)  # 8x8 -> 16x16
    x = layers.concatenate([x, skip_outputs[3]])
    x = layers.Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    
    x = layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same')(x)  # 16x16 -> 32x32
    x = layers.concatenate([x, skip_outputs[2]])
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    x = layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same')(x)  # 32x32 -> 64x64
    x = layers.concatenate([x, skip_outputs[0]])
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    
    # Output
    x = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=x)
    return model

def build_lite_model(img_size, channels=3):
    """Build an extremely lightweight model for Jetson Nano"""
    inputs = layers.Input((img_size, img_size, channels))
    
    # Encoder (very simplified)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    skip1 = x
    x = layers.MaxPooling2D((2, 2))(x)  # 64x64 -> 32x32
    
    # Use depthwise separable convolution for efficiency
    x = layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    skip2 = x
    x = layers.MaxPooling2D((2, 2))(x)  # 32x32 -> 16x16
    
    x = layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 16x16 -> 8x8
    
    # Bottleneck
    x = layers.SeparableConv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Decoder
    x = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2))(x)  # 8x8 -> 16x16
    
    x = layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2))(x)  # 16x16 -> 32x32
    x = layers.concatenate([x, skip2])
    
    x = layers.SeparableConv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2))(x)  # 32x32 -> 64x64
    x = layers.concatenate([x, skip1])
    
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    # Output
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    
    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model
