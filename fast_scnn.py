import os
import cv2
import numpy as np
import tensorflow as tf
from glob import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==== CONFIG ====
IMAGE_DIR = 'images'
MASK_DIR = 'masks'
IMG_SIZE = (256, 256)
BATCH_SIZE = 8
EPOCHS = 50

# ==== FUNÇÕES AUXILIARES ====
def load_image_mask(image_path, mask_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, IMG_SIZE)
    mask = (mask > 5).astype(np.float32)  # binário
    mask = np.expand_dims(mask, axis=-1)
    return img, mask

def data_generator(image_paths, mask_paths, augment=False):
    while True:
        idxs = np.arange(len(image_paths))
        np.random.shuffle(idxs)

        for i in range(0, len(image_paths), BATCH_SIZE):
            batch_idxs = idxs[i:i+BATCH_SIZE]
            imgs = []
            masks = []
            for j in batch_idxs:
                img, mask = load_image_mask(image_paths[j], mask_paths[j])

                if augment and np.random.rand() > 0.5:
                    img = np.fliplr(img)
                    mask = np.fliplr(mask)

                imgs.append(img)
                masks.append(mask)

            yield np.array(imgs), np.array(masks)

# ==== FAST-SCNN MODEL ====
def FastSCNN(input_shape=(256, 256, 3)):
    base = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                             include_top=False,
                                             weights=None)
    x = base.output
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.UpSampling2D(4)(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.UpSampling2D(4)(x)
    x = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x) 

    model = tf.keras.Model(inputs=base.input, outputs=x)
    return model

# ==== PREPARAR DADOS ====
img_paths = sorted(glob(os.path.join(IMAGE_DIR, '*')))
mask_paths = sorted(glob(os.path.join(MASK_DIR, '*')))

assert len(img_paths) == len(mask_paths), "Número de imagens e máscaras não coincide."

train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    img_paths, mask_paths, test_size=0.2, random_state=42
)

train_gen = data_generator(train_imgs, train_masks, augment=True)
val_gen = data_generator(val_imgs, val_masks, augment=False)

steps_per_epoch = len(train_imgs) // BATCH_SIZE
validation_steps = len(val_imgs) // BATCH_SIZE

# ==== COMPILAR MODELO ====
model = FastSCNN(input_shape=(256, 256, 3))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])

model.summary()

# ==== CALLBACKS ====
callbacks = [
    tf.keras.callbacks.ModelCheckpoint('fast_scnn_best.keras', save_best_only=True, monitor='val_loss'),
    tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True)
]

# ==== TREINO ====
history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==== PLOT CURVAS ====
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Curva de Loss')
plt.savefig('loss_curve.png')
plt.show()

