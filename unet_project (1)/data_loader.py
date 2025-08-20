import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# Paths
IMAGE_DIR = '/kaggle/input/tttttttttt/images_train/images_train'
MASK_DIR = '//kaggle/input/cccccc/output_masks'
IMG_SIZE = 720

# Load file paths
image_paths = sorted(glob(os.path.join(IMAGE_DIR, '*.tif')))
mask_paths = sorted(glob(os.path.join(MASK_DIR, '*.png')))

# Preprocess function
def preprocess_image(image_path, mask_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    mask = (cv2.resize(mask, (IMG_SIZE, IMG_SIZE)) > 127).astype(np.float32)
    return np.expand_dims(image, -1), np.expand_dims(mask, -1)

# Visualize
i = 0
image, mask = preprocess_image(image_paths[i], mask_paths[i])
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(image.squeeze(), cmap='gray')
plt.title("Image")
plt.subplot(1, 2, 2)
plt.imshow(mask.squeeze(), cmap='gray')
plt.title("Mask")
plt.show()


import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

# Albumentations augmentation pipeline
train_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
])

# Function to preprocess and optionally augment an image and mask
def preprocess_image_and_mask(image_path, mask_path, augment=False):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

    image = image.astype(np.float32) / 255.0
    mask = (mask > 127).astype(np.float32)

    if augment:
        augmented = train_aug(image=image, mask=mask)
        image, mask = augmented['image'], augmented['mask']

    # Expand dims to add channel dimension (H, W, 1)
    image = np.expand_dims(image, axis=-1)
    mask = np.expand_dims(mask, axis=-1)

    return image, mask

# Data generator function
def data_generator(image_paths, mask_paths, augment=False):
    while True:
        idxs = np.random.permutation(len(image_paths))
        for i in range(0, len(image_paths), BATCH_SIZE):
            batch_idxs = idxs[i:i + BATCH_SIZE]
            batch_images = []
            batch_masks = []

            for j in batch_idxs:
                img, msk = preprocess_image_and_mask(image_paths[j], mask_paths[j], augment)
                batch_images.append(img)
                batch_masks.append(msk)

            yield np.array(batch_images), np.array(batch_masks)

# --- Visualization of original vs augmented sample ---
# Load one sample image and mask
sample_image, sample_mask = preprocess_image_and_mask(image_paths[0], mask_paths[0], augment=False)

# Apply augmentation for visualization
augmented_sample = train_aug(image=sample_image.squeeze(), mask=sample_mask.squeeze())

# Convert back to HxWx1 for visualization consistency
aug_img = np.expand_dims(augmented_sample['image'], axis=-1)
aug_msk = np.expand_dims(augmented_sample['mask'], axis=-1)

# Plot
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(sample_image.squeeze(), cmap='gray')
plt.title("Original Image")

plt.subplot(1, 3, 2)
plt.imshow(aug_img.squeeze(), cmap='gray')
plt.title("Augmented Image")

plt.subplot(1, 3, 3)
plt.imshow(aug_msk.squeeze(), cmap='gray')
plt.title("Augmented Mask")

plt.show()


