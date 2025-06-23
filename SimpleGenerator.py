import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os

# Path to your dataset
DATA_DIR = "/media/training_data/"
IMG_SIZE = (224, 224)  # MobileNetV2 default input size
BATCH_SIZE = 32

# Augmentation settings (modify as needed)
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values
    rotation_range=20,  # Random rotation (degrees)
    width_shift_range=0.1,  # Horizontal shift
    height_shift_range=0.1,  # Vertical shift
    shear_range=0.1,  # Shear transformations
    zoom_range=0.1,  # Random zoom
    horizontal_flip=True,  # Random horizontal flip
    brightness_range=[0.9, 1.1],  # Slight brightness changes
    validation_split=0.2,  # 80/20 train-val split
)

# Train generator
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",  # For train/val split
    shuffle=True,
)

# Validation generator (no augmentation, just rescaling)
val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
val_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
)

# Print class indices (e.g., {"table_1": 0, "table_2": 1, ...})
print("Class labels:", train_generator.class_indices)