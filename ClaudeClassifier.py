import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os

# Aggressive data augmentation for small dataset
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,          # Rotate images up to 40 degrees
    width_shift_range=0.3,      # Shift horizontally up to 30%
    height_shift_range=0.3,     # Shift vertically up to 30%
    shear_range=0.3,           # Shear transformation
    zoom_range=0.3,            # Zoom in/out up to 30%
    horizontal_flip=True,       # Flip horizontally
    vertical_flip=True,         # Flip vertically (if makes sense for your data)
    brightness_range=[0.7, 1.3], # Adjust brightness
    fill_mode='nearest',        # Fill missing pixels
    validation_split=0.2        # Use 20% for validation if needed
)

# Validation data should only be rescaled
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Create data generators
def create_generators(data_dir, batch_size=8):
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    return train_generator, val_generator

# Enhanced model with dropout for regularization
def create_model(num_classes=3):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)  # Add dropout for regularization
    x = Dense(512, activation='relu')(x)  # Smaller dense layer
    x = Dropout(0.3)(x)  # More dropout
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze base model initially
    for layer in base_model.layers:
        layer.trainable = False
    
    return model, base_model

# Training strategy for small datasets
def train_small_dataset(data_dir, num_classes=3):
    # Create model
    model, base_model = create_model(num_classes)
    
    # Compile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create generators
    train_generator, val_generator = create_generators(data_dir, batch_size=4)
    
    # Callbacks for better training
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7, monitor='val_loss')
    ]
    
    # Phase 1: Train only top layers
    print("Phase 1: Training top layers only...")
    history1 = model.fit(
        train_generator,
        epochs=30,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-tune some layers
    print("Phase 2: Fine-tuning...")
    # Unfreeze top layers of base model
    for layer in base_model.layers[-20:]:
        layer.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        train_generator,
        epochs=20,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, train_generator

# Cross-validation approach for very small datasets
def cross_validation_training(images, labels, num_classes=3, k_folds=5):
    """
    Use this if you want to implement k-fold cross-validation
    images: array of image paths
    labels: array of corresponding labels
    """
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
        print(f"Training fold {fold + 1}/{k_folds}")
        
        # Create model for this fold
        model, _ = create_model(num_classes)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Split data for this fold
        train_images, val_images = images[train_idx], images[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]
        
        # Create generators for this fold (implement your own generator here)
        # train_gen = create_fold_generator(train_images, train_labels)
        # val_gen = create_fold_generator(val_images, val_labels)
        
        # Train model
        # history = model.fit(train_gen, validation_data=val_gen, epochs=30)
        
        # cv_scores.append(max(history.history['val_accuracy']))
    
    return np.mean(cv_scores), np.std(cv_scores)

# Usage example
if __name__ == "__main__":
    # Assuming your data is organized as:
    # data/
    #   ├── table1/
    #   ├── table2/
    #   └── table3/
    
    data_directory = "./data/"
    model, train_gen = train_small_dataset(data_directory, num_classes=3)
    
    # Save the model
    model.save("table_classifier.h5")
    
    # Save class indices
    import json
    with open("class_indices.json", "w") as f:
        json.dump(train_gen.class_indices, f)