import tensorflow as tf
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from SimpleGenerator import train_generator, val_generator

# Load pre-trained model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(5, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base layers and train only the top layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model (assuming you have a dataset generator)
model.fit(train_generator, epochs=10, validation_data=val_generator)

model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
)

import json
# Save class indices
with open("class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f)
