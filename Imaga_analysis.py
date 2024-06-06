#Image Analysis CIFAR10 data (C) 2023 Bashir Hussein

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, datasets, models

# Load CIFAR-10 dataset
(train_images_all, train_labels_all), (test_images_all, test_labels_all) = datasets.cifar10.load_data()

# Define class names and select indices
class_names = ['airplane', 'automobile', 'bird']
selected_indices = [0, 1, 2]  # Indices for 'airplane', 'automobile', and 'bird' in the CIFAR-10 dataset

# Select data for desired classes
train_indices = np.isin(train_labels_all, selected_indices).reshape(-1)
test_indices = np.isin(test_labels_all, selected_indices).reshape(-1)

train_images, train_labels = train_images_all[train_indices], train_labels_all[train_indices]
test_images, test_labels = test_images_all[test_indices], test_labels_all[test_indices]

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

num_classes = len(class_names)
img_height, img_width = 32, 32

# Data augmentation
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

# Build the model
model = models.Sequential([
    data_augmentation,
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(class_names))
])

# Model summary
model.summary()

# Define EarlyStopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True
)

# Compile the model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train the model with early stopping
history = model.fit(
    train_images, train_labels,
    epochs=200,
    validation_data=(test_images, test_labels),
    callbacks=[early_stopping]
)

# Get the epoch with the best validation loss
best_epoch = np.argmin(history.history['val_loss']) + 1
print("Epoch with the best validation loss:", best_epoch)

# Extract training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot the training and validation curves
epochs_used = len(history.history['loss'])
epochs_range = range(epochs_used)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model to a .tflite file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

if __name__ == '__main__':
    print(test_acc)
