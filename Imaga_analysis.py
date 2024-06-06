#Image Analysis CIFAR10 data (C) 2023 Bashir Hussein

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import datasets

# Retrieve the cifar10 data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Set the indices for 'airplane', 'automobile', and 'bird' in the CIFAR-10 dataset
selected_indices = [0, 1, 2]

# Select only the data for the desired classes
train_indices = np.isin(train_labels, selected_indices).reshape(-1)
test_indices = np.isin(test_labels, selected_indices).reshape(-1)
train_images, train_labels = train_images[train_indices], train_labels[train_indices]
test_images, test_labels = test_images[test_indices], test_labels[test_indices]

# Set height and width of the images
img_height = 32
img_width = 32

# Apply data augmentation
data_augmentation = keras.Sequential(
[
layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
layers.experimental.preprocessing.RandomFlip("horizontal"),
layers.experimental.preprocessing.RandomRotation(0.1),
layers.experimental.preprocessing.RandomZoom(0.1),
]
)

# Create the model
model = Sequential()
model.add(data_augmentation)
model.add(layers.Conv2D(64, kernel_size=(3,3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(len(selected_indices), activation='softmax'))


# Define EarlyStopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
monitor='val_loss',
patience=8,
restore_best_weights=True
)

# Compile the model
model.compile(
optimizer='adam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
metrics=['accuracy']
)

epochs = 50

# Train the model with early stopping
history = model.fit(train_images, train_labels,
epochs=epochs,
validation_data=(test_images, test_labels),
callbacks=[early_stopping])

# Get the epoch with the best validation loss
best_epoch = np.argmin(history.history['val_loss']) + 1
print("Epoch with the best validation loss:", best_epoch)

# Update the epochs variable to reflect the actual number of epochs used
epochs_used = len(history.history['loss'])
epochs_range = range(epochs_used)

# Create plots of the loss and accuracy on the training and validation sets
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

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

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)